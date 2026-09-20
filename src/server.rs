//! A bounded, concurrent text Chat Completions server.

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tokio_util::sync::CancellationToken;

use crate::{
    Model,
    generation::{Completion, Generator, Options, Sampling},
    tokenizer::{ChatMessage, Role},
};

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model_id: String,
    pub context_size: usize,
    pub max_concurrent_requests: usize,
    pub queue_capacity: usize,
    pub threads: usize,
}

struct App {
    model: Arc<Model>,
    config: ServerConfig,
    slots: Arc<Semaphore>,
    admission: Arc<Semaphore>,
    cpu: Arc<ThreadPool>,
    created: u64,
    shutdown: CancellationToken,
}

pub fn router(model: Arc<Model>, config: ServerConfig) -> Result<Router> {
    router_with_shutdown(model, config, CancellationToken::new())
}

pub fn router_with_shutdown(
    model: Arc<Model>,
    config: ServerConfig,
    shutdown: CancellationToken,
) -> Result<Router> {
    ensure!(!config.model_id.is_empty(), "model ID must not be empty");
    ensure!(
        config.max_concurrent_requests > 0 && config.threads > 0,
        "concurrency and threads must be positive"
    );
    model.new_session(config.context_size)?;
    let capacity = config
        .max_concurrent_requests
        .checked_add(config.queue_capacity)
        .context("queue capacity overflow")?;
    ensure!(
        capacity <= Semaphore::MAX_PERMITS,
        "queue capacity is too large"
    );
    let cpu = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .thread_name(|n| format!("hy-cpu-{n}"))
        .build()?;
    let app = Arc::new(App {
        slots: Arc::new(Semaphore::new(config.max_concurrent_requests)),
        admission: Arc::new(Semaphore::new(capacity)),
        model,
        config,
        cpu: Arc::new(cpu),
        created: unix_time(),
        shutdown,
    });
    Ok(Router::new()
        .route("/healthz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .fallback(|| async {
            ApiError::new(StatusCode::NOT_FOUND, "not_found", "Unknown endpoint")
        })
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(app))
}

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"status": "ok", "model": app.config.model_id}))
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": app.config.model_id, "object": "model", "created": app.created, "owned_by": "local"}]}),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    stop: Option<Stop>,
    n: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: Role,
    content: Content,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<TextPart>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextPart {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

struct Prepared {
    prompt: Vec<u32>,
    options: Options,
    stream: bool,
    include_usage: bool,
}

impl ChatRequest {
    fn prepare(self, app: &App) -> Result<Prepared> {
        ensure!(self.n.unwrap_or(1) == 1, "only n=1 is supported");
        ensure!(
            self.max_tokens.is_none() || self.max_completion_tokens.is_none(),
            "provide only one of max_tokens and max_completion_tokens"
        );
        ensure!(
            self.stream || self.stream_options.is_none(),
            "stream_options requires stream=true"
        );
        let messages: Vec<_> = self
            .messages
            .into_iter()
            .map(|message| {
                let content = match message.content {
                    Content::Text(text) => text,
                    Content::Parts(parts) => {
                        ensure!(
                            parts.iter().all(|p| p.kind == "text"),
                            "only text content parts are supported"
                        );
                        parts.into_iter().map(|p| p.text).collect::<String>()
                    }
                };
                Ok(ChatMessage {
                    role: message.role,
                    content,
                })
            })
            .collect::<Result<_>>()?;
        let prompt = app.model.tokenizer.encode_chat(&messages)?;
        ensure!(
            !prompt.is_empty() && prompt.len() < app.config.context_size,
            "prompt leaves no output space in the configured context"
        );
        let available = app.config.context_size - prompt.len();
        let max_tokens = self
            .max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(4096.min(available));
        ensure!(
            max_tokens <= available,
            "prompt plus output limit exceeds the configured context size"
        );
        let mut sampling = Sampling::for_model(app.model.config.architecture);
        if let Some(x) = self.temperature {
            sampling.temperature = x;
        }
        if let Some(x) = self.top_p {
            sampling.top_p = x;
        }
        if let Some(x) = self.top_k {
            sampling.top_k = x;
        }
        if let Some(x) = self.repetition_penalty {
            sampling.repetition_penalty = x;
        }
        sampling.seed = self.seed;
        let stop = match self.stop {
            None => vec![],
            Some(Stop::One(x)) => vec![x],
            Some(Stop::Many(xs)) => xs,
        };
        let options = Options {
            max_tokens,
            sampling,
            stop,
        };
        options.validate()?;
        Ok(Prepared {
            prompt,
            options,
            stream: self.stream,
            include_usage: self.stream_options.is_some_and(|o| o.include_usage),
        })
    }
}

async fn chat(
    State(app): State<Arc<App>>,
    request: std::result::Result<Json<ChatRequest>, JsonRejection>,
) -> std::result::Result<Response, ApiError> {
    let Json(request) = request.map_err(|e| {
        let status = if e.status() == StatusCode::UNPROCESSABLE_ENTITY {
            StatusCode::BAD_REQUEST
        } else {
            e.status()
        };
        ApiError::new(status, "invalid_request_error", e.body_text())
    })?;
    if request.model != app.config.model_id {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("Model {:?} is not loaded", request.model),
        ));
    }
    let admission = app.admission.clone().try_acquire_owned().map_err(|_| {
        ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "Request queue is full",
        )
    })?;
    let cancel = app.shutdown.child_token();
    let guard = cancel.clone().drop_guard();
    let prepare_app = app.clone();
    // Hold the admission permit even if the client leaves during tokenization.
    let (prepared, admission) = tokio::task::spawn_blocking(move || {
        let prepared = prepare_app.cpu.install(|| request.prepare(&prepare_app));
        prepared.map(|p| (p, admission))
    })
    .await
    .map_err(ApiError::internal)?
    .map_err(ApiError::invalid)?;
    let active = app
        .slots
        .clone()
        .acquire_owned()
        .await
        .map_err(ApiError::internal)?;
    let id = request_id();
    let created = unix_time();
    if !prepared.stream {
        let job_app = app.clone();
        let completion = tokio::task::spawn_blocking(move || {
            let (_admission, _active) = (admission, active);
            Generator::with_pool(&job_app.model, &job_app.cpu).generate(
                &prepared.prompt,
                job_app.config.context_size,
                &prepared.options,
                &cancel,
                |_| Ok(()),
            )
        })
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::internal)?;
        log_completion(&id, &completion);
        let response = Json(json!({
            "id": id, "object": "chat.completion", "created": created, "model": app.config.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": completion.text}, "finish_reason": completion.finish_reason}],
            "usage": completion.usage,
        }));
        drop(guard);
        return Ok(response.into_response());
    }
    let (sender, receiver) = mpsc::channel::<Event>(8);
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let (_admission, _active) = (admission, active);
        let send_event = |event: Event| -> Result<()> {
            runtime.block_on(async {
                tokio::select! {
                    _ = cancel.cancelled() => anyhow::bail!("request cancelled"),
                    result = sender.send(event) => result.context("stream receiver closed"),
                }
            })
        };
        let send = |value: Value| -> Result<()> { send_event(Event::default().json_data(value)?) };
        let chunk = |choices: Value, usage: Option<Value>| {
            let mut value = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": app.config.model_id, "choices": choices});
            if prepared.include_usage {
                value["usage"] = usage.unwrap_or(Value::Null);
            }
            value
        };
        let result = (|| -> Result<Completion> {
            send(chunk(
                json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]),
                None,
            ))?;
            Generator::with_pool(&app.model, &app.cpu).generate(
                &prepared.prompt,
                app.config.context_size,
                &prepared.options,
                &cancel,
                |text| {
                    send(chunk(
                        json!([{"index": 0, "delta": {"content": text}, "finish_reason": null}]),
                        None,
                    ))
                },
            )
        })();
        match result {
            Ok(completion) => {
                log_completion(&id, &completion);
                if send(chunk(
                    json!([{"index": 0, "delta": {}, "finish_reason": completion.finish_reason}]),
                    None,
                ))
                .is_err()
                {
                    return;
                }
                if prepared.include_usage
                    && send(chunk(json!([]), Some(json!(completion.usage)))).is_err()
                {
                    return;
                }
                let _ = send_event(Event::default().data("[DONE]"));
            }
            Err(error) => {
                if !cancel.is_cancelled() {
                    tracing::error!(request_id = id, error = %error, "generation failed");
                    let _ = send(
                        json!({"error": {"message": error.to_string(), "type": "server_error", "param": null, "code": null}}),
                    );
                }
            }
        }
    });
    // Dropping the HTTP body cancels prefill/decode and unblocks the sender.
    let stream = ReceiverStream::new(receiver).map(move |event| {
        let _keep_guard_until_body_drops = &guard;
        Ok::<Event, Infallible>(event)
    });
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn log_completion(id: &str, c: &Completion) {
    tracing::info!(
        request_id = id,
        prompt_tokens = c.usage.prompt_tokens,
        completion_tokens = c.usage.completion_tokens,
        first_token_ms = c.timing.first_token_ms,
        total_ms = c.timing.total_ms,
        "request complete"
    );
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn request_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "chatcmpl-{nanos:x}-{:x}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
        }
    }
    fn invalid(error: impl std::fmt::Display) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            error.to_string(),
        )
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        tracing::error!(error = %error, "request failed");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            error.to_string(),
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": {"message": self.message, "type": self.kind, "param": null, "code": self.kind}}))).into_response()
    }
}
