mod common;

use anyhow::Result;
use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use hy_mt_rs::server::{self, ServerConfig};
use serde_json::{Value, json};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

fn app(concurrency: usize, queue: usize) -> Result<(tempfile::NamedTempFile, Router)> {
    app_with_threads(concurrency, queue, 2, CancellationToken::new())
}

fn app_with_threads(
    concurrency: usize,
    queue: usize,
    threads: usize,
    shutdown: CancellationToken,
) -> Result<(tempfile::NamedTempFile, Router)> {
    let (file, model) = common::model(false, false)?;
    let router = server::router_with_shutdown(
        model,
        ServerConfig {
            model_id: "test".into(),
            context_size: 2048,
            max_concurrent_requests: concurrency,
            queue_capacity: queue,
            threads,
        },
        shutdown,
    )?;
    Ok((file, router))
}

#[tokio::test]
async fn a_slow_stream_does_not_block_the_only_cpu_worker() -> Result<()> {
    let (_file, app) = app_with_threads(2, 0, 1, CancellationToken::new())?;
    let mut body = base();
    body["stream"] = json!(true);
    body["max_tokens"] = json!(1024);
    let slow = app.clone().oneshot(request(body)).await?;
    tokio::time::sleep(Duration::from_millis(30)).await;
    let completion =
        tokio::time::timeout(Duration::from_secs(5), app.oneshot(request(base()))).await??;
    assert_eq!(completion.status(), StatusCode::OK);
    assert_eq!(
        json_body(completion).await?["choices"][0]["message"]["content"],
        "aaaa"
    );
    drop(slow);
    Ok(())
}

#[tokio::test]
async fn shutdown_cancels_a_stream_with_a_full_output_queue() -> Result<()> {
    let shutdown = CancellationToken::new();
    let (_file, app) = app_with_threads(1, 0, 1, shutdown.clone())?;
    let mut body = base();
    body["stream"] = json!(true);
    body["max_tokens"] = json!(1024);
    let response = app.oneshot(request(body)).await?;
    tokio::time::sleep(Duration::from_millis(30)).await;
    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), response.into_body().collect()).await??;
    Ok(())
}

fn request(body: Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn base() -> Value {
    json!({"model":"test", "messages":[{"role":"user","content":"Hi"}], "temperature":0, "max_tokens":4})
}

async fn json_body(response: axum::response::Response) -> Result<Value> {
    Ok(serde_json::from_slice(
        &response.into_body().collect().await?.to_bytes(),
    )?)
}

#[tokio::test]
async fn completions_and_sse_have_equal_text_and_usage() -> Result<()> {
    let (_file, app) = app(2, 1)?;
    let response = app.clone().oneshot(request(base())).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let full = json_body(response).await?;
    assert_eq!(full["choices"][0]["message"]["content"], "aaaa");
    assert_eq!(full["choices"][0]["finish_reason"], "length");
    let mut body = base();
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage":true});
    let response = app.oneshot(request(body)).await?;
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let bytes = response.into_body().collect().await?.to_bytes();
    let sse = std::str::from_utf8(&bytes)?;
    let mut text = String::new();
    let mut usage = Value::Null;
    let mut done = false;
    for line in sse.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done = true;
            continue;
        }
        let event: Value = serde_json::from_str(line)?;
        if let Some(delta) = event["choices"][0]["delta"]["content"].as_str() {
            text.push_str(delta);
        }
        if !event["usage"].is_null() {
            usage = event["usage"].clone();
        }
        assert_eq!(event["object"], "chat.completion.chunk");
    }
    assert!(done);
    assert_eq!(text, "aaaa");
    assert_eq!(usage, full["usage"]);
    Ok(())
}

#[tokio::test]
async fn invalid_requests_use_json_errors() -> Result<()> {
    let (_file, app) = app(1, 0)?;
    for (field, value, status) in [
        ("model", json!("missing"), StatusCode::NOT_FOUND),
        ("temperature", json!(-1), StatusCode::BAD_REQUEST),
        ("max_tokens", json!(10000), StatusCode::BAD_REQUEST),
        ("max_tokens", json!(0), StatusCode::BAD_REQUEST),
        ("tools", json!([]), StatusCode::BAD_REQUEST),
        ("n", json!(2), StatusCode::BAD_REQUEST),
        ("stop", json!(""), StatusCode::BAD_REQUEST),
    ] {
        let mut body = base();
        body[field] = value;
        let response = app.clone().oneshot(request(body)).await?;
        assert_eq!(response.status(), status, "field {field}");
        assert!(json_body(response).await?["error"]["message"].is_string());
    }
    Ok(())
}

#[tokio::test]
async fn concurrent_streams_hold_slots_and_disconnect_releases_them() -> Result<()> {
    let (_file, app) = app(2, 0)?;
    let mut body = base();
    body["stream"] = json!(true);
    body["max_tokens"] = json!(1024);
    let first = app.clone().oneshot(request(body.clone())).await?;
    let second = app.clone().oneshot(request(body)).await?;
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::OK);
    let rejected = app.clone().oneshot(request(base())).await?;
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    drop(first);
    drop(second);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let response = app.clone().oneshot(request(base())).await?;
            if response.status() == StatusCode::OK {
                assert_eq!(
                    json_body(response).await?["choices"][0]["message"]["content"],
                    "aaaa"
                );
                return Ok::<_, anyhow::Error>(());
            }
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await??;
    Ok(())
}

#[tokio::test]
async fn models_health_and_text_parts_work() -> Result<()> {
    let (_file, app) = app(1, 1)?;
    let response = app
        .clone()
        .oneshot(Request::get("/v1/models").body(Body::empty())?)
        .await?;
    assert_eq!(json_body(response).await?["data"][0]["id"], "test");
    let response = app
        .clone()
        .oneshot(Request::get("/healthz").body(Body::empty())?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = base();
    body["messages"][0]["content"] =
        json!([{"type":"text","text":"He"},{"type":"text","text":"llo"}]);
    assert_eq!(app.oneshot(request(body)).await?.status(), StatusCode::OK);
    Ok(())
}
