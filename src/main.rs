use std::{collections::BTreeMap, net::SocketAddr, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use hy_mt_rs::{
    Gguf, Model, Profile,
    generation::{Generator, Options, Sampling},
    quant,
    server::{self, ServerConfig},
    tokenizer::{ChatMessage, Role},
};
use rayon::ThreadPoolBuilder;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    version,
    about = "Pure Rust CPU server for Hy-MT2 GGUF translation models"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve one model through the text Chat Completions API.
    Serve(Serve),
    /// Validate GGUF layout and print model metadata.
    Inspect(Inspect),
    /// Measure local inference, without HTTP.
    Bench(Bench),
}

#[derive(Args)]
struct ModelArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    gguf_profile: Profile,
}

#[derive(Args)]
struct Serve {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, default_value = "hy-mt2")]
    model_id: String,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    #[arg(long, default_value_t = 8192)]
    ctx_size: usize,
    #[arg(long, default_value_t = 2)]
    max_concurrent_requests: usize,
    #[arg(long, default_value_t = 8)]
    queue_capacity: usize,
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
}

#[derive(Args)]
struct Inspect {
    #[command(flatten)]
    model: ModelArgs,
    /// Include each tensor's dimensions and byte range.
    #[arg(long)]
    tensors: bool,
}

#[derive(Args)]
struct Bench {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(
        long,
        default_value = "Translate the following text into Chinese:\nHello, world."
    )]
    prompt: String,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Treat the prompt as already formatted; do not apply the chat template.
    #[arg(long)]
    raw_prompt: bool,
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    #[arg(long, default_value_t = 8192)]
    ctx_size: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    match cli.command {
        Command::Inspect(args) => inspect(args),
        Command::Bench(args) => bench(args),
        Command::Serve(args) => serve(args),
    }
}

fn inspect(args: Inspect) -> Result<()> {
    let g = Gguf::open(&args.model.model, args.model.gguf_profile)?;
    let mut counts = BTreeMap::new();
    for t in g.tensors.values() {
        *counts.entry(format!("{:?}", t.dtype)).or_insert(0usize) += 1;
    }
    let metadata: BTreeMap<_, _> = g
        .metadata
        .iter()
        .filter(|(k, v)| !v.is_array() && k.as_str() != "tokenizer.chat_template")
        .collect();
    let mut summary = json!({"profile": g.profile, "file_bytes": g.byte_len(), "tensor_count": g.tensors.len(), "tensor_types": counts, "metadata": metadata});
    if args.tensors {
        summary["tensors"] = json!(g.tensors.values().collect::<Vec<_>>());
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn serve(args: Serve) -> Result<()> {
    let started = Instant::now();
    let model = Arc::new(Model::open(&args.model.model, args.model.gguf_profile)?);
    let config = ServerConfig {
        model_id: args.model_id,
        context_size: args.ctx_size,
        max_concurrent_requests: args.max_concurrent_requests,
        queue_capacity: args.queue_capacity,
        threads: args.threads,
    };
    let shutdown = CancellationToken::new();
    let router = server::router_with_shutdown(model.clone(), config, shutdown.clone())?;
    let kv_bytes = model
        .config
        .kv_bytes_per_token()
        .checked_mul(args.ctx_size)
        .and_then(|n| n.checked_mul(args.max_concurrent_requests))
        .context("KV estimate overflow")?;
    tracing::info!(architecture = ?model.config.architecture, profile = ?model.profile, weight_file_bytes = model.file_bytes, max_kv_bytes = kv_bytes, context = args.ctx_size, concurrency = args.max_concurrent_requests, threads = args.threads, kernel = quant::kernel_name(), load_ms = started.elapsed().as_millis(), "model ready");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let listener = tokio::net::TcpListener::bind(args.listen).await?;
            tracing::info!(address = %listener.local_addr()?, "listening");
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = tokio::signal::ctrl_c().await;
                    shutdown.cancel();
                })
                .await?;
            Ok(())
        })
}

fn bench(args: Bench) -> Result<()> {
    ensure!(
        args.concurrency > 0 && args.threads > 0,
        "concurrency and threads must be positive"
    );
    let started = Instant::now();
    let model = Model::open(&args.model.model, args.model.gguf_profile)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let text = if let Some(path) = args.prompt_file {
        std::fs::read_to_string(path)?
    } else {
        args.prompt
    };
    let prompt = if args.raw_prompt {
        model.tokenizer.encode(&text)?
    } else {
        model.tokenizer.encode_chat(&[ChatMessage {
            role: Role::User,
            content: text,
        }])?
    };
    let mut sampling = Sampling::for_model(model.config.architecture);
    sampling.temperature = args.temperature;
    sampling.seed = Some(args.seed);
    let options = Options {
        max_tokens: args.max_tokens,
        sampling,
        stop: vec![],
    };
    let pool = ThreadPoolBuilder::new().num_threads(args.threads).build()?;
    let wall_start = Instant::now();
    let runs = std::thread::scope(|scope| -> Result<Vec<_>> {
        let handles: Vec<_> = (0..args.concurrency)
            .map(|_| {
                scope.spawn(|| {
                    Generator::with_pool(&model, &pool).generate(
                        &prompt,
                        args.ctx_size,
                        &options,
                        &CancellationToken::new(),
                        |_| Ok(()),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("benchmark thread panicked"))?
            })
            .collect()
    })?;
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.;
    let reports: Vec<_> = runs.iter().map(|c| json!({
        "completion": c,
        "prefill_tokens_per_second": c.usage.prompt_tokens as f64 * 1000. / c.timing.prefill_ms.max(0.001),
        "decode_tokens_per_second": c.usage.completion_tokens.saturating_sub(1) as f64 * 1000. / c.timing.decode_ms.max(0.001),
    })).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "config": model.config, "profile": model.profile, "load_ms": load_ms, "wall_ms": wall_ms,
            "threads": args.threads, "concurrency": args.concurrency, "kernel": quant::kernel_name(),
            "peak_rss_bytes": peak_rss_bytes(), "prompt_token_ids": prompt, "runs": reports,
        }))?
    );
    Ok(())
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes this struct on success; check its return value.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let value = u64::try_from(usage.ru_maxrss).ok()?;
    #[cfg(target_os = "macos")]
    return Some(value);
    #[cfg(not(target_os = "macos"))]
    Some(value * 1024)
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}
