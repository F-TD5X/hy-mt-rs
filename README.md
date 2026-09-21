# hy-mt-rs

A Rust CPU inference library and Chat Completions server for Tencent Hy-MT2
GGUF models. It supports concurrent requests, streaming, and the official
1.8B 2Bit and 1.25Bit files.

Inference and tokenization run in Rust. The CPU path uses packed, memory-mapped
weights, bounded float scratch space, Candle CPU GEMM, and runtime-selected
NEON/AVX2 dot products with a scalar fallback. Single-token steps quantize the
activations to int8 and multiply them with ARMv8.2 integer dot products where
the CPU has them; batched prefill, AVX2, and scalar paths keep F32 activations.
The build disables native BLAS, GPU engines, Oniguruma, and the C++ tokenizer
feature.

## Build and run

Use stable Rust and either let the CLI download the default 2B-1.25Bit model
automatically into `data/`, or provide a local GGUF file:

```sh
cargo build --release --locked

# Start the server (downloads the 2B-1.25Bit model into data/ on first run):
./target/release/hy-mt-rs serve

# Or specify an existing model file:
./target/release/hy-mt-rs serve \
  --model models/Hy-MT2-1.8B-1.25Bit.gguf \
  --model-id hy-mt2 \
  --listen 127.0.0.1:8080
```

You can also explicitly pre-download the 2B-1.25Bit model:

```sh
./target/release/hy-mt-rs download
```

Download a GGUF from the official repositories below, or use the optional
development helper for a checksum-pinned download:

```sh
python3 scripts/fetch_models.py stq
```

The runtime reads the tokenizer and chat template from the GGUF. No tokenizer
sidecar or model conversion is needed. Keep the file unchanged while the
process uses it. Restart the process to select a different model.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hy-mt2",
    "messages": [{"role": "user", "content": "Translate into Chinese: Hello, world."}],
    "max_tokens": 128,
    "temperature": 0,
    "stream": true
  }'
```

The default bind address is local. The server has no built-in authentication.

## Supported models and formats

| Model | Official GGUF exports | Verification |
| --- | --- | --- |
| [1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | Q4_K_M, Q6_K, Q8_0 | Standard block decoders tested |
| [1.8B 2Bit](https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF) | Q2_0C with Q6_K embeddings | Full-model reference test passed |
| [1.8B 1.25Bit](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) | Legacy STQ1_0 with Q6_K embeddings | Full-model reference test passed |
| [7B](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) | Q4_K_M, Q6_K, Q8_0 | Q4_K_M full-model reference test passed |
| [30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF) | Q4_K_M, Q8_0 | Graph and template tested with small fixtures; full-model execution deferred |

The reader supports single-file GGUF v3 and tensor types F32, F16, Q4_K,
Q6_K, Q8_0, Q2_0C, and the released Tencent STQ layout. Split GGUFs and other
quantization types return an explicit error.

The 2Bit export uses **Q2_0C**, not TQ2_0. It stores 512 weights in 130 bytes.
The 1.25Bit export stores 256 sparse ternary weights in 42 bytes and uses a
stride-16 layout. Both keep some tensors in other formats.

Tencent's released low-bit files use numeric type IDs that conflict with
newer GGUF formats. `--gguf-profile auto` recognizes the pinned official
files by SHA-256. Renaming a file does not affect detection. For a known
compatible Tencent export, select the profile explicitly:

```sh
./target/release/hy-mt-rs inspect --model model.gguf --gguf-profile tencent-q2-0c
./target/release/hy-mt-rs inspect --model model.gguf --gguf-profile tencent-stq1
```

Explicit profiles still validate architecture, file type, block dimensions,
and tensor ranges. Unknown or ambiguous layouts fail before inference.
See [format and source references](docs/reference.md) for exact revisions.

## API

- `POST /v1/chat/completions`: text completions, with optional SSE streaming.
- `GET /v1/models`: the one model loaded by this process.
- `GET /healthz`: available after model loading succeeds.

Supported request fields:

| Field | Behavior |
| --- | --- |
| `model` | Must match `--model-id` |
| `messages` | Nonempty array; roles `system`, `user`, `assistant`; system message only at the start |
| `content` | A string or an array of `{ "type": "text", "text": "..." }` parts |
| `stream` | Boolean, default `false` |
| `stream_options.include_usage` | Send a final usage chunk when streaming |
| `max_tokens`, `max_completion_tokens` | Supply at most one; default is min(4096, remaining context) |
| `temperature`, `top_p`, `seed` | Sampling controls; temperature 0 selects greedy decoding |
| `stop` | A nonempty string or up to four nonempty strings |
| `n` | Only 1 is supported |
| `top_k`, `repetition_penalty` | Extensions; top-k -1 or 0 disables the filter |

The API rejects unsupported fields, tool calls, images, and structured-output
requests. It returns JSON error objects. Request bodies are limited to 1 MiB.
Validation errors return 400, an unknown model returns 404, and a full queue
returns 429. Oversized prompts are rejected without truncation.

Streaming sends an assistant-role chunk, text deltas, a finish-reason chunk,
optional usage, then `[DONE]`. UTF-8 characters and stop strings can cross
token boundaries. Streamed text matches the non-streaming result for the same
sampling state. Usage includes prompt/template tokens and sampled end tokens.

The server follows Tencent's model-card sampling defaults:

| Model | Temperature | Top-p | Top-k | Repetition penalty |
| --- | ---: | ---: | ---: | ---: |
| Dense 1.8B / 7B | 0.7 | 0.6 | 20 | 1.05 |
| 30B-A3B | 0.7 | 1.0 | disabled | 1.0 |

There is no added system prompt. The 30B template retains its `no_think`
default. The server preserves each model's own EOS/EOT rules.

## Concurrency and memory

Each process shares one immutable model across requests. Each request owns
its KV cache, random generator, and stop state. CPU work uses one bounded
thread pool. A slow streaming client does not occupy a CPU worker.

```sh
./target/release/hy-mt-rs serve \
  --model models/Hy-MT2-1.8B-1.25Bit.gguf \
  --threads 8 \
  --max-concurrent-requests 2 \
  --queue-capacity 8 \
  --ctx-size 8192
```

Defaults are two active requests, eight queued requests, an 8,192-token
context, and one CPU worker per performance core. On macOS the thread count is
read from `hw.perflevel0` rather than every logical CPU, because the pool runs
at user-interactive QoS and the scheduler places those threads on performance
cores only. KV caches use F32 and grow in
256-token chunks. At an 8,192-token context, the maximum KV cache per active
request is 1 GiB for 1.8B, 2 GiB for 7B, and 1.5 GiB for 30B. Packed weights,
tokenizer state, and working memory are additional. Startup logs show the
configured total KV bound.

Client disconnects cancel work. Ctrl-C cancels active generation and shuts
down the HTTP server. GPU support, model switching, and continuous batching
are outside this release.

## Inspect and benchmark

```sh
./target/release/hy-mt-rs inspect --model models/Hy-MT2-1.8B-1.25Bit.gguf --tensors
./target/release/hy-mt-rs bench --model models/Hy-MT2-1.8B-1.25Bit.gguf --threads 8
./target/release/hy-mt-rs bench --model models/Hy-MT2-1.8B-1.25Bit.gguf --threads 8 --concurrency 2
```

Benchmarks emit JSON with load time, prefill rate, first-token latency, decode
rate, process peak RSS, output text, and token IDs. Use `--prompt` or
`--prompt-file` to select input; `--raw-prompt` skips chat formatting.
Benchmark sampling defaults to temperature 0 and seed 42.

## Tests

```sh
cargo fmt --check
python3 scripts/check_rust_dependencies.py
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

Normal tests require no model downloads. They cover independent decoder
fixtures, all three official chat templates, small dense/MoE graphs, KV-cache
equivalence, HTTP/SSE output, concurrent requests, overload, and cancellation.
CI runs natively on Linux x86-64 and macOS ARM64 without `target-cpu=native`.

Full-model tests are opt-in and use about 5.3 GiB of downloaded files:

```sh
python3 scripts/fetch_models.py stq q2c 7b
cargo test --release --test models -- --ignored --test-threads=1 --nocapture
```

Set `HY_MT_MODELS` to use another model directory. The tests compare exact
template text, prompt IDs, greedy token IDs, and next-token logits with pinned
reference engines. Rust keeps F32 activations for prefill and for its portable
kernels, and quantizes them to int8 for the single-token GEMV path, while the
quantized reference rounds activations to Q8, so logits use a documented
numerical tolerance.

30B inference and benchmarks are deferred on this 16 GiB development machine.
Its MoE graph and template have small-fixture coverage. See
[validation details](docs/validation.md) for what was actually run.

To regenerate the independent fixtures, see [the test-reference guide](tests/reference/README.md).
Those optional C++/Python tools are separate from the Rust build and runtime.

## Library

`Model::open` loads shared weights and the embedded tokenizer. Use
`model.tokenizer.encode_chat`, then `generation::Generator::with_pool` with a
shared Rayon pool. Each call to `generate` creates a separate session. For
direct token-by-token use, create a `Session` with `Model::new_session` and
call `Model::forward` with at most 256 tokens per call.

The Candle dependency is pinned to a revision that uses Rust `fancy-regex`;
the published 0.11 package enables Oniguruma. Commit `Cargo.lock` with this
source pin. `scripts/check_rust_dependencies.py` checks the selected features.

## License

Project code is MIT licensed. See [third-party notices](THIRD_PARTY_NOTICES.md)
for decoder sources and Tencent's Apache-2.0 chat-template fixtures.
