# Validation

Checked on 2026-09-20 with Rust 1.98.1 on an Apple M2 Pro, 16 GiB RAM,
macOS ARM64. Model downloads were verified against the SHA-256 values in
`tests/fixtures/models.json`.

## Checks run

| Check | Result |
| --- | --- |
| `cargo fmt --check` | Passed |
| `cargo clippy --locked --all-targets -- -D warnings` | Passed |
| `cargo test --locked` | 27 passed; 3 full-model tests ignored by default |
| `cargo test --release --test models -- --ignored --test-threads=1 --nocapture` | All 3 full-model tests passed |
| `cargo build --locked --release` | Passed |
| `cargo check --target x86_64-unknown-linux-gnu --all-targets` | Passed |
| `python3 scripts/check_rust_dependencies.py` | Passed |
| Real HTTP health/model endpoints and simultaneous JSON/SSE completions | Passed |

Native Linux execution is configured in CI and was not run on this Mac.
The ARM64 release binary's dynamic library list contains only macOS
`libSystem` and `libiconv`; it does not link an inference engine or BLAS.

Normal tests cover exact decoded F32 bit patterns from independent C
decoders, scalar/SIMD agreement, all three official chat templates, dense
and MoE cache equivalence, context bounds, request-local state, HTTP/SSE
output and usage, queue saturation, slow clients, and shutdown cancellation.
The slow-client case also passes with only one CPU worker.
The real HTTP test used the 1.25Bit model, two active requests, and four CPU
threads. Both responses returned `你好，世界。` with 14 prompt tokens and 5
completion tokens. The test server was stopped after the check.

## Real models

Each tested model produced the exact reference prompt token IDs and the
same greedy token sequence for the fixed translation prompt. The output
was `你好，世界。`, ending on the model's own EOS/EOT token.

| Model | Relative L2 logit error | First-token top-10 overlap | Greedy output |
| --- | ---: | ---: | --- |
| 1.8B 1.25Bit | 0.006536 | 10/10 | Exact match |
| 1.8B 2Bit | 0.017630 | 10/10 | Exact match |
| 7B Q4_K_M | 0.022069 | 10/10 | Exact match |

The independent CPU references quantize activations to Q8; Rust uses F32
activations. Full-model tests therefore require relative L2 error below
0.03, the same highest logit, and the same greedy output IDs. Decoder tests
require exact F32 bit patterns and do not use this tolerance. Fixture source
revisions are stored in `tests/fixtures/model_reference.json`.

No 30B model inference or benchmark was run. Per the requested memory limit,
its validation is limited to source/header inspection, small MoE graph tests,
and the real chat-template fixtures. Its unfinished download was removed.

## Sample CPU measurements

These are single short translation runs with 8 CPU threads, a 512-token
context limit, and greedy sampling. They ran on a shared desktop. Repeat
with the target workload to assess performance. Rates below are per request;
RSS is the maximum for the whole process.

| Model | Concurrent requests | Mean first token (ms) | Mean decode tokens/s | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1.8B 1.25Bit | 1 | 297.5 | 5.26 | 554.3 |
| 1.8B 1.25Bit | 2 | 703.2 | 2.53 | 589.5 |
| 1.8B 2Bit | 1 | 416.5 | 2.75 | 682.3 |
| 1.8B 2Bit | 2 | 808.3 | 2.88 | 720.7 |
| 7B Q4_K_M | 1 | 8139.0 | 0.14 | 2759.4 |
| 7B Q4_K_M | 2 | 9680.8 | 0.10 | 2892.0 |

The 7B CPU path is slow on this machine. These measurements do not establish
a throughput target. Weights remain packed, and concurrent requests share
the model mapping. The smaller extra RSS at concurrency two comes mainly
from the second request's cache and working memory.

[Raw measurements](benchmarks.json) include load time, complete timings,
token IDs, output text, hardware, and model hashes. Reproduce them with
`python3 scripts/benchmark_models.py --threads 8`. This script excludes 30B.
