# Hy-MT2 implementation reference

Checked 2026-09-20 against official model files and source. Tensor counts, types, dimensions, and metadata below came from HTTP range reads of the actual GGUF headers. File hashes came from Hugging Face LFS metadata; this research did not download and hash each full file or run inference.

Subsequent implementation tests are recorded in [validation.md](validation.md).
The executed standard-format oracle uses commit `b23efaa2ef147f547ee75cbf0c621d61904de80e`;
the source-inspection links below also reference `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

## Pinned inputs

All model repositories are under `tencent/` on Hugging Face. Use the revision in the link rather than `main` when making test fixtures.

| Repository | Revision | Files |
| --- | --- | --- |
| [Hy-MT2-1.8B-1.25Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF/tree/9df5c824a00a744fb0512a29c640466f4d97dfb0) | `9df5c824a00a744fb0512a29c640466f4d97dfb0` | 1.25Bit |
| [Hy-MT2-1.8B-2bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-2bit-GGUF/tree/b630487d19ab7f336664a15b07c638d0d1071471) | `b630487d19ab7f336664a15b07c638d0d1071471` | 2Bit |
| [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/tree/a0c709d9fac510f2c807aa3af52872340dc37a4a) | `a0c709d9fac510f2c807aa3af52872340dc37a4a` | Q4_K_M, Q6_K, Q8_0 |
| [Hy-MT2-7B-GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF/tree/ab8472660ac61fac25f1af43fac2599d52a8a775) | `ab8472660ac61fac25f1af43fac2599d52a8a775` | Q4_K_M, Q6_K, Q8_0 |
| [Hy-MT2-30B-A3B-GGUF](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF/tree/fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3) | `fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3` | Q4_K_M, Q8_0 |

The 30B GGUF release exists even though some older model cards omit it. No converter is required to serve the official series.

| Exact filename | Bytes | SHA256 |
| --- | ---: | --- |
| `Hy-MT2-1.8B-1.25Bit.gguf` | 461860800 | `cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93` |
| `Hy-MT2-1.8B-2Bit.gguf` | 600534880 | `dcc33bbae9b28d923c8c76a64f6157840841d26f8774f3dfd770d5fabeeb1cd7` |
| `Hy-MT2-1.8B-Q4_K_M.gguf` | 1133080448 | `dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699` |
| `Hy-MT2-1.8B-Q6_K.gguf` | 1474785120 | `d98fe604dec1f28f58f80d7d560f7177e584d3b8e5835862687660e5ff97cb40` |
| `Hy-MT2-1.8B-Q8_0.gguf` | 1908528192 | `5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4` |
| `Hy-MT2-7B-Q4_K_M.gguf` | 4624648896 | `9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b` |
| `HY-MT2-7B-Q6_K.gguf` | 6164482720 | `88ef0aba59952a4cfe4be36cb5baf797dbb370bc60e9dcbd7297036021e52831` |
| `HY-MT2-7B-Q8_0.gguf` | 7981928896 | `58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0` |
| `Hy-MT2-30B-A3B-Q4_K_M.gguf` | 18236702880 | `bb44b11bb0f7cd3d1321645b41e911cd3de2e473731227fc6bf37aa18b543f88` |
| `Hy-MT2-30B-A3B-Q8_0.gguf` | 31985729376 | `d4e74d3b9479db5a7e2e4728879e15554b385e9549b411cc8b0ad03b2970ce98` |

## Quantized tensor contract

Required storage types for these releases are F32, Q4_K, Q6_K, Q8_0, Q2_0C, and STQ. Marketing bit counts are not GGML type names. In particular, the official 2Bit file is **not TQ2_0**.

| Storage | Tensor type ID | Weights/block | Bytes/block | Field order |
| --- | ---: | ---: | ---: | --- |
| F32 | 0 | 1 | 4 | little-endian float |
| Q8_0 | 8 | 32 | 34 | FP16 scale, 32 signed bytes |
| Q4_K | 12 | 256 | 144 | FP16 `d`, FP16 `dmin`, 12 scale/min bytes, 128 quant bytes |
| Q6_K | 14 | 256 | 210 | 128 low-bit bytes, 64 high-bit bytes, 16 signed scales, FP16 `d` |
| Tencent Q2_0C | 40 | 512 | 130 | FP16 scale, 128 quant bytes |
| Tencent legacy STQ | 42 | 256 | 42 | 32 index bytes, 8 sign bytes, FP16 scale |

Q4_K and Q6_K use interleaved layouts, not consecutive nibbles. Port the indexed reference decoders directly and retain their operation order for fixtures. Q4_K computes a scaled nibble minus a scaled minimum; Q6_K combines four low/high streams, subtracts 32, and applies a signed subscale. [Pinned standard decoders](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-quants.c#L1529)

Q2_0C decodes each byte from low to high two-bit pairs, mapping codes `0,1,2,3` to `-3,-1,+1,+3`, then multiplies by the block scale. Its storage rate is 2.03125 bits/weight. [Layout](https://github.com/chaxu01/llama.cpp/blob/2af64dd00a6689a7bfaf69b4768a944d0ec6bade/ggml/src/ggml-common.h#L290), [decoder](https://github.com/chaxu01/llama.cpp/blob/2af64dd00a6689a7bfaf69b4768a944d0ec6bade/ggml/src/ggml-quants.c#L2618)

STQ has 64 groups per block. For group `g`, read nibble `(qs[g/2] >> (4*(g%2))) & 15`, sign bit `(sign[g/8] >> (g%8)) & 1`, and codebook entry `(sign_bit << 4) | nibble`. Each two-bit codebook lane gives `(code - 1) * scale`. Write lane `p` to `(g/16)*64 + (g%16) + p*16`. This stride-16 placement is required. STQ has one zero and three signed nonzero values in each group; storage is 1.3125 bits/weight. [Codebook and layout](https://github.com/sjl623/llama.cpp/blob/1e411d8f5a1e23525fa3265dfb4bd76265465397/ggml/src/ggml-common.h#L290), [stride-16 decoder](https://github.com/sjl623/llama.cpp/blob/1e411d8f5a1e23525fa3265dfb4bd76265465397/ggml/src/ggml-quants.c#L2548)

The published low-bit files predate current enum assignments. Q2_0C uses tensor ID 40 and `general.file_type=39`; STQ uses ID 42 and file type 41. Current standard GGML assigns those tensor IDs to other formats. Use a validated legacy profile; never globally reinterpret IDs 40/42. Check architecture, dimensions, file type, tensor byte extents, and the known profile. Reject an ambiguous file. The STQ file also records `general.finetune=2bit-stride16`.

Observed tensor counts:

- Each low-bit 1.8B file: 224 custom tensors, 129 F32 tensors, one Q6_K embedding tensor. The output shares this embedding tensor.
- Each inspected 1.8B/7B Q4_K_M file: 192 Q4_K, 33 Q6_K, 129 F32.
- 30B Q4_K_M: 407 Q4_K, 72 Q6_K, 287 F32; Q8_0: 479 Q8_0, 287 F32.

## Model graphs and RoPE

| Property | 1.8B | 7B | 30B-A3B |
| --- | ---: | ---: | ---: |
| GGUF architecture | `hunyuan-dense` | `hunyuan-dense` | `hy_v3` |
| Blocks | 32 | 32 | 48 |
| Hidden dimension | 2048 | 4096 | 2048 |
| Query/KV heads | 16/4 | 32/8 | 32/4 |
| Head dimension | 128 | 128 | 128 |
| Dense FFN dimension | 6144 | 14336 | 6912 in block 0 |
| Vocabulary | 120818 | 128167 | 120832 |
| Embedding/output weights | tied | tied | separate |

All three use RMS epsilon 1e-5, SiLU gated FFNs, context metadata 262144, and effective GGUF RoPE base 11158840. Read those values from GGUF. The dense HF config's base 10000 and dynamic scaling were already folded into the GGUF base; do not apply the scaling again. [Conversion source](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/conversion/hunyuan.py#L253)

Both architectures use **NeoX split-half RoPE**: for head dimension 128, rotate coordinates `j` and `j+64` at angle `position * base^(-2*j/128)`. They do not use adjacent-coordinate rotation. Their conversion classes do not permute Q/K rows; do not apply the Llama HF-to-GGUF row permutation to these matrices. [RoPE architecture selection](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-model.cpp#L3028), [conversion classes](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/conversion/hunyuan.py#L285)

For dense Hunyuan, project Q/K, apply RoPE, then apply learned per-head Q/K RMS normalization. For HYV3, project Q/K, apply that normalization, then RoPE. The norm weights each contain 128 values shared across heads. This order changes results because the learned scale is not constant. [Dense graph](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/hunyuan-vl.cpp#L98), [HYV3 graph](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/hy-v3.cpp#L155)

HYV3 expands a 2048-wide hidden vector to a 4096-wide query vector. Never infer head dimension as hidden size divided by query-head count. Block 0 is dense; blocks 1–47 each have 128 routed experts, select eight per token, and add one always-active shared expert. Expert FFN dimension is 768. Compute router logits in F32, apply sigmoid, and add `exp_probs_b` only for top-eight selection. Gather the unbiased sigmoid values, normalize their sum, multiply by 2.826, and use these weights to combine the selected FFN outputs. Add the shared expert without a router gate. [HF reference](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/hy_v3/modeling_hy_v3.py#L281)

Important HYV3 tensor names and dimensions, in GGUF order (fastest dimension first):

| Tensor suffix under `blk.N.` | Dimensions |
| --- | --- |
| `attn_q.weight` / `attn_output.weight` | `[2048,4096]` / `[4096,2048]` |
| `attn_k.weight`, `attn_v.weight` | `[2048,512]` |
| `ffn_gate_inp.weight` / `exp_probs_b` | `[2048,128]` / `[128]`, both F32 |
| `ffn_gate_exps.weight`, `ffn_up_exps.weight` | `[2048,768,128]` |
| `ffn_down_exps.weight` | `[768,2048,128]` |
| `ffn_gate_shexp.weight`, `ffn_up_shexp.weight` | `[2048,768]` |
| `ffn_down_shexp.weight` | `[768,2048]` |

Slice only the selected experts from mapped packed weights. The 30B files contain no MTP/NextN blocks. [Tensor loading contract](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/hy-v3.cpp#L28)

## Tokenization, prompts, and stopping

All inspected GGUFs contain vocabulary, token types, BPE merges, special IDs, and the full Jinja chat template. No external tokenizer file is needed.

- `hunyuan-dense` pretokenizer (1.8B and 30B): apply three **sequential** isolated regex splits: numeric runs of length 1–3, CJK/Hiragana/Katakana spans, then the remaining letter/punctuation/space regex. Follow with byte-level encoding, `add_prefix_space=false`, `use_regex=false`. Do not join the three regexes into one alternation. No Unicode normalization is applied.
- `hunyuan` pretokenizer (7B): use the Qwen2-style expression, including single-digit splitting. Do not use the 1.8B expression.
- Reconstruct the byte-BPE vocabulary from GGUF strings and preserve merge rank order. Register the special tokens with their existing IDs. The rendered template contains BOS already: encode it without adding another BOS or EOS.

Exact expressions: [pinned pretokenizer source](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-vocab.cpp#L319). The sequential structure is also explicit in the [official 1.8B tokenizer](https://huggingface.co/tencent/Hy-MT2-1.8B/blob/main/tokenizer.json) and [30B tokenizer](https://huggingface.co/tencent/Hy-MT2-30B-A3B/blob/main/tokenizer.json).

| Size | BOS | GGUF EOS | Additional declared stop |
| --- | ---: | ---: | --- |
| 1.8B | 120000 | 120020 | none |
| 7B | 127958 | 3 | EOT 127960 |
| 30B | 120000 | 120025 | none |

Stop on EOS and any declared EOT/EOM; do not emit those tokens as response text. Do not treat every control token as a stop. In 30B, EOS is `<eos:6124c78e>`; the HF config's `eod_token_id=120026` is not the released GGUF EOS and is not declared as an extra stop there. [Reference EOG handling](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-vocab.cpp#L2867)

Render the embedded template rather than a shared hard-coded prompt. The 7B template differs from 1.8B. The 30B template defaults to `reasoning_effort=no_think`, adds `<｜reasoning_mode｜>reasoning_effort:no_think` to the prefix, and ends a user-turn generation prompt with `<｜hy_Assistant｜><think></think>`. It also inserts empty think tags before past assistant content. The template distinguishes a final assistant message from an earlier assistant message. [1.8B template](https://huggingface.co/tencent/Hy-MT2-1.8B/blob/main/chat_template.jinja), [7B template](https://huggingface.co/tencent/Hy-MT2-7B/blob/main/chat_template.jinja), [30B template](https://huggingface.co/tencent/Hy-MT2-30B-A3B/blob/main/chat_template.jinja)

Dense model-card defaults are temperature 0.7, top-p 0.6, top-k 20, repetition penalty 1.05; 30B defaults are temperature 0.7, top-p 1, no top-k limit, repetition penalty 1. All recommend 4096 new tokens and no default system prompt. Dense GGUF metadata instead records top-p 0.8. The server should state which default source it follows. [Official guidance](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF#inference-and-deployment)

## Independent parity oracle

These are test-only C/C++ references; they need not become runtime or build dependencies of the Rust program. Source inspection establishes feasibility, but this document does not claim the reference builds or full-model comparisons have run.

| Input | Reference checkout | Input adaptation |
| --- | --- | --- |
| Standard dense / HYV3 | `ggml-org/llama.cpp` at `b23efaa2ef147f547ee75cbf0c621d61904de80e` | none |
| Q2_0C | `chaxu01/llama.cpp` at `2af64dd00a6689a7bfaf69b4768a944d0ec6bade` | none; ID 40 and file type 39 match |
| STQ | `sjl623/llama.cpp` at `1e411d8f5a1e23525fa3265dfb4bd76265465397` | in a separate copy, change all custom tensor-table IDs 42 to 43 and `general.file_type` 41 to 42 |

For the STQ oracle copy, change only those fixed-width header fields; retain every payload byte and tensor offset. Verify the source file hash first, retain it unchanged, record the copy's hash and patch description, and verify all 224 changed tensor entries. The older STQ PR decoder used consecutive groups and is unsuitable for the released stride-16 file. Both custom PRs were still unmerged when checked. [STQ PR](https://github.com/ggml-org/llama.cpp/pull/22836), [Q2_0C PR](https://github.com/ggml-org/llama.cpp/pull/19357), [STQ current enum](https://github.com/sjl623/llama.cpp/blob/1e411d8f5a1e23525fa3265dfb4bd76265465397/ggml/include/ggml.h#L429), [STQ file-type enum](https://github.com/sjl623/llama.cpp/blob/1e411d8f5a1e23525fa3265dfb4bd76265465397/include/llama.h#L157)

Use CPU-only reference builds: disable Metal, CUDA, BLAS, Accelerate, KleidiAI, and CPU repacking; run with no GPU layers and one thread. The custom formats have generic CPU dot kernels, so SME hardware is not required. Use `llama-completion` for a smoke check and a small external harness around `llama_tokenize`, `llama_decode`, and `llama_get_logits_ith` to capture token IDs and logits. [Q2_0C generic dot](https://github.com/chaxu01/llama.cpp/blob/2af64dd00a6689a7bfaf69b4768a944d0ec6bade/ggml/src/ggml-cpu/quants.c#L571), [STQ generic dot](https://github.com/sjl623/llama.cpp/blob/1e411d8f5a1e23525fa3265dfb4bd76265465397/ggml/src/ggml-cpu/quants.c#L571), [reference API](https://github.com/sjl623/llama.cpp/blob/1e411d8f5a1e23525fa3265dfb4bd76265465397/include/llama.h#L1016)

Compare in this order:

1. Exact decoded values for synthetic blocks covering every low-bit code, sign, lane, block edge, and non-unit scale; compare real blocks too. Use the pinned C reference decoder, not a second Rust copy of the implementation.
2. Packed Rust matrix-vector output against reference-decoded F32 weights and an F32 dot product. Include quantized output embeddings and selected MoE experts.
3. Exact template text and prompt token IDs for multilingual input, numbers, punctuation, combining marks, whitespace, system messages, and multi-turn history.
4. Model logits for a fixed prompt, then cached incremental decoding against full-prefix decoding, followed by greedy token/output comparison.

The llama.cpp quantized CPU path rounds activations to Q8_K for custom-format dots. A Rust path that decodes weights and multiplies F32 activations will not have bit-identical logits. Use strict decoder tests and measured logit tolerances; report top-token disagreements instead of relaxing tolerance until they disappear. HYV3 router normalization also differs at tiny sums: Transformers uses `sum + 1e-20`, while llama.cpp clamps to `6.103515625e-5`. Keep the chosen numerical reference explicit. [Reference router normalization](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-graph.cpp)
