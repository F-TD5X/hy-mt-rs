# Independent reference fixtures

These tools are used only to regenerate test fixtures. The Rust application
does not build, link, or start them.

```sh
python3 -m venv .cache/reference-env
.cache/reference-env/bin/python -m pip install cmake ninja tokenizers==0.22.2 jinja2==3.1.6
python3 scripts/build_oracle.py stq
python3 scripts/build_oracle.py q2c
python3 scripts/build_oracle.py standard
.cache/reference-env/bin/python scripts/make_tokenizer_fixtures.py
python3 scripts/make_quant_fixtures.py
python3 scripts/make_model_fixtures.py
```

Download the `stq`, `q2c`, and `7b` models first with `scripts/fetch_models.py`.
`build_oracle.py` pins every source revision and disables GPU, BLAS,
Accelerate, OpenMP, and CPU repacking. Builds and logs remain under `.cache`.
The model fixture script deliberately excludes 30B.

The STQ reference uses updated enum IDs. `make_model_fixtures.py` verifies the
original file's SHA-256 and creates a separate copy: it changes all 224
STQ tensor IDs from 42 to 43, and the file-type metadata from 41 to 42.
Weight payloads remain unchanged. The Rust application reads the original
published file directly.

Fixtures:

- `quant.json`: three synthetic blocks of each supported quantized type,
  including negative/zero scales. Expected F32 bit patterns come from the C
  decoders. Rust must match them exactly.
- `tokenizers.json`: original templates plus expected text and IDs from
  Python Jinja2 and the official tokenizer files. Source revisions are stored
  in the fixture. It includes Chinese, Japanese, Arabic digits, combining
  marks, whitespace, added tokens, system messages, and conversation history.
- `model_reference.json` and `logits/*.f32`: a fixed translation prompt,
  next-token logits, and greedy outputs from the pinned C++ engines. The
  fixture records source revisions and the model hash.

The model oracle uses F32 KV caches and no repetition penalty. Its quantized
matrix products round activations to Q8; Rust instead multiplies F32
activations against decoded weight blocks. The model tests allow relative
L2 logit error below 0.03 and require the same top token and complete greedy
token sequence. The block decoder test has no numerical tolerance.

To inspect logits from Rust:

```sh
cargo run --release --example reference_dump -- \
  models/Hy-MT2-1.8B-1.25Bit.gguf .cache/1.8b.prompt.txt .cache/stq-rust
```

The `.f32` files use little-endian floats. The `.prompt.json` files contain
the exact token IDs used for prefill.
