# Third-party notices

The GGUF quantization layouts, reference-derived Q4_K/Q6_K decoding, and STQ
codebook in `src/quant.rs` follow ggml/llama.cpp and its Tencent quantization
branches. Source revisions and links are in [the reference notes](docs/reference.md).

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Chat-template text in `tests/fixtures/tokenizers.json` comes from Tencent's
Hy-MT2 repositories under Apache License 2.0. The fixture records the source
repository and revision. The accompanying license is preserved in
[`licenses/Hy-MT2-APACHE-2.0.txt`](licenses/Hy-MT2-APACHE-2.0.txt).

Rust dependencies retain their own licenses. Reference engines downloaded by
the optional developer scripts stay in `.cache` and are not linked into the
server. Model weights are not distributed in this repository.
