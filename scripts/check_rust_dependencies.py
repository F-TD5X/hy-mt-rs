#!/usr/bin/env python3
"""Reject native inference, BLAS, and tokenizer features in the resolved Rust build."""
import json
import subprocess

metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--locked", "--format-version", "1"]))
packages = {p["id"]: p["name"] for p in metadata["packages"]}
blocked = {"onig", "onig_sys", "llama-cpp-sys-2", "llama_cpp_sys", "ggml-sys", "accelerate-src", "intel-mkl-src", "candle-kernels", "candle-metal-kernels", "cudarc", "blas-src", "openblas-src"}
for node in metadata["resolve"]["nodes"]:
    name = packages[node["id"]]
    assert name not in blocked, f"Unexpected native dependency: {name}"
    if name == "esaxx-rs":
        assert "cpp" not in node["features"], "esaxx C++ feature is enabled"
    if name == "tokenizers":
        assert not {"onig", "esaxx_fast"}.intersection(node["features"]), "native tokenizer feature is enabled"
    if name == "candle-core":
        assert not {"cuda", "metal", "mkl", "accelerate"}.intersection(node["features"]), "native tensor backend is enabled"
print("Dependency check passed: Rust CPU inference and tokenization features only.")
