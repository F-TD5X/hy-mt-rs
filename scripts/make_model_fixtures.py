#!/usr/bin/env python3
"""Capture pinned reference logits and greedy tokens. Deliberately excludes 30B."""
import json
from pathlib import Path
import shutil
import subprocess
from build_oracle import PINS
from make_quant_fixtures import patch_stq


def main():
    root = Path(__file__).resolve().parent.parent
    models = json.loads((root / "tests/fixtures/models.json").read_text())
    tokenizers = json.loads((root / "tests/fixtures/tokenizers.json").read_text())
    patch_stq(root)
    records = {}
    destination = root / "tests/fixtures/logits"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["stq", "q2c", "7b"]:
        oracle = name if name != "7b" else "standard"
        tokenizer = "7b" if name == "7b" else "1.8b"
        model = root / "models" / models[name]["file"] if name != "stq" else root / ".cache/stq-oracle.gguf"
        prefix = root / ".cache" / f"{name}-ref"
        prompt = root / ".cache" / f"{tokenizer}.prompt.txt"
        prompt.write_text(tokenizers[tokenizer]["cases"][0]["rendered"])
        with (root / ".cache" / f"{name}-ref.log").open("w") as log:
            subprocess.run([str(root / f".cache/oracles/{oracle}/build/hy-reference"), str(model), str(prompt), str(prefix), "16", "4"], stdout=log, stderr=log, check=True)
        shutil.copyfile(str(prefix) + ".logits.f32", destination / f"{name}.f32")
        records[name] = {
            "oracle_repo": PINS[oracle][0], "oracle_revision": PINS[oracle][1],
            "model_sha256": models[name]["sha256"], "tokenizer": tokenizer,
            "prompt_tokens": json.loads(Path(str(prefix) + ".prompt.json").read_text()),
            "generated_tokens": json.loads(Path(str(prefix) + ".generated.json").read_text()),
            "text": Path(str(prefix) + ".text").read_text(),
            "logits_file": f"logits/{name}.f32",
            "relative_l2_limit": 0.03,
        }
        print(f"Captured {name}", flush=True)
    (root / "tests/fixtures/model_reference.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
