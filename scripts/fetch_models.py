#!/usr/bin/env python3
"""Fetch checksum-pinned integration-test models. Not used by the Rust server."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(model, directory):
    path = directory / model["file"]
    if path.exists():
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while block := f.read(4 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != model["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}; remove it to retry")
        print(f"Verified {path}", flush=True)
        return
    url = f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{model["file"]}?download=true'
    partial = path.with_suffix(".gguf.part")
    print(f"Downloading {model['file']}", flush=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as target:
        while block := source.read(4 * 1024 * 1024):
            target.write(block)
            digest.update(block)
    if digest.hexdigest() != model["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {partial}")
    partial.replace(path)
    print(f"Verified {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=["stq", "q2c", "7b", "30b"])
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).parent.parent / "tests/fixtures/models.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda name: fetch(manifest[name], args.directory), dict.fromkeys(args.models)))


if __name__ == "__main__":
    main()
