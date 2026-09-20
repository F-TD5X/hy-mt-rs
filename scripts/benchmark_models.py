#!/usr/bin/env python3
"""Record sequential 1.8B/7B CPU benchmarks. 30B is excluded on this machine."""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=["stq", "q2c", "7b"], default=["stq", "q2c", "7b"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "tests/fixtures/models.json").read_text())
    hardware = {"os": platform.system(), "architecture": platform.machine()}
    if platform.system() == "Darwin":
        hardware["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        hardware["memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "hardware": hardware, "runs": []}
    for name in args.models:
        for concurrency in [1, 2]:
            command = [str(root / "target/release/hy-mt-rs"), "bench", "--model", str(root / "models" / manifest[name]["file"]), "--threads", str(args.threads), "--concurrency", str(concurrency), "--max-tokens", "16", "--ctx-size", "512"]
            result = json.loads(subprocess.check_output(command, text=True))
            report["runs"].append({"model": name, "model_sha256": manifest[name]["sha256"], **result})
            print(f"Finished {name}, concurrency {concurrency}", flush=True)
    (root / "docs/benchmarks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Wrote docs/benchmarks.json", flush=True)


if __name__ == "__main__":
    main()
