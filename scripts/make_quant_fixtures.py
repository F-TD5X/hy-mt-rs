#!/usr/bin/env python3
"""Generate exact decoder fixtures with pinned C oracles; optionally patch a test-only STQ copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess


def header(path):
    with path.open("rb") as f:
        def num(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        def text():
            return f.read(num("Q")).decode()
        def value(kind):
            if kind == 8:
                return text()
            if kind == 9:
                element, count = num("I"), num("Q")
                for _ in range(count):
                    value(element)
                return None
            return num({0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[kind])
        assert f.read(4) == b"GGUF" and num("I") == 3
        nt, nm = num("Q"), num("Q")
        meta, positions = {}, {}
        for _ in range(nm):
            key, kind = text(), num("I")
            positions[key] = f.tell()
            meta[key] = value(kind)
        tensors = []
        for _ in range(nt):
            name = text()
            dims = [num("Q") for _ in range(num("I"))]
            type_position = f.tell()
            dtype, offset = num("I"), num("Q")
            tensors.append({"name":name, "dims":dims, "dtype":dtype, "offset":offset, "type_position":type_position})
        alignment = meta.get("general.alignment", 32)
        start = (f.tell() + alignment - 1) // alignment * alignment
        return meta, positions, tensors, start


def patch_stq(root):
    source = root / "models/Hy-MT2-1.8B-1.25Bit.gguf"
    target = root / ".cache/stq-oracle.gguf"
    digest = hashlib.sha256()
    with source.open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            digest.update(block)
    assert digest.hexdigest() == "cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93"
    meta, positions, tensors, _ = header(source)
    assert meta["general.file_type"] == 41
    legacy = [t for t in tensors if t["dtype"] == 42]
    assert len(legacy) == 224
    shutil.copyfile(source, target)
    with target.open("r+b") as f:
        f.seek(positions["general.file_type"])
        f.write(struct.pack("<I", 42))
        for t in legacy:
            f.seek(t["type_position"])
            f.write(struct.pack("<I", 43))
    print(f"Patched a separate oracle copy: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-stq", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.patch_stq:
        patch_stq(root)
        return
    configs = {"Q8_0":(8,32,34,"standard"), "Q4_K":(12,256,144,"standard"), "Q6_K":(14,256,210,"standard"), "Q2_0C":(40,512,130,"q2c"), "STQ1_0":(43,256,42,"stq")}
    cases = []
    work = root / ".cache/quant-fixtures"
    work.mkdir(parents=True, exist_ok=True)
    for name, (type_id, count, size, oracle) in configs.items():
        encoded = bytearray()
        for block in range(3):
            b = bytearray((i * 37 + block * 79 + 11) % 256 for i in range(size))
            scale = [0.03125, -1.5, 0.0][block]
            offset = {"Q6_K":208, "STQ1_0":40}.get(name, 0)
            b[offset:offset+2] = struct.pack("<e", scale)
            if name == "Q4_K":
                b[2:4] = struct.pack("<e", 0.0625)
            encoded.extend(b)
        input_path, output_path = work / f"{name}.bin", work / f"{name}.f32"
        input_path.write_bytes(encoded)
        subprocess.run([str(root / f".cache/oracles/{oracle}/build/hy-reference"), "--decode", str(type_id), str(input_path), str(output_path)], check=True)
        expected = list(struct.unpack(f"<{count * 3}I", output_path.read_bytes()))
        cases.append({"dtype":name, "encoded_hex":encoded.hex(), "expected_f32_bits":expected, "oracle":oracle})
    (root / "tests/fixtures/quant.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("Wrote tests/fixtures/quant.json")


if __name__ == "__main__":
    main()
