#!/usr/bin/env python3
"""Build independent tokenizer/template fixtures from pinned Hugging Face files."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request

import jinja2
import tokenizers

SOURCES = {
    "1.8b": ("Hy-MT2-1.8B", "9a341cd1b679d3efd23b46e847b01745a71ed792"),
    "7b": ("Hy-MT2-7B", "9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223"),
    "30b": ("Hy-MT2-30B-A3B", "d3ead4dba61c09aac60a261a96ad1df3e705febb"),
}

CASES = [
    [{"role": "user", "content": "Translate the following text into Chinese:\nHello, world."}],
    [{"role": "user", "content": "你好，日本語、ひらがな カタカナ. English isn't ALL CAPS! 1234567 ١٢٣ 😀 e\u0301\t\n  end\n\n"}],
    [{"role": "system", "content": "Translate into German."}, {"role": "user", "content": "Good morning."}, {"role": "assistant", "content": "Guten Morgen."}, {"role": "user", "content": "Good night."}],
    [{"role": "user", "content": "Keep these tags: <think></think> <answer>hi</answer>. /no_think"}],
]


def make(name, source, root):
    repo, revision = source
    cache = root / ".cache/tokenizers" / name
    cache.mkdir(parents=True, exist_ok=True)
    for file in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        path = cache / file
        if not path.exists():
            with urllib.request.urlopen(f"https://huggingface.co/tencent/{repo}/resolve/{revision}/{file}", timeout=120) as response:
                path.write_bytes(response.read())
    tok = tokenizers.Tokenizer.from_file(str(cache / "tokenizer.json"))
    config = json.loads((cache / "tokenizer_config.json").read_text())
    source_template = (cache / "chat_template.jinja").read_text()
    template = jinja2.Environment().from_string(source_template)
    cases = []
    for messages in CASES:
        rendered = template.render(messages=messages, add_generation_prompt=True, bos_token=config["bos_token"], eos_token=config["eos_token"], tools=[], reasoning_effort="no_think")
        cases.append({"messages": messages, "rendered": rendered, "tokens": tok.encode(rendered, add_special_tokens=False).ids})
    return name, {"repo": f"tencent/{repo}", "revision": revision, "template": source_template, "bos_token": config["bos_token"], "eos_token": config["eos_token"], "cases": cases}


def main():
    root = Path(__file__).resolve().parent.parent
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(lambda item: make(*item, root), SOURCES.items()))
    (root / "tests/fixtures/tokenizers.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for name, fixture in results.items():
        (root / ".cache" / f"{name}.prompt.txt").write_text(fixture["cases"][0]["rendered"])
    print("Wrote tests/fixtures/tokenizers.json")


if __name__ == "__main__":
    main()
