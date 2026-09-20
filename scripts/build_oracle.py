#!/usr/bin/env python3
"""Build a pinned C++ test oracle under .cache; never linked to the Rust program."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

PINS = {
    "stq": ("sjl623/llama.cpp", "1e411d8f5a1e23525fa3265dfb4bd76265465397"),
    "q2c": ("chaxu01/llama.cpp", "2af64dd00a6689a7bfaf69b4768a944d0ec6bade"),
    "standard": ("ggml-org/llama.cpp", "b23efaa2ef147f547ee75cbf0c621d61904de80e"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PINS)
    parser.add_argument("--jobs", default=3, type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    work = root / ".cache/oracles" / args.profile
    source = work / "source"
    project = work / "project"
    project.mkdir(parents=True, exist_ok=True)
    repo, revision = PINS[args.profile]
    if not source.exists():
        archive = work / "source.tar.gz"
        print(f"Downloading {repo}@{revision}", flush=True)
        with urllib.request.urlopen(f"https://codeload.github.com/{repo}/tar.gz/{revision}", timeout=120) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk():
                    continue
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                destination = source.joinpath(*parts)
                if ".." in parts or not str(destination.resolve()).startswith(str(source.resolve()) + os.sep):
                    raise RuntimeError("Unsafe archive path")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as data, destination.open("wb") as output:
                        shutil.copyfileobj(data, output)
    cmake = f'''cmake_minimum_required(VERSION 3.20)
project(hy_reference LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory("{source}" llama EXCLUDE_FROM_ALL)
add_executable(hy-reference "{root / 'tests/reference/oracle.cpp'}")
target_include_directories(hy-reference PRIVATE "{source / 'ggml/src'}")
target_link_libraries(hy-reference PRIVATE llama ggml)
'''
    (project / "CMakeLists.txt").write_text(cmake)
    env = os.environ.copy()
    env["PATH"] = str(root / ".cache/reference-env/bin") + os.pathsep + env["PATH"]
    off = ["BUILD_SHARED_LIBS", "GGML_METAL", "GGML_CUDA", "GGML_BLAS", "GGML_ACCELERATE", "GGML_OPENMP", "GGML_NATIVE", "GGML_CPU_KLEIDIAI", "GGML_CPU_REPACK", "LLAMA_BUILD_TESTS", "LLAMA_BUILD_EXAMPLES", "LLAMA_BUILD_TOOLS", "LLAMA_BUILD_SERVER", "LLAMA_BUILD_COMMON", "LLAMA_CURL"]
    subprocess.run(["cmake", "-S", str(project), "-B", str(work / "build"), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"] + [f"-D{x}=OFF" for x in off], env=env, check=True)
    subprocess.run(["cmake", "--build", str(work / "build"), "--target", "hy-reference", "-j", str(args.jobs)], env=env, check=True)
    print(work / "build/hy-reference", flush=True)


if __name__ == "__main__":
    main()
