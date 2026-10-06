"""Prepare the verified ModelOpt checkpoint using the pinned Strata converters."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

WORKSPACE = Path(__file__).resolve().parents[1]
RUNTIME = WORKSPACE / "runtime" / "strata-nvfp4"
MODEL = WORKSPACE / "models" / "qwen38-flash-next-uncensored-strata"
REVISION = "f24d2b68ff2814f24455ae86717be276619b5664"
RUNTIME_VERSION, RUNTIME_COMMIT = "0.1.39-nvfp4.3", "b17e5ef"
# Releases whose converters below (and data/draft_vocab.bin) are byte-identical to the pinned one's, so assets they
# prepared stay valid (compared 2026-10-06).
COMPATIBLE_PREPARED = {"0.1.37-nvfp4.2", RUNTIME_VERSION}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-download", action="store_true")
    args = parser.parse_args()
    python = RUNTIME / ".venv-convert" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError("Run scripts/setup-strata-runtime.ps1 first")
    source_manifest = MODEL / "checkpoint-manifest.json"
    while True:
        manifest = json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
        if manifest.get("complete") and manifest.get("revision") == REVISION:
            break
        if not args.wait_for_download:
            raise RuntimeError("The pinned checkpoint is not fully verified; run scripts/stage-qwen38-strata.py first")
        print("Waiting for the verified checkpoint download...", flush=True)
        time.sleep(30)
    checkpoint = MODEL / "checkpoint"
    for item in manifest["files"]:
        path = checkpoint / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Verified checkpoint file is now missing or has changed: {path}")

    build = json.loads((RUNTIME / "engine" / "BUILD.json").read_text())
    if build["version"] != RUNTIME_VERSION or build["commit"] != RUNTIME_COMMIT:
        raise RuntimeError("Unexpected runtime version; use scripts/setup-strata-runtime.ps1")
    state_path = MODEL / "preparation.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "revision": REVISION, "runtime_version": build["version"], "complete": False, "steps": {}
    }
    if state["revision"] != REVISION or state["runtime_version"] not in COMPATIBLE_PREPARED:
        raise RuntimeError("Existing prepared assets belong to a different checkpoint or runtime")
    state["llama_cpp_commit"] = (RUNTIME / "third_party/llama.cpp/COMMIT").read_text().strip()
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")

    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    def run(command):
        print("RUN " + subprocess.list2cmdline([str(x) for x in command]), flush=True)
        subprocess.run([str(python), *map(str, command)], cwd=RUNTIME, env=environment, check=True)

    def step(name, commands, outputs):
        prior = state["steps"].get(name)
        if prior and all(p.is_file() and p.stat().st_size == prior["outputs"].get(str(p.relative_to(MODEL))) for p in outputs):
            print(f"Already prepared: {name}", flush=True)
            return
        state["complete"] = False
        state["steps"].pop(name, None)
        save()
        started = time.monotonic()
        for command in commands:
            run(command)
        for output in outputs:
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty conversion output: {output}")
        state["steps"][name] = {
            "seconds": round(time.monotonic() - started, 2),
            "outputs": {str(p.relative_to(MODEL)): p.stat().st_size for p in outputs},
        }
        save()

    step("ple", [["tools/ple_fp8_pack.py", "--model", checkpoint, "--out", MODEL / "ple-fp8.gguf"]], [MODEL / "ple-fp8.gguf"])
    step("embedding", [["tools/embd_bf16_pack.py", "--model", checkpoint, "--out", MODEL / "token-embd-bf16.gguf"]], [MODEL / "token-embd-bf16.gguf"])
    step("vision", [["third_party/llama.cpp/convert_hf_to_gguf.py", checkpoint, "--mmproj", "--outtype", "f32", "--outfile", MODEL / "mmproj-f32.gguf"]], [MODEL / "mmproj-f32.gguf"])
    gguf = MODEL / "orca-nvfp4.gguf"
    step("gguf", [["tools/nvfp4_convert.py", "--model", checkpoint, "--outfile", gguf]], [gguf])
    # A release extracted inside this workspace inherits its parent .git lookup.
    # Record the bundled converter's COMMIT instead of that unrelated revision.
    conversion_path = gguf.with_suffix(".manifest.json")
    conversion = json.loads(conversion_path.read_text())
    conversion["llama_cpp_commit"] = state["llama_cpp_commit"]
    conversion["ple_table"]["use"] = "--ple-gguf ple-fp8.gguf (checkpoint FP8 E4M3)"
    conversion_path.write_text(json.dumps(conversion, indent=2) + "\n", encoding="utf-8")
    pack = MODEL / "pack"
    step("pack", [["tools/iq_pack.py", "--gguf", gguf, "--out", pack]], [pack / "experts.bin", pack / "dense.bin", pack / "native_experts.txt", pack / "index.txt", pack / "tokenizer/vocab.json", pack / "tokenizer/token_type.json"])
    mtp = MODEL / "mtp"
    step("mtp", [
        ["tools/mtp_extract.py", "--model", checkpoint, "--out", mtp],
        ["tools/mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", mtp / "mtp-q2_0.gguf"],
        ["tools/mtp_rt.py", "--gguf", mtp / "mtp-q2_0.gguf", "--out", mtp / "rt"],
    ], [mtp / "rt/experts.bin", mtp / "rt/dense.bin", mtp / "rt/dense.txt"])
    # The shipped subset (English/code + Cyrillic) holds 27 of 55,328 Han tokens, so Chinese replies drafted ~1.4
    # tokens a round. With CJK added (+171 MiB of draft head): Chinese 110 -> 138 tok/s, English unchanged (2026-10-06).
    step("draft_vocab", [["tools/draft_vocab.py", "--gguf", gguf, "--base", RUNTIME / "data/draft_vocab.bin",
                          "--add", "cjk", "--out", mtp / "rt/draft_vocab.bin"]], [mtp / "rt/draft_vocab.bin"])
    state["complete"] = True
    save()
    print(f"Prepared Strata assets: {MODEL}", flush=True)


if __name__ == "__main__":
    main()
