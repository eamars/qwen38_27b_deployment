"""Reproduce the isolated loader source without changing the existing FreeToken install."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runtime/freetoken-a80b4d3"
TARGET = ROOT / "runtime/freetoken-qwen38-uncensored"
REVISION = "af71ba43206e124f5ff6419b47ee36c6e9981078"
PATCH = ROOT / "scripts/patches/qwen38-uncensored-loader.patch"


def main():
    if TARGET.exists():
        manifest = json.loads((TARGET / "runtime-manifest.json").read_text())
        if manifest["patch_sha256"] != hashlib.sha256(PATCH.read_bytes()).hexdigest():
            raise SystemExit("Existing runtime uses a different patch; inspect before replacing")
        for name, expected in manifest["sha256"].items():
            if hashlib.sha256((TARGET / name).read_bytes()).hexdigest() != expected:
                raise SystemExit(f"Existing runtime differs from its manifest: {name}")
        print("Existing isolated runtime verified; no files changed.")
        return
    revision = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected base revision {REVISION}; found {revision}")
    dirty = subprocess.check_output(["git", "-C", str(BASE), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise SystemExit("Base runtime has tracked changes; inspect before copying")
    shutil.copytree(BASE / "python", TARGET / "python", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(BASE / "LICENSE", TARGET / "LICENSE")
    # The Windows checkout has CRLF; the portable patch uses LF context.
    for name in ("config.py", "weight.py", "ple_disk.py"):
        path = TARGET / "python/freetoken/models/qwen4_exp" / name
        path.write_text(path.read_text())
    subprocess.run(["git", "apply", "--no-index", str(PATCH)], cwd=TARGET, check=True)
    hashes = {p.relative_to(TARGET).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in TARGET.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    manifest = {"base_revision": REVISION, "base_source": str(BASE),
                "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(), "sha256": hashes}
    (TARGET / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {TARGET}. Shared venv, compiled kernels and CUDA toolkit reused. No model loaded.")


if __name__ == "__main__":
    main()
