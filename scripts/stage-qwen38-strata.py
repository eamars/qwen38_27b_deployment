"""Download and verify the pinned ModelOpt checkpoint used by Strata on Windows."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

WORKSPACE = Path(__file__).resolve().parents[1]
REPOSITORY = "jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4"
REVISION = "f24d2b68ff2814f24455ae86717be276619b5664"
MODEL_ROOT = WORKSPACE / "models" / "qwen38-flash-next-uncensored-strata"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_XET_CACHE", str(MODEL_ROOT / ".xet-cache"))
    from huggingface_hub import HfApi, hf_hub_download

    checkpoint = MODEL_ROOT / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"Unexpected revision: {info.sha}")
    files = info.siblings
    manifest_path = MODEL_ROOT / "checkpoint-manifest.json"
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "complete": False,
        "total_bytes": sum(f.size for f in files),
        "files": [],
    }

    def save_manifest():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    def fetch_and_verify(entry):
        path = Path(hf_hub_download(
            REPOSITORY, entry.rfilename, revision=REVISION, local_dir=checkpoint
        ))
        if path.stat().st_size != entry.size:
            raise RuntimeError(f"Size mismatch: {path}")
        if entry.lfs:
            expected = entry.lfs.sha256
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            algorithm = "sha256"
        else:
            expected = entry.blob_id
            digest = hashlib.sha1(f"blob {entry.size}\0".encode())
            digest.update(path.read_bytes())
            actual = digest.hexdigest()
            algorithm = "git-blob-sha1"
        if actual != expected:
            raise RuntimeError(f"Hash mismatch: {path}; expected {expected}, got {actual}")
        return {"name": entry.rfilename, "bytes": entry.size, algorithm: actual}

    save_manifest()
    started = time.monotonic()
    print(f"Downloading {REPOSITORY}@{REVISION}", flush=True)
    print(f"Destination: {checkpoint}; {manifest['total_bytes'] / 2**30:.2f} GiB", flush=True)
    # Fetch metadata first so the converter can be inspected during the bulk transfer.
    ordered_files = sorted(files, key=lambda f: (f.rfilename.endswith(".safetensors"), f.rfilename))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_and_verify, entry) for entry in ordered_files]
        completed_bytes = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            manifest["files"].append(result)
            completed_bytes += result["bytes"]
            save_manifest()
            print(
                f"Verified {len(manifest['files'])}/{len(files)} "
                f"({completed_bytes / 2**30:.2f} GiB): {result['name']}",
                flush=True,
            )
    manifest["files"].sort(key=lambda f: f["name"])
    manifest["complete"] = True
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
    save_manifest()
    print(f"Complete: all {len(files)} files verified in {manifest['elapsed_seconds']} seconds.", flush=True)


if __name__ == "__main__":
    main()
