"""Resume and verify a pinned WSL checkpoint with bounded buffers, without loading it."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import quote

os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
os.environ.setdefault("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import httpx
from huggingface_hub import HfApi, get_token, hf_hub_download

REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4"
REVISION = "c1209bda15a6bbc4c68b585e93d40c0d85f50306"
BUFFER = 4 << 20
RANGE = 512 << 20
FLUSH = 64 << 20


def discard_cache(handle):
    handle.flush()
    os.fsync(handle.fileno())
    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def digest_file(path, size):
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(BUFFER):
            sha256.update(block)
            git_sha1.update(block)
            os.posix_fadvise(handle.fileno(), 0, handle.tell(), os.POSIX_FADV_DONTNEED)
    return sha256, git_sha1


def download_ranged(folder, entry, token):
    name, size = entry["name"], entry["size"]
    path = folder / name
    if path.parent != folder:
        raise RuntimeError("Unexpected nested checkpoint path")
    partial = path.with_name(path.name + ".partial")
    if path.exists():
        if path.stat().st_size != size:
            raise RuntimeError(f"Existing file size mismatch: {name}")
        sha256, git_sha1 = digest_file(path, size)
    else:
        headers = {"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"}
        with httpx.Client(headers=headers, follow_redirects=True, timeout=90) as client:
            for attempt in range(6):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > size:
                    raise RuntimeError(f"Partial file is larger than source: {name}")
                if offset:
                    sha256, git_sha1 = digest_file(partial, size)
                else:
                    sha256 = hashlib.sha256()
                    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
                try:
                    with partial.open("ab", buffering=0) as handle:
                        last_report = last_flush = offset
                        while offset < size:
                            end = min(size, offset + RANGE) - 1
                            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(name)}"
                            # Avoid a stale CDN redirect signed for an earlier range.
                            url += f"?download=true&range_start={offset}"
                            request_headers = {"Range": f"bytes={offset}-{end}"}
                            with client.stream("GET", url, headers=request_headers) as response:
                                if response.status_code >= 400:
                                    raise RuntimeError(f"HTTP {response.status_code}: {name}")
                                content_range = f"bytes {offset}-{end}/{size}"
                                ranged = (response.status_code == 206 and
                                          response.headers.get("content-range") == content_range)
                                whole_small = (offset == 0 and end == size - 1 and
                                               response.status_code == 200)
                                if not ranged and not whole_small:
                                    raise RuntimeError(f"Unexpected range response: {name}")
                                for block in response.iter_bytes(chunk_size=BUFFER):
                                    if offset + len(block) > end + 1:
                                        raise RuntimeError(f"Oversized range: {name}")
                                    handle.write(block)
                                    sha256.update(block)
                                    git_sha1.update(block)
                                    offset += len(block)
                                    if offset - last_flush >= FLUSH:
                                        discard_cache(handle)
                                        last_flush = offset
                                if offset != end + 1:
                                    raise RuntimeError(f"Incomplete range: {name}")
                            if offset - last_report >= RANGE or offset == size:
                                print(f"{name}: {offset / (1 << 30):.2f}/{size / (1 << 30):.2f} GiB", flush=True)
                                last_report = offset
                        discard_cache(handle)
                    break
                except (httpx.HTTPError, RuntimeError) as exc:
                    if attempt == 5:
                        raise RuntimeError(f"Download failed after retries: {name} ({type(exc).__name__})") from None
                    print(f"Retrying {name}: {type(exc).__name__}", flush=True)
                    time.sleep(2 ** attempt)
    actual = sha256.hexdigest()
    if entry["sha256"]:
        valid = actual == entry["sha256"]
    else:
        valid = git_sha1.hexdigest() == entry["git_blob_sha1"]
    if not valid:
        raise RuntimeError(f"Checksum mismatch: {name}; file retained for inspection")
    if not path.exists():
        partial.replace(path)
    print(f"Verified: {name}", flush=True)
    return {**entry, "sha256": actual, "verified": True}


def download_file(folder, entry, token):
    path = folder / entry["name"]
    if entry["name"].endswith(".safetensors") and not path.exists():
        print(f"Xet download: {entry['name']}", flush=True)
        for attempt in range(4):
            try:
                hf_hub_download(REPO, entry["name"], revision=REVISION,
                                local_dir=folder, token=token)
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"Xet download failed: {entry['name']} ({type(exc).__name__})") from None
                time.sleep(2 ** attempt)
    verified = download_ranged(folder, entry, token)
    # Old ranged data is redundant only after the final Xet file passes its hash.
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    return verified


def trim_download_cache(folder, stopped):
    while not stopped.wait(3):
        files = list((folder / ".cache/huggingface/download").glob("*.incomplete"))
        files.extend(folder.glob("*.safetensors"))
        for path in files:
            try:
                with path.open("rb", buffering=0) as handle:
                    os.fdatasync(handle.fileno())
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass  # Xet can rename an incomplete file between discovery and open.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(
        "/home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise SystemExit("Run hf auth login in WSL first; do not pass tokens on the command line")
    folder = args.model_dir.resolve()
    if folder.name != "Qwen3.8-Flash-Next-Uncensored-NVFP4":
        raise SystemExit("Use the separate uncensored checkpoint directory")
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (folder / ".staging.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
        entries = [{"name": f.rfilename, "size": f.size,
                    "sha256": getattr(f.lfs, "sha256", None), "git_blob_sha1": f.blob_id}
                   for f in info.siblings if f.rfilename != ".gitattributes"]
        remaining = sum(max(0, e["size"] - (folder / e["name"]).stat().st_size)
                        if (folder / e["name"]).exists() else e["size"] for e in entries)
        if shutil.disk_usage(folder).free < remaining + (5 << 30):
            raise SystemExit("Insufficient disk space for the complete checkpoint plus 5 GiB")
        manifest_path = folder / "staging-manifest.json"
        manifest = {"repo": REPO, "revision": REVISION, "complete": False,
                    "total_bytes": sum(e["size"] for e in entries), "files": []}
        def save():
            temp = manifest_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(manifest, indent=2) + "\n")
            temp.replace(manifest_path)
        save()
        print(f"Staging {manifest['total_bytes'] / (1 << 30):.2f} GiB into {folder}", flush=True)
        # Small metadata first; every indexed shard (including the unused MTP file)
        # stays intact so the original checkpoint index remains reproducible.
        entries.sort(key=lambda e: (e["name"].endswith(".safetensors"), e["name"]))
        stopped = threading.Event()
        trimmer = threading.Thread(target=trim_download_cache, args=(folder, stopped), daemon=True)
        trimmer.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download_file, folder, e, token) for e in entries]
                for future in concurrent.futures.as_completed(futures):
                    manifest["files"].append(future.result())
                    save()
        finally:
            stopped.set()
            trimmer.join(timeout=10)
        manifest["files"].sort(key=lambda e: e["name"])
        manifest["complete"] = True
        save()
        print("Checkpoint staged and verified. No model was initialized.", flush=True)


if __name__ == "__main__":
    main()
