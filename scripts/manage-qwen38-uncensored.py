"""Preflight and supervise only the isolated uncensored FreeToken process."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ALIAS = "qwen38-next-uncensored-freetoken"
REVISION = "c1209bda15a6bbc4c68b585e93d40c0d85f50306"
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime/freetoken-qwen38-uncensored"
VENV = Path("/home/rba90/.freetoken-qwen38/venv")


def port_available(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def start_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def owned(record):
    try:
        pid = int(record["pid"])
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        alias_index = command.index(b"--served-model-name") + 1
        return (pid > 1 and os.getpgid(pid) == pid and
                start_ticks(pid) == record["start_ticks"] and
                command[alias_index] == ALIAS.encode() and
                str(VENV / "bin/ft").encode() in command)
    except (OSError, ValueError, KeyError, IndexError):
        return False


def stop_managed(pid_file):
    if not pid_file.exists():
        print("No managed uncensored process recorded.")
        return
    record = json.loads(pid_file.read_text())
    if not owned(record):
        raise RuntimeError("PID record does not match a live uncensored process; no signal sent")
    os.killpg(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and owned(record):
        time.sleep(.2)
    if owned(record):
        os.killpg(record["pid"], signal.SIGKILL)
    # A concurrent launcher may have written a new record after the old exit.
    if pid_file.exists() and json.loads(pid_file.read_text()) == record:
        pid_file.unlink()
    print("Stopped the managed uncensored process.")


def check_assets(model):
    manifest = json.loads((model / "staging-manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("revision") != REVISION:
        raise RuntimeError("Pinned uncensored checkpoint has not finished staging and verification")
    recorded = {entry["name"]: entry for entry in manifest["files"]}
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    required = set(index.values()) | {"config.json", "tokenizer.json", "tokenizer_config.json",
                                      "model.safetensors.index.json", "chat_template.jinja"}
    for name in required:
        entry = recorded.get(name)
        if not entry or not entry.get("verified") or (model / name).stat().st_size != entry["size"]:
            raise RuntimeError(f"Missing, unverified, or incomplete checkpoint file: {name}")
    provenance = json.loads((RUNTIME / "runtime-manifest.json").read_text())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Isolated runtime changed since preparation: {name}")
    if not os.access(VENV / "bin/ft", os.X_OK):
        raise RuntimeError("Shared FreeToken executable is missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("check", "start", "stop"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise RuntimeError("Invalid port")
    pid_file = Path(f"/tmp/qwen38-flash-next-uncensored-freetoken-{args.port}.json")
    if args.action == "stop":
        stop_managed(pid_file)
        return 0
    if args.action == "start" and not port_available(args.port):
        raise RuntimeError(f"Port {args.port} is occupied; no model started and no process stopped")
    check_assets(args.model)
    gpu_list = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True)
    if not any(row.startswith(args.gpu + ",") for row in gpu_list.splitlines()):
        raise RuntimeError(f"GPU UUID not found: {args.gpu}")
    if args.action == "check":
        print(json.dumps({"assets_ready": True, "port": args.port,
                          "port_available": port_available(args.port),
                          "model_initialized": False, "runtime": str(RUNTIME)}, indent=2))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command[:2] != [str(VENV / "bin/ft"), "serve"] or ALIAS not in command:
        raise RuntimeError("Expected the uncensored FreeToken serve command")
    with Path(f"/tmp/qwen38-flash-next-uncensored-{args.port}.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if pid_file.exists() and owned(json.loads(pid_file.read_text())):
            raise RuntimeError("An uncensored process is already recorded")
        if not port_available(args.port):
            raise RuntimeError(f"Port {args.port} became occupied; no model started")
        env = os.environ.copy()
        env.update({"CUDA_HOME": "/usr/local/cuda-13.3", "TVM_FFI_CUDA_ARCH_LIST": "12.0",
                    "CUDA_VISIBLE_DEVICES": args.gpu,
                    "PYTHONPATH": str(RUNTIME / "python"),
                    "PATH": f"{VENV}/bin:/usr/local/cuda-13.3/bin:" + env.get("PATH", ""),
                    "LD_LIBRARY_PATH": "/usr/local/cuda-13.3/targets/x86_64-linux/lib:" + env.get("LD_LIBRARY_PATH", "")})
        child = subprocess.Popen(command, env=env, start_new_session=True)
        record = {"pid": child.pid, "start_ticks": start_ticks(child.pid), "model": str(args.model)}
        temp = pid_file.with_suffix(".tmp")
        temp.write_text(json.dumps(record))
        temp.replace(pid_file)
        try:
            return child.wait()
        except KeyboardInterrupt:
            if owned(record):
                stop_managed(pid_file)
            return 130
        finally:
            if pid_file.exists() and json.loads(pid_file.read_text()) == record:
                pid_file.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from None
