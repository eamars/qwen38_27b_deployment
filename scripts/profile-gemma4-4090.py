"""Find a usable RTX 4090 configuration for Gemma 4 Persona GGUF + MTP.

The profiler is deliberately dry-run by default. It validates the local
runtime/assets and prints every command without starting llama-server. Pass
--run only when the RTX 4090 is available; each case then starts a fresh
server, verifies /health and /v1/models, measures generation, records MTP
acceptance metrics, and captures observed GPU memory use.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
RUNTIME_DEFAULT = (
    WORKSPACE
    / "runtime"
    / "llama.cpp-dflash2"
    / "build-dflash2"
    / "bin"
    / "Release"
    / "llama-server.exe"
)
TARGET_DEFAULT = (
    WORKSPACE
    / "models"
    / "gemma4"
    / "Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf"
)
DRAFTER_DEFAULT = WORKSPACE / "models" / "gemma4" / "mtp-gemma-4-31B-it-Q8_0.gguf"
OUTPUT_DEFAULT = WORKSPACE / "benchmarks" / "gemma4-4090-mtp-profile.json"
GPU_UUID_DEFAULT = "GPU-eed52936-813f-8d68-1654-bfb56cb42bc3"
GPU_NAME_DEFAULT = "RTX 4090"
DEFAULT_KV_PROFILES = "q8_0:q8_0,q8_0:q4_0,q4_0:q4_0"
DEFAULT_DRAFT_KV_PROFILE = "q8_0:q8_0"
DEFAULT_DRAFT_N_MAX_VALUES = "2,3"
ALLOWED_CACHE_TYPES = {
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
}
DEFAULT_PROMPT = (
    "You are a coding assistant. Write a compact Python implementation for a "
    "streaming JSON-lines validator, then explain its handling of malformed UTF-8, "
    "partial final lines, duplicate keys, and oversized records. Keep the answer "
    "precise and practical."
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def project_path(value: Path, label: str, require_file: bool = True) -> Path:
    path = value.expanduser().resolve()
    try:
        path.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside the project: {path}") from exc
    if require_file and not path.is_file():
        raise SystemExit(f"{label} is missing: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile Gemma 4 Isometry-Fabled-Persona on one RTX 4090 across "
            "target KV-cache types and Google MTP draft lengths."
        )
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--run",
        action="store_true",
        help="start llama-server and run the matrix; omitted by default for safety",
    )
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="print validated commands without starting a server (the default)",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="query nvidia-smi and validate the RTX 4090 even in dry-run mode",
    )
    parser.add_argument("--runtime", type=Path, default=RUNTIME_DEFAULT)
    parser.add_argument("--target", type=Path, default=TARGET_DEFAULT)
    parser.add_argument("--drafter", type=Path, default=DRAFTER_DEFAULT)
    parser.add_argument("--gpu-uuid", default=GPU_UUID_DEFAULT)
    parser.add_argument("--bind-address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--context-size", type=int, default=56320)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument(
        "--kv-profiles",
        default=DEFAULT_KV_PROFILES,
        help="comma-separated target K:V profiles, for example q8_0:q8_0,q4_0:q4_0",
    )
    parser.add_argument(
        "--draft-kv-profile",
        default=DEFAULT_DRAFT_KV_PROFILE,
        help="MTP drafter K:V cache profile used by every MTP case",
    )
    parser.add_argument(
        "--draft-n-max-values",
        default=DEFAULT_DRAFT_N_MAX_VALUES,
        help="comma-separated MTP draft lengths, for example 2,3",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def parse_kv_profile(value: str, label: str) -> dict[str, str]:
    parts = [part.strip().lower() for part in value.split(":")]
    if len(parts) != 2 or any(part not in ALLOWED_CACHE_TYPES for part in parts):
        allowed = ", ".join(sorted(ALLOWED_CACHE_TYPES))
        raise SystemExit(
            f"{label} must contain K:V cache types from [{allowed}], got {value!r}"
        )
    return {"k": parts[0], "v": parts[1], "label": f"{parts[0]}-{parts[1]}"}


def parse_kv_profiles(value: str) -> list[dict[str, str]]:
    profiles: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        profile = parse_kv_profile(item, "--kv-profiles")
        if profile["label"] not in seen:
            profiles.append(profile)
            seen.add(profile["label"])
    if not profiles:
        raise SystemExit("--kv-profiles cannot be empty")
    return profiles


def parse_draft_n_max_values(value: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise SystemExit(
                f"--draft-n-max-values must contain integers, got {item!r}"
            ) from exc
        if number < 1 or number > 16:
            raise SystemExit(f"MTP draft length must be between 1 and 16, got {number}")
        if number not in seen:
            values.append(number)
            seen.add(number)
    if not values:
        raise SystemExit("--draft-n-max-values cannot be empty")
    return values


def validate_args(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], list[dict[str, str]], dict[str, str], list[int]]:
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    for name in ("context_size", "batch_size", "ubatch_size", "repetitions", "max_tokens"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.ubatch_size > args.batch_size:
        raise SystemExit("--ubatch-size cannot exceed --batch-size")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("timeouts must be positive")

    paths = {
        "runtime": project_path(args.runtime, "--runtime"),
        "target": project_path(args.target, "--target"),
        "drafter": project_path(args.drafter, "--drafter"),
        "output": project_path(args.output, "--output", require_file=False),
    }
    if "Isometry-Fabled-Persona" not in paths["target"].name:
        raise SystemExit(
            "--target must be an Isometry-Fabled-Persona GGUF; refusing an unmodified Gemma target"
        )
    if args.prompt_file is not None:
        paths["prompt"] = project_path(args.prompt_file, "--prompt-file")
    kv_profiles = parse_kv_profiles(args.kv_profiles)
    draft_kv_profile = parse_kv_profile(args.draft_kv_profile, "--draft-kv-profile")
    n_max_values = parse_draft_n_max_values(args.draft_n_max_values)
    return paths, kv_profiles, draft_kv_profile, n_max_values


def runtime_version(runtime: Path) -> str:
    try:
        completed = subprocess.run(
            [str(runtime), "--version"],
            cwd=str(runtime.parent),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    output = (completed.stdout or completed.stderr).strip()
    return output if output else f"version command exit={completed.returncode}"


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def build_cases(
    kv_profiles: list[dict[str, str]],
    draft_kv_profile: dict[str, str],
    draft_n_max_values: list[int],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for kv in kv_profiles:
        cases.append(
            {
                "id": f"target-only-{kv['label']}",
                "mode": "target-only",
                "target_k": kv["k"],
                "target_v": kv["v"],
                "draft_k": None,
                "draft_v": None,
                "draft_n_max": None,
            }
        )
        for n_max in draft_n_max_values:
            cases.append(
                {
                    "id": f"mtp-{kv['label']}-n{n_max}",
                    "mode": "mtp",
                    "target_k": kv["k"],
                    "target_v": kv["v"],
                    "draft_k": draft_kv_profile["k"],
                    "draft_v": draft_kv_profile["v"],
                    "draft_n_max": n_max,
                }
            )
    return cases


def build_command(
    runtime: Path,
    target: Path,
    drafter: Path,
    args: argparse.Namespace,
    case: dict[str, Any],
) -> list[str]:
    alias = f"gemma4-31b-isometry-fabled-persona-4090-{case['id']}"
    command = [
        str(runtime),
        "--model",
        str(target),
    ]
    if case["mode"] == "mtp":
        command.extend(
            [
                "--spec-draft-model",
                str(drafter),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(case["draft_n_max"]),
                "--spec-draft-device",
                "CUDA0",
                "--spec-draft-ngl",
                "all",
                "--spec-draft-type-k",
                case["draft_k"],
                "--spec-draft-type-v",
                case["draft_v"],
            ]
        )
    command.extend(
        [
            "--alias",
            alias,
            "--host",
            args.bind_address,
            "--port",
            str(args.port),
            "--device",
            "CUDA0",
            "--split-mode",
            "none",
            "--gpu-layers",
            "all",
            "--ctx-size",
            str(args.context_size),
            "--parallel",
            "1",
            "--kv-unified",
            "--flash-attn",
            "on",
            "--cache-type-k",
            case["target_k"],
            "--cache-type-v",
            case["target_v"],
            "--batch-size",
            str(args.batch_size),
            "--ubatch-size",
            str(args.ubatch_size),
            "--fit",
            "off",
            "--no-mmproj",
            "--no-context-shift",
            "--jinja",
            "--reasoning",
            "auto",
            "--reasoning-preserve",
            "--metrics",
        ]
    )
    return command


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return exc.code, raw
    except urllib.error.URLError as exc:
        return 0, str(exc)


def read_metrics(port: int) -> dict[str, float]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return {}

    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(None, 1)
        name = name.split("{", 1)[0]
        try:
            values[name] = values.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return values


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def stream_chat(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "top_k": 1,
        "seed": 1234,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    before = read_metrics(port)
    started = time.perf_counter()
    first_token_at: float | None = None
    events = 0
    output_parts: list[str] = []
    usage: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == b"[DONE]":
                    continue
                event = json.loads(raw.decode("utf-8"))
                events += 1
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_parts.append(piece)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"completion request returned HTTP {exc.code}: {body[:1000]}") from exc

    finished = time.perf_counter()
    after = read_metrics(port)
    metric_changes = {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in set(before) | set(after)
    }

    prompt_tokens_metric = metric_delta(before, after, "llamacpp:prompt_tokens_total")
    predicted_tokens_metric = metric_delta(before, after, "llamacpp:tokens_predicted_total")
    prompt_seconds = metric_delta(before, after, "llamacpp:prompt_seconds_total")
    predicted_seconds = metric_delta(before, after, "llamacpp:tokens_predicted_seconds_total")
    drafts = metric_delta(before, after, "llamacpp:spec_decode_num_drafts_total")
    drafted_tokens = metric_delta(before, after, "llamacpp:spec_decode_num_draft_tokens_total")
    accepted_tokens = metric_delta(before, after, "llamacpp:spec_decode_num_accepted_tokens_total")

    prompt_tokens = (usage or {}).get("prompt_tokens")
    if prompt_tokens is None and prompt_tokens_metric > 0:
        prompt_tokens = int(round(prompt_tokens_metric))
    predicted_tokens = (usage or {}).get("completion_tokens")
    if predicted_tokens is None and predicted_tokens_metric > 0:
        predicted_tokens = int(round(predicted_tokens_metric))

    ttft_seconds = None if first_token_at is None else first_token_at - started
    decode_wall_seconds = finished - (first_token_at or started)
    if predicted_seconds > 0 and predicted_tokens is not None:
        tg_tokens_per_second = float(predicted_tokens) / predicted_seconds
    elif predicted_tokens and decode_wall_seconds > 0:
        tg_tokens_per_second = float(predicted_tokens) / decode_wall_seconds
    else:
        tg_tokens_per_second = None

    return {
        "measured_at": utc_now(),
        "model": model,
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "events": events,
        "output_chars": len("".join(output_parts)),
        "usage": usage,
        "ttft_seconds": ttft_seconds,
        "wall_seconds": finished - started,
        "prompt_tokens": prompt_tokens,
        "pp_tokens_per_second": (
            None
            if prompt_seconds <= 0 or prompt_tokens is None
            else float(prompt_tokens) / prompt_seconds
        ),
        "predicted_tokens": predicted_tokens,
        "tg_tokens_per_second": tg_tokens_per_second,
        "draft_verification_steps": drafts,
        "drafted_tokens": drafted_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "acceptance_ratio": None if drafted_tokens <= 0 else accepted_tokens / drafted_tokens,
        "mean_accepted_per_verification": None if drafts <= 0 else accepted_tokens / drafts,
        "metrics_delta": metric_changes,
    }


def log_tail(path: Path, limit: int = 8000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "<log unavailable>"
    return text[-limit:]


def wait_for_server(
    process: subprocess.Popen[bytes],
    port: int,
    timeout: float,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status: Any = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}; log {log_path}\n{log_tail(log_path)}"
            )
        status, body = request_json(f"http://127.0.0.1:{port}/health", None, 5)
        last_status = {"status": status, "body": body}
        if status == 200:
            return last_status
        time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for llama-server on port {port}; last response {last_status}; log {log_path}"
    )


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def gpu_snapshot(uuid: str) -> dict[str, Any]:
    nvidia = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if nvidia is None:
        return {}
    completed = subprocess.run(
        [
            nvidia,
            "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6 or fields[2] != uuid:
            continue
        try:
            used = float(fields[3])
            total = float(fields[4])
            utilization = float(fields[5])
        except ValueError:
            used = None
            total = None
            utilization = None
        return {
            "index": fields[0],
            "name": fields[1],
            "uuid": fields[2],
            "memory_used_mib": used,
            "memory_total_mib": total,
            "utilization_gpu_percent": utilization,
        }
    return {"error": f"GPU UUID not found in nvidia-smi output: {uuid}"}


def validate_gpu(uuid: str) -> dict[str, Any]:
    nvidia = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if nvidia is None:
        raise SystemExit("nvidia-smi was not found in PATH")
    completed = subprocess.run(
        [nvidia, "--query-gpu=index,name,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"nvidia-smi failed: {completed.stderr.strip()}")
    matches = [
        line
        for line in completed.stdout.splitlines()
        if GPU_NAME_DEFAULT in line and uuid in line
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one {GPU_NAME_DEFAULT} with UUID {uuid}; matches={matches}"
        )
    return gpu_snapshot(uuid)


def assert_port_available(port: int, bind_address: str) -> None:
    probe_address = "127.0.0.1" if bind_address in {"0.0.0.0", "::"} else bind_address
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((probe_address, port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is already in use; stop the existing server or choose --port"
            ) from exc


def median_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return None if not values else statistics.median(values)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(records),
        "median_ttft_seconds": median_field(records, "ttft_seconds"),
        "median_wall_seconds": median_field(records, "wall_seconds"),
        "median_pp_tokens_per_second": median_field(records, "pp_tokens_per_second"),
        "median_tg_tokens_per_second": median_field(records, "tg_tokens_per_second"),
        "median_drafted_tokens": median_field(records, "drafted_tokens"),
        "median_accepted_draft_tokens": median_field(records, "accepted_draft_tokens"),
        "median_acceptance_ratio": median_field(records, "acceptance_ratio"),
        "median_mean_accepted_per_verification": median_field(
            records, "mean_accepted_per_verification"
        ),
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def run_case(
    case: dict[str, Any],
    command: list[str],
    args: argparse.Namespace,
    prompts: list[str],
    output_path: Path,
) -> dict[str, Any]:
    assert_port_available(args.port, args.bind_address)
    log_path = output_path.with_name(f"{output_path.stem}-{case['id']}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(Path(command[0]).parent),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    try:
        health = wait_for_server(process, args.port, args.startup_timeout, log_path)
        status, models = request_json(f"http://127.0.0.1:{args.port}/v1/models", None, 30)
        if status != 200:
            raise RuntimeError(f"/v1/models returned HTTP {status}: {models}")

        gpu_samples = [gpu_snapshot(args.gpu_uuid)]
        model_name = f"gemma4-31b-isometry-fabled-persona-4090-{case['id']}"
        for warmup_index in range(args.warmup):
            warmup_prompt = (
                prompts[0]
                + f"\n\nWarm-up request {warmup_index + 1}; answer in one sentence."
            )
            stream_chat(
                args.port,
                model_name,
                warmup_prompt,
                min(args.max_tokens, 64),
                args.request_timeout,
            )
            gpu_samples.append(gpu_snapshot(args.gpu_uuid))

        records: list[dict[str, Any]] = []
        for repetition, prompt in enumerate(prompts, start=1):
            record = stream_chat(
                args.port,
                model_name,
                prompt,
                args.max_tokens,
                args.request_timeout,
            )
            record["case_id"] = case["id"]
            record["mode"] = case["mode"]
            record["repetition"] = repetition
            records.append(record)
            gpu_samples.append(gpu_snapshot(args.gpu_uuid))
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "repetition": repetition,
                        "ttft_seconds": record["ttft_seconds"],
                        "tg_tokens_per_second": record["tg_tokens_per_second"],
                        "acceptance_ratio": record["acceptance_ratio"],
                        "mean_accepted_per_verification": record[
                            "mean_accepted_per_verification"
                        ],
                    },
                    sort_keys=True,
                )
            )

        memory_samples = [
            sample.get("memory_used_mib")
            for sample in gpu_samples
            if isinstance(sample.get("memory_used_mib"), (int, float))
        ]
        return {
            "case": case,
            "health": health,
            "models": models,
            "gpu_samples": gpu_samples,
            "peak_memory_used_mib_observed": max(memory_samples) if memory_samples else None,
            "records": records,
            "summary": summarize(records),
            "log": str(log_path),
        }
    finally:
        stop_server(process)


def build_comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = {
        case_id: result["summary"]
        for case_id, result in results.items()
        if result.get("summary") is not None
    }
    comparison: dict[str, Any] = {"cases": summaries}
    speedups: list[dict[str, Any]] = []
    for case_id, result in results.items():
        case = result["case"]
        if case["mode"] != "mtp":
            continue
        baseline_id = f"target-only-{case['target_k']}-{case['target_v']}"
        mtp_summary = result.get("summary", {})
        baseline = summaries.get(baseline_id, {})
        speedups.append(
            {
                "case": case_id,
                "baseline": baseline_id,
                "tg_speedup_mtp_over_target_only": ratio(
                    mtp_summary.get("median_tg_tokens_per_second"),
                    baseline.get("median_tg_tokens_per_second"),
                ),
                "wall_time_ratio_mtp_over_target_only": ratio(
                    mtp_summary.get("median_wall_seconds"),
                    baseline.get("median_wall_seconds"),
                ),
                "median_acceptance_ratio": mtp_summary.get("median_acceptance_ratio"),
                "median_mean_accepted_per_verification": mtp_summary.get(
                    "median_mean_accepted_per_verification"
                ),
            }
        )
    comparison["mtp_vs_target_only"] = speedups
    usable_speedups = [
        item
        for item in speedups
        if item["tg_speedup_mtp_over_target_only"] is not None
    ]
    comparison["best_mtp_case_by_generation_speed"] = (
        max(usable_speedups, key=lambda item: item["tg_speedup_mtp_over_target_only"])
        if usable_speedups
        else None
    )
    usable_acceptance = [
        item for item in speedups if item["median_mean_accepted_per_verification"] is not None
    ]
    comparison["best_mtp_case_by_acceptance"] = (
        max(
            usable_acceptance,
            key=lambda item: item["median_mean_accepted_per_verification"],
        )
        if usable_acceptance
        else None
    )
    return comparison


def main() -> int:
    args = parse_args()
    paths, kv_profiles, draft_kv_profile, draft_n_max_values = validate_args(args)
    prompt = (
        DEFAULT_PROMPT
        if args.prompt_file is None
        else paths["prompt"].read_text(encoding="utf-8")
    )
    if not prompt.strip():
        raise SystemExit("The benchmark prompt cannot be empty")
    prompts = [
        prompt + f"\n\nBenchmark repetition {index}."
        for index in range(1, args.repetitions + 1)
    ]

    cases = build_cases(kv_profiles, draft_kv_profile, draft_n_max_values)
    commands = {
        case["id"]: build_command(
            paths["runtime"], paths["target"], paths["drafter"], args, case
        )
        for case in cases
    }
    version = runtime_version(paths["runtime"])
    gpu_info: dict[str, Any] = {}
    if args.check_gpu or args.run:
        gpu_info = validate_gpu(args.gpu_uuid)

    if not args.run:
        print("Dry run: no llama-server process was started and no model was loaded.")
        print(f"Runtime: {version}")
        print(f"Target: {paths['target']} ({paths['target'].stat().st_size} bytes)")
        print(f"Google MTP drafter: {paths['drafter']} ({paths['drafter'].stat().st_size} bytes)")
        if gpu_info:
            print(f"GPU check: {json.dumps(gpu_info, sort_keys=True)}")
        print(
            f"Matrix: {len(kv_profiles)} target KV profiles x "
            f"({1 + len(draft_n_max_values)} cases per profile)"
        )
        for case in cases:
            print(f"{case['id']}: {display_command(commands[case['id']])}")
        print("Pass --run when the RTX 4090 is available to execute the matrix.")
        return 0

    output_path = paths["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for case in cases:
        case_id = case["id"]
        print(f"Starting case {case_id}...")
        try:
            results[case_id] = run_case(
                case, commands[case_id], args, prompts, output_path
            )
        except Exception as exc:  # preserve successful cases and the failing case's log
            errors[case_id] = f"{type(exc).__name__}: {exc}"
            print(f"{case_id} failed: {errors[case_id]}", file=sys.stderr)

    profile = {
        "generated_at": utc_now(),
        "executed": True,
        "workspace": str(WORKSPACE),
        "gpu_uuid": args.gpu_uuid,
        "gpu": gpu_info,
        "runtime": {"path": str(paths["runtime"]), "version": version},
        "assets": {
            "target": {"path": str(paths["target"]), "bytes": paths["target"].stat().st_size},
            "google_mtp_drafter": {
                "path": str(paths["drafter"]),
                "bytes": paths["drafter"].stat().st_size,
            },
        },
        "configuration": {
            "bind_address": args.bind_address,
            "port": args.port,
            "context_size": args.context_size,
            "parallel": 1,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "target_kv_profiles": kv_profiles,
            "draft_kv_profile": draft_kv_profile,
            "draft_n_max_values": draft_n_max_values,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_k": 1,
            "seed": 1234,
        },
        "commands": {case_id: display_command(command) for case_id, command in commands.items()},
        "results": results,
        "errors": errors,
        "comparison": build_comparison(results),
    }
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Profile written: {output_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
