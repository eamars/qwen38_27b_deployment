#!/usr/bin/env python3
"""Plan or run staged Qwen3.8-Flash-Next deployment profiles.

The default mode is plan-only. The model server is started only when the
caller passes --run. The runner keeps one fresh two-slot server per case, records
latency and speculative-decoding metrics, samples both GPUs and system I/O,
and writes restartable JSON/CSV results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "runtime" / "llama.cpp-qwen4exp" / "build" / "bin" / "Release" / "llama-server.exe"
DEFAULT_RUNTIME_SOURCE = ROOT / "runtime" / "llama.cpp-qwen4exp"
DEFAULT_MODEL = ROOT / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
DEFAULT_MTP_MODEL = ROOT / "models" / "Qwen3.8-Flash-Next-MTP-F16.gguf"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "qwen38_flash_next" / "matrix-e1-e2-short-mid-r3.json"
DEFAULT_GPU0_UUID = "GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0"
DEFAULT_GPU1_UUID = "GPU-eed52936-813f-8d68-1654-bfb56cb42bc3"
MODEL_QUANT_DEFAULT = "UD-Q4_K_XL"
MIN_FREE_VRAM_DEFAULT = 0
FULL_CONTEXT = 262_144
SERVER_PARALLEL = 2
SERVER_CONTEXT = FULL_CONTEXT * SERVER_PARALLEL
EXECUTOR_COUNTS = (1, 2)
FIXED_REPETITIONS = 3

ANCHORS = (
    "FLASH-NEXT-BEGIN-ORCHID-5090",
    "FLASH-NEXT-MIDDLE-QUARTZ-4090",
    "FLASH-NEXT-END-LANTERN-262144",
)
FILLER = (
    "This fixed calibration record belongs to the Qwen3.8-Flash-Next deployment. "
    "Keep the layer split, host PLE placement, Q4-or-better target, two full-context "
    "server slots, and measured VRAM reserve unchanged while reading this record. "
)
CODE_BLOCK = (
    "def replace_range(source, start, end, replacement):\n"
    "    if start < 0 or end < start or end > len(source):\n"
    "        raise ValueError('invalid edit range')\n"
    "    return source[:start] + replacement + source[end:]\n"
)


@dataclass(frozen=True)
class Workload:
    id: str
    target_tokens: int
    output_tokens: int
    description: str


WORKLOADS = {
    "short": Workload("short", 4_000, 1_000, "short interactive"),
    "mid": Workload("mid", 128_000, 2_000, "mid-context reasoning"),
}


@dataclass(frozen=True)
class Case:
    id: str
    stage: str
    context_size: int
    gpu_split: str
    ncmoe: int
    threads: int | None
    threads_batch: int | None
    batch_size: int
    ubatch_size: int
    kv_k: str
    kv_v: str
    mtp: bool
    mtp_precision: str | None
    mtp_device: str | None
    mtp_nmax: int | None


CSV_FIELDS = [
    "timestamp",
    "llama_sha",
    "model_sha256",
    "model_quant",
    "ctx",
    "prompt_tokens",
    "output_tokens",
    "gpu_split",
    "ncmoe",
    "threads",
    "batch",
    "ubatch",
    "kv_k",
    "kv_v",
    "mtp",
    "mtp_precision",
    "mtp_device",
    "mtp_nmax",
    "mtp_acceptance",
    "pp_tps",
    "tg_tps",
    "ttft_ms",
    "wall_ms",
    "gpu0_peak_mb",
    "gpu1_peak_mb",
    "ram_peak_mb",
    "disk_read_mb",
    "case_id",
    "stage",
    "workload",
    "executors",
    "repetition",
    "record_type",
    "prompt_eval_ms",
    "generation_ms",
    "retrieval_correct",
    "parity_exact",
    "status",
    "minimum_free_gpu0_mib",
    "minimum_free_gpu1_mib",
    "server_log",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def workspace_path(value: str | Path, label: str, require_file: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside the project: {path}") from exc
    if require_file and not path.is_file():
        raise SystemExit(f"{label} is missing: {path}")
    return path


def parse_int_list(value: str, label: str, minimum: int = 1) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise SystemExit(f"{label} must contain integers, got {item!r}") from exc
        if number < minimum:
            raise SystemExit(f"{label} values must be >= {minimum}, got {number}")
        if number not in seen:
            result.append(number)
            seen.add(number)
    if not result:
        raise SystemExit(f"{label} cannot be empty")
    return result


def parse_split_list(value: str, label: str = "--gpu-split-values") -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"{label} values must look like 3,2, got {item!r}")
        try:
            numbers = [float(part) for part in parts]
        except ValueError as exc:
            raise SystemExit(f"{label} values must be numeric, got {item!r}") from exc
        if any(not math.isfinite(number) or number <= 0 for number in numbers):
            raise SystemExit(f"{label} values must be positive, got {item!r}")
        normalised = f"{parts[0]},{parts[1]}"
        if normalised not in seen:
            result.append(normalised)
            seen.add(normalised)
    if not result:
        raise SystemExit(f"{label} cannot be empty")
    return result


def parse_batch_list(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2:
            raise SystemExit(f"--batch-values must look like 2048:1024, got {item!r}")
        try:
            batch, ubatch = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise SystemExit(f"--batch-values must contain integers, got {item!r}") from exc
        if batch < 1 or ubatch < 1 or ubatch > batch:
            raise SystemExit(f"batch/ubatch pair is invalid: {item!r}")
        pair = (batch, ubatch)
        if pair not in seen:
            result.append(pair)
            seen.add(pair)
    if not result:
        raise SystemExit("--batch-values cannot be empty")
    return result


def parse_kv_list(value: str, label: str = "--kv-values") -> list[tuple[str, str]]:
    allowed = {"f16", "bf16", "q8_0"}
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2 or any(part not in allowed for part in parts):
            raise SystemExit(f"{label} must contain F16, BF16, or Q8_0 K:V pairs, got {item!r}")
        pair = (parts[0], parts[1])
        if pair not in seen:
            result.append(pair)
            seen.add(pair)
    if not result:
        raise SystemExit(f"{label} cannot be empty")
    return result


def parse_device_list(value: str) -> list[str]:
    result = []
    for item in value.split(","):
        item = item.strip().upper()
        if item not in {"CUDA0", "CUDA1"}:
            raise SystemExit(f"--mtp-devices only supports CUDA0 or CUDA1, got {item!r}")
        if item not in result:
            result.append(item)
    if not result:
        raise SystemExit("--mtp-devices cannot be empty")
    return result


def safe_id(value: str) -> str:
    return value.replace(",", "-").replace(".", "p").replace(":", "-")


def base_case_values(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "context_size": args.stage_context,
        "gpu_split": args.base_gpu_split,
        "ncmoe": args.base_ncmoe,
        "threads": args.threads if args.threads > 0 else None,
        "threads_batch": args.threads_batch if args.threads_batch > 0 else None,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "kv_k": "f16",
        "kv_v": "f16",
        "mtp": False,
        "mtp_precision": None,
        "mtp_device": None,
        "mtp_nmax": None,
    }


def make_case(case_id: str, stage: str, values: dict[str, Any]) -> Case:
    if values["context_size"] != FULL_CONTEXT:
        raise SystemExit(
            f"case {case_id} requested ctx {values['context_size']}; all profiling cases "
            f"must use the full deployment context {FULL_CONTEXT}"
        )
    if values["mtp"] and values["mtp_precision"] not in {"f16", "bf16"}:
        raise SystemExit("MTP cases must use f16 or bf16 precision")
    if values["ubatch_size"] > values["batch_size"]:
        raise SystemExit(f"case {case_id} has ubatch larger than batch")
    return Case(id=case_id, stage=stage, **values)


def build_stage_cases(stage: str, args: argparse.Namespace) -> list[Case]:
    base = base_case_values(args)
    if stage != "context":
        raise SystemExit(f"unsupported profiling stage: {stage}")
    return [
        make_case(
            f"context-{context_size}",
            stage,
            {**base, "context_size": context_size},
        )
        for context_size in parse_int_list(args.context_values, "--context-values")
    ]


def build_cases(args: argparse.Namespace) -> list[Case]:
    return build_stage_cases(args.stage, args)


def select_workload_ids(args: argparse.Namespace) -> list[str]:
    del args
    return ["short", "mid"]


def nominally_compatible_workloads(case: Case, workload_ids: list[str]) -> list[str]:
    return [
        workload_id
        for workload_id in workload_ids
        if WORKLOADS[workload_id].target_tokens + WORKLOADS[workload_id].output_tokens
        <= case.context_size
    ]


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def command_for(case: Case, args: argparse.Namespace) -> list[str]:
    command = [
        str(args.runtime),
        "--model",
        str(args.model),
        "--load-mode",
        "none",
        "--alias",
        f"qwen38-flash-next-{case.id}",
        "--host",
        args.bind_address,
        "--port",
        str(args.port),
        "--device",
        "CUDA0,CUDA1",
        "--split-mode",
        "layer",
        "--tensor-split",
        case.gpu_split,
        "--gpu-layers",
        "all",
        "--ctx-size",
        str(SERVER_CONTEXT),
        "--parallel",
        str(SERVER_PARALLEL),
        "--override-tensor",
        "per_layer_token_embd=CPU",
        "--n-cpu-moe",
        str(case.ncmoe),
        "--flash-attn",
        "on",
        "--cache-type-k",
        case.kv_k,
        "--cache-type-v",
        case.kv_v,
        "--batch-size",
        str(case.batch_size),
        "--ubatch-size",
        str(case.ubatch_size),
        "--fit",
        "off",
        "--cache-ram",
        "0",
        "--no-mmproj",
        "--no-context-shift",
        "--jinja",
        "--metrics",
    ]
    if case.threads is not None:
        command.extend(["--threads", str(case.threads)])
    if case.threads_batch is not None:
        command.extend(["--threads-batch", str(case.threads_batch)])
    if case.mtp:
        command.extend(
            [
                "--spec-draft-model",
                str(args.mtp_model),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-device",
                str(case.mtp_device),
                "--spec-draft-ngl",
                "all",
                "--spec-draft-type-k",
                str(case.mtp_precision),
                "--spec-draft-type-v",
                str(case.mtp_precision),
                "--spec-draft-n-max",
                str(case.mtp_nmax),
            ]
        )
    return command


def run_no_model_command(command: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except OSError as exc:
        return 127, str(exc)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def runtime_version(runtime: Path) -> tuple[str | None, str]:
    code, output = run_no_model_command([str(runtime), "--version"])
    text = output.strip()
    match = re.search(r"commit\s+([0-9a-f]{7,40})", text, re.IGNORECASE)
    return (match.group(1) if match else None), text if text else f"exit={code}"


def git_sha(source: Path) -> str | None:
    if not source.is_dir():
        return None
    code, output = run_no_model_command(["git", "-C", str(source), "rev-parse", "HEAD"])
    value = output.strip()
    return value if code == 0 and re.fullmatch(r"[0-9a-f]{40}", value, re.IGNORECASE) else None


def gpu_rows() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    code, output = run_no_model_command(command)
    if code != 0:
        raise RuntimeError(f"nvidia-smi failed: {output.strip()}")
    rows: list[dict[str, str]] = []
    for raw in output.splitlines():
        fields = next(csv.reader([raw]))
        fields = [field.strip() for field in fields]
        if len(fields) < 6:
            continue
        rows.append(
            {
                "index": fields[0],
                "name": fields[1],
                "uuid": fields[2],
                "memory_total": fields[3],
                "memory_used": fields[4],
                "memory_free": fields[5],
            }
        )
    return rows


def validate_gpu_mapping(gpu0_uuid: str, gpu1_uuid: str) -> list[dict[str, str]]:
    rows = gpu_rows()
    first = [row for row in rows if row["uuid"] == gpu0_uuid]
    second = [row for row in rows if row["uuid"] == gpu1_uuid]
    if len(first) != 1 or "RTX 5090" not in first[0]["name"]:
        raise RuntimeError(f"expected CUDA0 source GPU to be RTX 5090 with UUID {gpu0_uuid}")
    if len(second) != 1 or "RTX 4090" not in second[0]["name"]:
        raise RuntimeError(f"expected CUDA1 source GPU to be RTX 4090 with UUID {gpu1_uuid}")
    return rows


def validate_runtime(
    runtime: Path,
    gpu0_uuid: str,
    gpu1_uuid: str,
    require_mtp: bool = False,
) -> dict[str, Any]:
    baseline_options = (
        "--load-mode",
        "--n-cpu-moe",
        "--split-mode",
        "--tensor-split",
        "--override-tensor",
        "--cache-type-k",
        "--cache-type-v",
        "--flash-attn",
        "--fit",
        "--cache-ram",
        "--metrics",
    )
    mtp_options = (
        "--spec-type",
        "--spec-draft-model",
        "--spec-draft-device",
        "--spec-draft-ngl",
        "--spec-draft-type-k",
        "--spec-draft-type-v",
    )
    required_options = baseline_options + (mtp_options if require_mtp else ())
    code, help_text = run_no_model_command([str(runtime), "--help"])
    if code != 0:
        raise RuntimeError(f"runtime --help failed: {help_text.strip()}")
    missing = [option for option in required_options if option not in help_text]
    if missing:
        raise RuntimeError(f"runtime is missing required options: {', '.join(missing)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{gpu0_uuid},{gpu1_uuid}"
    code, devices = run_no_model_command([str(runtime), "--list-devices"], env=env)
    if code != 0 or "CUDA0" not in devices or "CUDA1" not in devices:
        raise RuntimeError(f"runtime device isolation check failed:\n{devices}")
    if "RTX 5090" not in devices or "RTX 4090" not in devices:
        raise RuntimeError(f"runtime device names do not match the expected order:\n{devices}")
    commit, version = runtime_version(runtime)
    return {
        "version": version,
        "commit": commit,
        "devices": devices,
        "required_options": list(required_options),
        "mtp_options_required": require_mtp,
    }


def validate_model_shards(model: Path) -> list[str]:
    if not model.is_file():
        raise RuntimeError(f"main model is missing: {model}")
    match = re.match(r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<count>\d{5})\.gguf$", model.name, re.IGNORECASE)
    if not match:
        return [model.name]
    expected_count = int(match.group("count"))
    prefix = match.group("prefix")
    siblings = sorted(model.parent.glob(f"{prefix}-*-of-{expected_count:05d}.gguf"))
    if len(siblings) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} GGUF shards matching {prefix}-*-of-{expected_count:05d}.gguf; found {len(siblings)}"
        )
    return [path.name for path in siblings]


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def metrics(port: int) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15) as response:
            raw = response.read().decode("utf-8")
    except Exception:
        return {}
    result: dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(None, 1)
        name = name.split("{", 1)[0]
        try:
            result[name] = float(value)
        except ValueError:
            continue
    return result


def metric_delta(before: dict[str, float], after: dict[str, float], names: tuple[str, ...]) -> float:
    for name in names:
        if name in before or name in after:
            return after.get(name, 0.0) - before.get(name, 0.0)
    return 0.0


def tokenize(port: int, text: str) -> int:
    result = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": text}, timeout=180)
    if isinstance(result.get("count"), int):
        return int(result["count"])
    return len(result.get("tokens", []))


def prompt_for_repetitions(workload: Workload, repetitions: int) -> str:
    first = repetitions // 2
    second = repetitions - first
    block = FILLER + (CODE_BLOCK if workload.id == "w7" else "")
    return (
        f"Fixed workload {workload.id}: {workload.description}. "
        f"The first immutable fact is {ANCHORS[0]}.\n"
        + block * first
        + f"The middle immutable fact is {ANCHORS[1]}.\n"
        + block * second
        + f"The final immutable fact is {ANCHORS[2]}.\n"
        + "Return the three immutable facts in their original order, then explain "
        "the result using the fixed deployment constraints."
    )


def calibrate_prompt(port: int, workload: Workload) -> tuple[str, int]:
    low = 0
    high = max(64, workload.target_tokens // 32)
    while tokenize(port, prompt_for_repetitions(workload, high)) < workload.target_tokens:
        high *= 2
        if high > 2_000_000:
            raise RuntimeError(f"could not calibrate {workload.target_tokens} tokens")

    best_prompt = prompt_for_repetitions(workload, 0)
    best_count = tokenize(port, best_prompt)
    best_distance = abs(best_count - workload.target_tokens)
    while low <= high:
        middle = (low + high) // 2
        prompt = prompt_for_repetitions(workload, middle)
        count = tokenize(port, prompt)
        distance = abs(count - workload.target_tokens)
        if distance < best_distance:
            best_prompt, best_count, best_distance = prompt, count, distance
        if count < workload.target_tokens:
            low = middle + 1
        elif count > workload.target_tokens:
            high = middle - 1
        else:
            break
    return best_prompt, best_count


def ordered_facts(text: str) -> bool:
    cursor = -1
    for fact in ANCHORS:
        cursor = text.find(fact, cursor + 1)
        if cursor < 0:
            return False
    return True


def stream_chat(port: int, alias: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "top_k": 1,
        "seed": 1234,
        "max_tokens": max_tokens,
        "cache_prompt": False,
    }
    before = metrics(port)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token: float | None = None
    output: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first_token is None:
                        first_token = time.perf_counter()
                    output.append(piece)
    finished = time.perf_counter()
    after = metrics(port)
    metric_values = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in set(before) | set(after)}
    prompt_seconds = metric_delta(before, after, ("llamacpp:prompt_seconds_total",))
    generation_seconds = metric_delta(before, after, ("llamacpp:tokens_predicted_seconds_total",))
    prompt_tokens = int(usage.get("prompt_tokens") or metric_delta(before, after, ("llamacpp:prompt_tokens_total",)))
    predicted_tokens = int(usage.get("completion_tokens") or metric_delta(before, after, ("llamacpp:tokens_predicted_total",)))
    drafted_tokens = metric_delta(
        before,
        after,
        (
            "llamacpp:spec_decode_num_draft_tokens_total",
            "llamacpp:spec_decode_draft_tokens_total",
        ),
    )
    accepted_tokens = metric_delta(
        before,
        after,
        (
            "llamacpp:spec_decode_num_accepted_tokens_total",
            "llamacpp:spec_decode_accepted_tokens_total",
        ),
    )
    verification_steps = metric_delta(
        before,
        after,
        (
            "llamacpp:spec_decode_num_drafts_total",
            "llamacpp:spec_decode_drafts_total",
        ),
    )
    response_text = "".join(output)
    return {
        "started_utc": utc_now(),
        "wall_seconds": finished - started,
        "ttft_seconds": None if first_token is None else first_token - started,
        "prompt_tokens": prompt_tokens,
        "predicted_tokens": predicted_tokens,
        "prompt_eval_seconds": prompt_seconds,
        "generation_seconds": generation_seconds,
        "pp_tps": prompt_tokens / prompt_seconds if prompt_seconds > 0 else None,
        "tg_tps": predicted_tokens / generation_seconds if generation_seconds > 0 else None,
        "drafted_tokens": drafted_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "mtp_acceptance": accepted_tokens / drafted_tokens if drafted_tokens > 0 else None,
        "mean_accepted_per_verification": accepted_tokens / verification_steps if verification_steps > 0 else None,
        "retrieval_correct": ordered_facts(response_text),
        "output_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "output": response_text,
        "usage": usage,
        "metrics_delta": metric_values,
    }


def gated_stream_chat(
    start_gate: threading.Event,
    executor_id: int,
    port: int,
    alias: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    if not start_gate.wait(timeout=30):
        raise RuntimeError("executor start gate timed out")
    result = stream_chat(port, alias, prompt, max_tokens, timeout)
    result["executor_id"] = executor_id
    return result


def run_executor_group(
    executors: int,
    port: int,
    alias: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    if executors == 1:
        run = stream_chat(port, alias, prompt, max_tokens, timeout)
        run["executor_id"] = 0
        return {"observed_max_busy_slots": 1, "runs": [run]}
    if executors != 2:
        raise RuntimeError(f"unsupported executor count: {executors}")

    start_gate = threading.Event()
    observed_max_busy = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                gated_stream_chat,
                start_gate,
                executor_id,
                port,
                alias,
                prompt,
                max_tokens,
                timeout,
            )
            for executor_id in range(2)
        ]
        start_gate.set()
        while not all(future.done() for future in futures):
            try:
                slots = http_json(f"http://127.0.0.1:{port}/slots", timeout=10)
                observed_max_busy = max(
                    observed_max_busy,
                    sum(bool(slot.get("is_processing")) for slot in slots),
                )
            except Exception:
                pass
            time.sleep(0.25)
        runs = [future.result() for future in futures]
    return {"observed_max_busy_slots": observed_max_busy, "runs": runs}


def numeric(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def windows_memory_snapshot() -> dict[str, float | None]:
    if os.name != "nt":
        return {"ram_used_mb": None}

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"ram_used_mb": None}
    return {
        "ram_total_mb": status.ullTotalPhys / (1024 * 1024),
        "ram_available_mb": status.ullAvailPhys / (1024 * 1024),
        "ram_used_mb": (status.ullTotalPhys - status.ullAvailPhys) / (1024 * 1024),
    }


def windows_process_snapshot(pid: int) -> dict[str, float | int | None]:
    if os.name != "nt":
        return {}

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    process_query_information = 0x0400
    process_vm_read = 0x0010
    handle = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
    if not handle:
        return {}
    try:
        memory = ProcessMemoryCountersEx()
        memory.cb = ctypes.sizeof(memory)
        memory_ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(memory), ctypes.sizeof(memory)
        )
        io = IoCounters()
        io_ok = kernel32.GetProcessIoCounters(handle, ctypes.byref(io))
        creation = FileTime()
        exit_time = FileTime()
        kernel = FileTime()
        user = FileTime()
        times_ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        kernel_ticks = (kernel.dwHighDateTime << 32) | kernel.dwLowDateTime
        user_ticks = (user.dwHighDateTime << 32) | user.dwLowDateTime
        return {
            "working_set_mb": memory.WorkingSetSize / (1024 * 1024) if memory_ok else None,
            "page_faults": int(memory.PageFaultCount) if memory_ok else None,
            "cpu_seconds": (kernel_ticks + user_ticks) / 10_000_000 if times_ok else None,
            "read_bytes": int(io.ReadTransferCount) if io_ok else None,
            "read_count": int(io.ReadOperationCount) if io_ok else None,
        }
    finally:
        kernel32.CloseHandle(handle)


def windows_disk_rate() -> dict[str, float | None]:
    if os.name != "nt":
        return {"read_bytes_per_second": None, "read_count_per_second": None}
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            "$c = Get-Counter '\\PhysicalDisk(_Total)\\Disk Read Bytes/sec',"
            "'\\PhysicalDisk(_Total)\\Disk Reads/sec' -ErrorAction Stop; "
            "$c.CounterSamples | ForEach-Object { [pscustomobject]@{ "
            "Path=$_.Path; Value=$_.CookedValue } } | ConvertTo-Json -Compress"
        ),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except OSError:
        return {"read_bytes_per_second": None, "read_count_per_second": None}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"read_bytes_per_second": None, "read_count_per_second": None}
    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"read_bytes_per_second": None, "read_count_per_second": None}
    samples = decoded if isinstance(decoded, list) else [decoded]
    bytes_rate = None
    count_rate = None
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        path = str(sample.get("Path", ""))
        value = numeric(str(sample.get("Value", "")))
        if value is None:
            continue
        if "Disk Read Bytes/sec" in path:
            bytes_rate = value
        elif "Disk Reads/sec" in path:
            count_rate = value
    return {"read_bytes_per_second": bytes_rate, "read_count_per_second": count_rate}


def query_gpu_telemetry() -> list[dict[str, Any]]:
    queries = (
        "uuid,index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu,pcie.link.gen.current,pcie.link.width.current,pcie.rx_throughput,pcie.tx_throughput",
        "uuid,index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu",
    )
    last_error = ""
    for query in queries:
        code, output = run_no_model_command(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        )
        if code != 0:
            last_error = output.strip()
            continue
        result: list[dict[str, Any]] = []
        for raw in output.splitlines():
            fields = [field.strip() for field in next(csv.reader([raw]))]
            if len(fields) < 6:
                continue
            result.append(
                {
                    "uuid": fields[0],
                    "index": fields[1],
                    "name": fields[2],
                    "memory_total_mb": numeric(fields[3]),
                    "memory_used_mb": numeric(fields[4]),
                    "memory_free_mb": numeric(fields[5]),
                    "gpu_utilization": numeric(fields[6]) if len(fields) > 6 else None,
                    "memory_utilization": numeric(fields[7]) if len(fields) > 7 else None,
                    "power_w": numeric(fields[8]) if len(fields) > 8 else None,
                    "sm_clock_mhz": numeric(fields[9]) if len(fields) > 9 else None,
                    "memory_clock_mhz": numeric(fields[10]) if len(fields) > 10 else None,
                    "temperature_c": numeric(fields[11]) if len(fields) > 11 else None,
                    "pcie_link_gen": numeric(fields[12]) if len(fields) > 12 else None,
                    "pcie_link_width": numeric(fields[13]) if len(fields) > 13 else None,
                    "pcie_rx_kb_s": numeric(fields[14]) if len(fields) > 14 else None,
                    "pcie_tx_kb_s": numeric(fields[15]) if len(fields) > 15 else None,
                }
            )
        return result
    raise RuntimeError(f"nvidia-smi telemetry failed: {last_error}")


class TelemetryPoller:
    def __init__(self, pid: int, interval: float, gpu_uuids: tuple[str, str]):
        self.pid = pid
        self.interval = interval
        self.gpu_uuids = gpu_uuids
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_process_cpu_seconds: float | None = None
        self.last_process_sample_time: float | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="qwen38-telemetry", daemon=True)
        self.thread.start()

    def _system_sample(self) -> dict[str, Any]:
        disk = None
        disk_rate = {"read_bytes_per_second": None, "read_count_per_second": None}
        if psutil is not None:
            virtual = psutil.virtual_memory()
            disk = psutil.disk_io_counters()
            ram_used_mb = (virtual.total - virtual.available) / (1024 * 1024)
            ram_total_mb = virtual.total / (1024 * 1024)
            ram_available_mb = virtual.available / (1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_per_core_percent = psutil.cpu_percent(interval=None, percpu=True)
        else:
            memory = windows_memory_snapshot()
            ram_used_mb = memory.get("ram_used_mb")
            ram_total_mb = memory.get("ram_total_mb")
            ram_available_mb = memory.get("ram_available_mb")
            process = windows_process_snapshot(self.pid)
            process_rss = process.get("working_set_mb")
            page_faults = process.get("page_faults")
            disk_read_bytes = process.get("read_bytes")
            disk_read_count = process.get("read_count")
            now = time.monotonic()
            cpu_seconds = process.get("cpu_seconds")
            cpu_percent = None
            if (
                isinstance(cpu_seconds, (int, float))
                and self.last_process_cpu_seconds is not None
                and self.last_process_sample_time is not None
                and now > self.last_process_sample_time
            ):
                cpu_percent = max(
                    0.0,
                    (float(cpu_seconds) - self.last_process_cpu_seconds)
                    / (now - self.last_process_sample_time)
                    * 100.0,
                )
            if isinstance(cpu_seconds, (int, float)):
                self.last_process_cpu_seconds = float(cpu_seconds)
                self.last_process_sample_time = now
            cpu_per_core_percent = None
        if psutil is not None:
            process_rss = None
            page_faults = None
            disk_read_bytes = None if disk is None else disk.read_bytes
            disk_read_count = None if disk is None else disk.read_count
        if psutil is not None:
            try:
                memory_info = psutil.Process(self.pid).memory_info()
                process_rss = memory_info.rss / (1024 * 1024)
                page_faults = getattr(memory_info, "num_page_faults", None)
            except (psutil.Error, OSError):
                pass
        return {
            "ram_total_mb": ram_total_mb,
            "ram_available_mb": ram_available_mb,
            "ram_used_mb": ram_used_mb,
            "server_working_set_mb": process_rss,
            "page_faults": page_faults,
            "cpu_percent": cpu_percent,
            "cpu_per_core_percent": cpu_per_core_percent,
            "disk_read_bytes": disk_read_bytes,
            "disk_read_count": disk_read_count,
            "disk_read_bytes_per_second": disk_rate["read_bytes_per_second"],
            "disk_read_count_per_second": disk_rate["read_count_per_second"],
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            sample: dict[str, Any] = {"timestamp": utc_now()}
            try:
                sample["gpus"] = query_gpu_telemetry()
            except Exception as exc:
                sample["gpus"] = []
                self.errors.append(str(exc))
            sample["system"] = self._system_sample()
            self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(15.0, self.interval * 4))
        try:
            self.samples.append({"timestamp": utc_now(), "gpus": query_gpu_telemetry(), "system": self._system_sample()})
        except Exception as exc:
            self.errors.append(str(exc))

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"gpus": {}, "errors": list(dict.fromkeys(self.errors))}
        for uuid in self.gpu_uuids:
            rows = [
                gpu
                for sample in self.samples
                for gpu in sample.get("gpus", [])
                if gpu.get("uuid") == uuid
            ]
            if not rows:
                result["gpus"][uuid] = {"peak_used_mb": None, "minimum_free_mb": None}
                continue
            result["gpus"][uuid] = {
                "name": rows[-1].get("name"),
                "peak_used_mb": max(
                    (row["memory_used_mb"] for row in rows if row.get("memory_used_mb") is not None),
                    default=None,
                ),
                "minimum_free_mb": min(
                    (row["memory_free_mb"] for row in rows if row.get("memory_free_mb") is not None),
                    default=None,
                ),
                "peak_gpu_utilization": max(
                    (row["gpu_utilization"] for row in rows if row.get("gpu_utilization") is not None),
                    default=None,
                ),
                "peak_memory_utilization": max(
                    (row["memory_utilization"] for row in rows if row.get("memory_utilization") is not None),
                    default=None,
                ),
                "peak_power_w": max(
                    (row["power_w"] for row in rows if row.get("power_w") is not None),
                    default=None,
                ),
                "peak_pcie_rx_kb_s": max(
                    (row["pcie_rx_kb_s"] for row in rows if row.get("pcie_rx_kb_s") is not None),
                    default=None,
                ),
                "peak_pcie_tx_kb_s": max(
                    (row["pcie_tx_kb_s"] for row in rows if row.get("pcie_tx_kb_s") is not None),
                    default=None,
                ),
            }
        system_samples = [sample.get("system", {}) for sample in self.samples]
        available_values = [sample["ram_available_mb"] for sample in system_samples if sample.get("ram_available_mb") is not None]
        ram_values = [sample["ram_used_mb"] for sample in system_samples if sample.get("ram_used_mb") is not None]
        rss_values = [sample["server_working_set_mb"] for sample in system_samples if sample.get("server_working_set_mb") is not None]
        cpu_values = [sample["cpu_percent"] for sample in system_samples if sample.get("cpu_percent") is not None]
        cpu_core_values = [
            max(sample["cpu_per_core_percent"])
            for sample in system_samples
            if sample.get("cpu_per_core_percent")
        ]
        disk_values = [sample["disk_read_bytes"] for sample in system_samples if sample.get("disk_read_bytes") is not None]
        disk_counts = [sample["disk_read_count"] for sample in system_samples if sample.get("disk_read_count") is not None]
        disk_byte_rates = [sample["disk_read_bytes_per_second"] for sample in system_samples if sample.get("disk_read_bytes_per_second") is not None]
        disk_count_rates = [sample["disk_read_count_per_second"] for sample in system_samples if sample.get("disk_read_count_per_second") is not None]
        result["ram_peak_mb"] = max(ram_values, default=None)
        result["ram_available_min_mb"] = min(available_values, default=None)
        result["server_working_set_peak_mb"] = max(rss_values, default=None)
        result["cpu_peak_percent"] = max(cpu_values, default=None)
        result["cpu_per_core_peak_percent"] = max(cpu_core_values, default=None)
        duration_seconds = max(self.interval, self.interval * max(1, len(self.samples) - 1))
        if len(disk_values) >= 2:
            result["disk_read_mb"] = (max(disk_values) - min(disk_values)) / (1024 * 1024)
            result["disk_read_iops"] = ((max(disk_counts) - min(disk_counts)) / duration_seconds) if len(disk_counts) >= 2 else None
            result["disk_metrics_source"] = (
                "psutil cumulative counters"
                if psutil is not None
                else "Windows process cumulative counters"
            )
        elif disk_byte_rates:
            result["disk_read_mb"] = sum(disk_byte_rates) * self.interval / (1024 * 1024)
            result["disk_read_iops"] = sum(disk_count_rates) * self.interval / duration_seconds if disk_count_rates else None
            result["disk_metrics_source"] = "Windows Get-Counter rates"
        else:
            result["disk_read_mb"] = None
            result["disk_read_iops"] = None
            result["disk_metrics_source"] = None
        result["system_metrics_source"] = (
            "psutil" if psutil is not None else "Windows kernel process counters + GlobalMemoryStatusEx"
        )
        result["sample_count"] = len(self.samples)
        return result


def wait_for_server(port: int, process: subprocess.Popen[Any], log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:] if log_path.exists() else ""
            raise RuntimeError(f"server exited with code {process.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy within {timeout}s: {last_error}")


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def port_is_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def median_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def summarise_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "wall_seconds",
        "ttft_seconds",
        "prompt_eval_seconds",
        "generation_seconds",
        "pp_tps",
        "tg_tps",
        "prompt_tokens",
        "predicted_tokens",
        "drafted_tokens",
        "accepted_draft_tokens",
        "mtp_acceptance",
        "mean_accepted_per_verification",
    )
    summary: dict[str, Any] = {}
    for field in fields:
        values = [run.get(field) for run in runs]
        summary[f"median_{field}"] = median_or_none(values)
        clean = [float(value) for value in values if value is not None]
        summary[f"min_{field}"] = min(clean) if clean else None
        summary[f"max_{field}"] = max(clean) if clean else None
    summary["all_retrieval_correct"] = all(run.get("retrieval_correct", False) for run in runs)
    return summary


def case_telemetry_value(telemetry: dict[str, Any], uuid: str, key: str) -> Any:
    return (telemetry.get("gpus", {}).get(uuid) or {}).get(key)


def run_case(
    case: Case,
    workload_ids: list[str],
    args: argparse.Namespace,
    prompt_catalog: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    command = command_for(case, args)
    raw_dir = output_path.parent / f"{output_path.stem}-raw"
    prompt_dir = output_path.parent / f"{output_path.stem}-prompts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{case.id}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    alias = args.server_alias if args.reuse_server_pid else f"qwen38-flash-next-{case.id}"
    result: dict[str, Any] = {
        "case": asdict(case),
        "command": command,
        "command_text": display_command(command),
        "server_log": str(log_path),
        "started_utc": utc_now(),
        "model_loaded": False,
        "model_load_performed": False,
        "attached_server_pid": args.reuse_server_pid or None,
        "status": "error",
        "workloads": [],
    }
    process: subprocess.Popen[Any] | None = None
    poller: TelemetryPoller | None = None
    try:
        if args.reuse_server_pid:
            if not port_is_in_use(args.port):
                raise RuntimeError(f"no server is listening on port {args.port}")
            if not windows_process_snapshot(args.reuse_server_pid):
                raise RuntimeError(f"server PID {args.reuse_server_pid} is not accessible")
            attached_models = http_json(f"http://127.0.0.1:{args.port}/v1/models", timeout=30)
            attached_slots = http_json(f"http://127.0.0.1:{args.port}/slots", timeout=30)
            model_rows = attached_models.get("data", []) if isinstance(attached_models, dict) else []
            if len(model_rows) != 1 or model_rows[0].get("meta", {}).get("n_ctx") != FULL_CONTEXT:
                raise RuntimeError("attached server must expose one model with n_ctx=262144")
            if not isinstance(attached_slots, list) or len(attached_slots) != SERVER_PARALLEL:
                raise RuntimeError("attached server must expose exactly two inference slots")
            if any(slot.get("n_ctx") != FULL_CONTEXT for slot in attached_slots):
                raise RuntimeError("each attached server slot must use n_ctx=262144")
            log_path.write_text(
                f"Attached to existing server PID {args.reuse_server_pid}; server was not restarted or unloaded.\n",
                encoding="utf-8",
            )
            poller = TelemetryPoller(
                args.reuse_server_pid,
                args.telemetry_interval,
                (args.gpu0_uuid, args.gpu1_uuid),
            )
            poller.start()
        else:
            if port_is_in_use(args.port):
                raise RuntimeError(f"port {args.port} is already in use; refusing to profile an unknown server")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = f"{args.gpu0_uuid},{args.gpu1_uuid}"
            environment["CUDA_MODULE_LOADING"] = "EAGER"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            poller = TelemetryPoller(process.pid, args.telemetry_interval, (args.gpu0_uuid, args.gpu1_uuid))
            poller.start()
            wait_for_server(args.port, process, log_path, args.startup_timeout)
            result["model_load_performed"] = True
        result["model_loaded"] = True
        result["health"] = http_json(f"http://127.0.0.1:{args.port}/health", timeout=30)
        result["models"] = http_json(f"http://127.0.0.1:{args.port}/v1/models", timeout=30)

        for workload_id in workload_ids:
            workload = WORKLOADS[workload_id]
            if workload.target_tokens + workload.output_tokens > case.context_size:
                result["workloads"].append(
                    {
                        "workload": workload_id,
                        "target_tokens": workload.target_tokens,
                        "output_tokens": workload.output_tokens,
                        "status": "skipped_context",
                    }
                )
                continue
            if workload_id not in prompt_catalog:
                prompt, actual_tokens = calibrate_prompt(args.port, workload)
                prompt_path = prompt_dir / f"{workload_id}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                prompt_catalog[workload_id] = {
                    "prompt": prompt,
                    "actual_tokens": actual_tokens,
                    "target_tokens": workload.target_tokens,
                    "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "path": str(prompt_path),
                }
            prompt_info = prompt_catalog[workload_id]
            prompt = str(prompt_info["prompt"])
            if int(prompt_info["actual_tokens"]) + workload.output_tokens > case.context_size:
                result["workloads"].append(
                    {
                        "workload": workload_id,
                        "target_tokens": workload.target_tokens,
                        "output_tokens": workload.output_tokens,
                        "actual_prompt_tokens": prompt_info["actual_tokens"],
                        "status": "skipped_context",
                    }
                )
                continue
            server_pid = args.reuse_server_pid or (process.pid if process is not None else 0)
            for executor_count in EXECUTOR_COUNTS:
                axis_poller = TelemetryPoller(
                    server_pid,
                    args.telemetry_interval,
                    (args.gpu0_uuid, args.gpu1_uuid),
                )
                groups: list[dict[str, Any]] = []
                runs: list[dict[str, Any]] = []
                axis_poller.start()
                try:
                    for repetition in range(1, FIXED_REPETITIONS + 1):
                        group = run_executor_group(
                            executor_count,
                            args.port,
                            alias,
                            prompt,
                            workload.output_tokens,
                            args.request_timeout,
                        )
                        group["repetition"] = repetition
                        for run in group["runs"]:
                            run["repetition"] = repetition
                            runs.append(run)
                        groups.append(group)
                finally:
                    axis_poller.stop()
                axis_telemetry = axis_poller.summary()
                summary = summarise_runs(runs)
                observed_max_busy = max(
                    (group["observed_max_busy_slots"] for group in groups),
                    default=0,
                )
                status = (
                    "ok"
                    if summary["all_retrieval_correct"] and observed_max_busy == executor_count
                    else "correctness_failed"
                )
                result["workloads"].append(
                    {
                        "workload": workload_id,
                        "executors": executor_count,
                        "repetitions": FIXED_REPETITIONS,
                        "target_tokens": workload.target_tokens,
                        "output_tokens": workload.output_tokens,
                        "actual_prompt_tokens": prompt_info["actual_tokens"],
                        "prompt_sha256": prompt_info["sha256"],
                        "prompt_path": prompt_info["path"],
                        "observed_max_busy_slots": observed_max_busy,
                        "groups": groups,
                        "runs": runs,
                        "summary": summary,
                        "telemetry": axis_telemetry,
                        "minimum_free_gpu0_mib": case_telemetry_value(
                            axis_telemetry, args.gpu0_uuid, "minimum_free_mb"
                        ),
                        "minimum_free_gpu1_mib": case_telemetry_value(
                            axis_telemetry, args.gpu1_uuid, "minimum_free_mb"
                        ),
                        "status": status,
                    }
                )
        measured = [item for item in result["workloads"] if "runs" in item]
        result["status"] = (
            "ok"
            if measured and all(item.get("status") == "ok" for item in measured)
            else "correctness_failed" if measured else "no_compatible_workload"
        )
    except Exception as exc:
        result["error"] = repr(exc)
        if log_path.exists():
            result["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    finally:
        if process is not None:
            stop_server(process)
        if poller is not None:
            poller.stop()
            result["telemetry"] = poller.summary()
        result["finished_utc"] = utc_now()

    telemetry = result.get("telemetry", {})
    free0 = case_telemetry_value(telemetry, args.gpu0_uuid, "minimum_free_mb")
    free1 = case_telemetry_value(telemetry, args.gpu1_uuid, "minimum_free_mb")
    result["minimum_free_gpu0_mib"] = free0
    result["minimum_free_gpu1_mib"] = free1
    if result["status"] == "ok":
        if any(item.get("status") == "correctness_failed" for item in result["workloads"]):
            result["status"] = "correctness_failed"
        elif free0 is None or free1 is None:
            result["status"] = "telemetry_missing"
        result["vram_headroom_advisory"] = {
            "target_mib": args.min_free_vram_mib,
            "gpu0_below_target": free0 is not None and free0 < args.min_free_vram_mib,
            "gpu1_below_target": free1 is not None and free1 < args.min_free_vram_mib,
        }
    print(
        f"{case.id}: {result['status']} min_free_mib=({free0},{free1}) log={log_path}",
        flush=True,
    )
    return result


def parity_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["context_size"],
        case["gpu_split"],
        case["ncmoe"],
        case["threads"],
        case["threads_batch"],
        case["batch_size"],
        case["ubatch_size"],
        case["kv_k"],
        case["kv_v"],
    )


def apply_parity(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    mtp: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for case_result in results:
        case = case_result["case"]
        if case["mtp"]:
            mtp.setdefault(parity_key(case), []).append(case_result)
        else:
            target.setdefault(parity_key(case), {})[case_result["case"]["id"]] = case_result
    checks = []
    for key, mtp_cases in mtp.items():
        target_cases = target.get(key, {})
        if not target_cases:
            continue
        baseline = next(iter(target_cases.values()))
        baseline_workloads = {item["workload"]: item for item in baseline.get("workloads", [])}
        for mtp_case in mtp_cases:
            for item in mtp_case.get("workloads", []):
                baseline_item = baseline_workloads.get(item["workload"])
                if not baseline_item or "runs" not in item or "runs" not in baseline_item:
                    continue
                baseline_outputs = [run.get("output") for run in baseline_item["runs"]]
                mtp_outputs = [run.get("output") for run in item["runs"]]
                exact = baseline_outputs == mtp_outputs
                item["parity_exact"] = exact
                checks.append(
                    {
                        "target_case": baseline["case"]["id"],
                        "mtp_case": mtp_case["case"]["id"],
                        "workload": item["workload"],
                        "exact": exact,
                    }
                )
                if not exact and item.get("status") == "ok":
                    item["status"] = "parity_failed"
    return checks


def row_for(case_result: dict[str, Any], item: dict[str, Any], run: dict[str, Any] | None, document: dict[str, Any], record_type: str, repetition: str | int) -> dict[str, Any]:
    case = case_result["case"]
    telemetry = item.get("telemetry") or case_result.get("telemetry", {})
    summary = item.get("summary", {})
    values = run or {
        "prompt_tokens": item.get("actual_prompt_tokens"),
        "predicted_tokens": summary.get("median_predicted_tokens"),
        "pp_tps": summary.get("median_pp_tps"),
        "tg_tps": summary.get("median_tg_tps"),
        "ttft_seconds": summary.get("median_ttft_seconds"),
        "wall_seconds": summary.get("median_wall_seconds"),
        "prompt_eval_seconds": summary.get("median_prompt_eval_seconds"),
        "generation_seconds": summary.get("median_generation_seconds"),
        "drafted_tokens": summary.get("median_drafted_tokens"),
        "accepted_draft_tokens": summary.get("median_accepted_draft_tokens"),
        "mtp_acceptance": summary.get("median_mtp_acceptance"),
        "retrieval_correct": summary.get("all_retrieval_correct"),
    }
    return {
        "timestamp": run.get("started_utc") if run else case_result.get("finished_utc"),
        "llama_sha": document.get("llama_sha"),
        "model_sha256": document.get("model_sha256"),
        "model_quant": document.get("model_quant"),
        "ctx": case["context_size"],
        "prompt_tokens": values.get("prompt_tokens"),
        "output_tokens": values.get("predicted_tokens", item.get("output_tokens")),
        "gpu_split": case["gpu_split"],
        "ncmoe": case["ncmoe"],
        "threads": case["threads"],
        "batch": case["batch_size"],
        "ubatch": case["ubatch_size"],
        "kv_k": case["kv_k"],
        "kv_v": case["kv_v"],
        "mtp": str(case["mtp"]).lower(),
        "mtp_precision": case["mtp_precision"],
        "mtp_device": case["mtp_device"],
        "mtp_nmax": case["mtp_nmax"],
        "mtp_acceptance": values.get("mtp_acceptance"),
        "pp_tps": values.get("pp_tps"),
        "tg_tps": values.get("tg_tps"),
        "ttft_ms": None if values.get("ttft_seconds") is None else values["ttft_seconds"] * 1000,
        "wall_ms": None if values.get("wall_seconds") is None else values["wall_seconds"] * 1000,
        "gpu0_peak_mb": case_telemetry_value(telemetry, document["gpu0_uuid"], "peak_used_mb"),
        "gpu1_peak_mb": case_telemetry_value(telemetry, document["gpu1_uuid"], "peak_used_mb"),
        "ram_peak_mb": telemetry.get("ram_peak_mb"),
        "disk_read_mb": telemetry.get("disk_read_mb"),
        "case_id": case["id"],
        "stage": case["stage"],
        "workload": item.get("workload"),
        "executors": item.get("executors"),
        "repetition": repetition,
        "record_type": record_type,
        "prompt_eval_ms": None if values.get("prompt_eval_seconds") is None else values["prompt_eval_seconds"] * 1000,
        "generation_ms": None if values.get("generation_seconds") is None else values["generation_seconds"] * 1000,
        "retrieval_correct": values.get("retrieval_correct"),
        "parity_exact": item.get("parity_exact"),
        "status": item.get("status", case_result.get("status")),
        "minimum_free_gpu0_mib": item.get(
            "minimum_free_gpu0_mib", case_result.get("minimum_free_gpu0_mib")
        ),
        "minimum_free_gpu1_mib": item.get(
            "minimum_free_gpu1_mib", case_result.get("minimum_free_gpu1_mib")
        ),
        "server_log": case_result.get("server_log"),
    }


def write_outputs(document: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    rows: list[dict[str, Any]] = []
    for case_result in document.get("cases", []):
        for item in case_result.get("workloads", []):
            if item.get("status") == "skipped_context":
                rows.append(row_for(case_result, item, None, document, "skipped", ""))
                continue
            for run in item.get("runs", []):
                rows.append(row_for(case_result, item, run, document, "run", run.get("repetition", "")))
            if item.get("summary"):
                rows.append(row_for(case_result, item, None, document, "median", "median"))
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Profile JSON written: {output_path}")
    print(f"Profile CSV written: {csv_path}")


def write_best_config(document: dict[str, Any], path: Path) -> None:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for case_result in document.get("cases", []):
        case = case_result["case"]
        telemetry = case_result.get("telemetry", {})
        for item in case_result.get("workloads", []):
            summary = item.get("summary", {})
            if item.get("status") != "ok" or not summary.get("all_retrieval_correct"):
                continue
            if item.get("parity_exact") is False:
                continue
            if any(
                case_telemetry_value(telemetry, uuid, "minimum_free_mb") is None
                or case_telemetry_value(telemetry, uuid, "minimum_free_mb") < document["min_free_vram_mib"]
                for uuid in (document["gpu0_uuid"], document["gpu1_uuid"])
            ):
                continue
            median_wall_seconds = summary.get("median_wall_seconds")
            if median_wall_seconds is None:
                continue
            candidate = {
                "case_id": case["id"],
                "workload": item["workload"],
                "ctx": case["context_size"],
                "tensor_split": case["gpu_split"],
                "n_cpu_moe": case["ncmoe"],
                "batch": case["batch_size"],
                "ubatch": case["ubatch_size"],
                "threads": case["threads"],
                "kv": {"k": case["kv_k"], "v": case["kv_v"]},
                "mtp": {
                    "enabled": case["mtp"],
                    "precision": case["mtp_precision"],
                    "device": case["mtp_device"],
                    "n_max": case["mtp_nmax"],
                },
                "median_wall_ms": median_wall_seconds * 1000,
                "median_pp_tps": summary.get("median_pp_tps"),
                "median_tg_tps": summary.get("median_tg_tps"),
            }
            candidates.setdefault(item["workload"], []).append(candidate)
    best_by_workload = {
        workload: min(items, key=lambda item: item["median_wall_ms"])
        for workload, items in candidates.items()
        if items and all(item.get("median_wall_ms") is not None for item in items)
    }
    payload = {
        "model": document.get("model"),
        "quant": document.get("model_quant"),
        "llama_cpp_sha": document.get("llama_sha"),
        "selection_basis": "minimum median wall time among measured, correct, VRAM-safe rows",
        "recommended_default": None,
        "best_by_workload": best_by_workload,
        "note": "A single default remains null until workload priorities are chosen from the measured profiles.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Best-config candidates written: {path}")


def load_prompt_catalog(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for case_result in document.get("cases", []):
        for item in case_result.get("workloads", []):
            prompt_path = item.get("prompt_path")
            if not prompt_path:
                continue
            try:
                path = workspace_path(prompt_path, "saved prompt", True)
                prompt = path.read_text(encoding="utf-8")
            except (OSError, SystemExit):
                continue
            catalog[item["workload"]] = {
                "prompt": prompt,
                "actual_tokens": item.get("actual_prompt_tokens"),
                "target_tokens": item.get("target_tokens"),
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "path": str(path),
            }
    return catalog


def load_resume_document(
    path: Path,
    document: dict[str, Any],
    cases: list[Case],
) -> dict[str, Any]:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot resume from {path}: {exc}") from exc
    if not isinstance(existing, dict):
        raise SystemExit(f"cannot resume from {path}: result document is not an object")
    for key in (
        "model",
        "model_quant",
        "runtime",
        "stage",
        "gpu0_uuid",
        "gpu1_uuid",
        "min_free_vram_mib",
        "model_sha256",
    ):
        if existing.get(key) != document.get(key):
            raise SystemExit(f"--resume refused: {key} differs from the existing result")
    planned = [asdict(case) for case in cases]
    if existing.get("planned_cases") != planned:
        raise SystemExit("--resume refused: selected cases differ from the existing result")

    valid_case_ids = {case.id for case in cases}
    completed_statuses = {"ok", "rejected_vram", "correctness_failed", "no_compatible_workload"}
    kept: list[dict[str, Any]] = []
    for case_result in existing.get("cases", []):
        case = case_result.get("case", {})
        if (
            case.get("id") in valid_case_ids
            and case == asdict(next(item for item in cases if item.id == case.get("id")))
            and case_result.get("status") in completed_statuses
        ):
            kept.append(case_result)
    existing["cases"] = kept
    existing["run_requested"] = True
    existing["model_load_requested"] = True
    existing["model_load_performed"] = any(
        case_result.get("model_loaded", False) for case_result in kept
    )
    existing["planned_cases"] = planned
    return existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run the staged Qwen3.8-Flash-Next deployment profiler; default is no-load plan-only mode."
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--run", action="store_true", help="start a fresh llama-server for each case")
    execution.add_argument("--dry-run", action="store_true", help="print/write the plan without starting a server")
    parser.add_argument("--stage", choices=("context",), default="context")
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--runtime-source", type=Path, default=DEFAULT_RUNTIME_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mtp-model", type=Path, default=DEFAULT_MTP_MODEL)
    parser.add_argument("--model-quant", default=MODEL_QUANT_DEFAULT)
    parser.add_argument("--model-sha256")
    parser.add_argument("--gpu0-uuid", default=DEFAULT_GPU0_UUID)
    parser.add_argument("--gpu1-uuid", default=DEFAULT_GPU1_UUID)
    parser.add_argument("--bind-address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reuse-server-pid", type=int, default=0)
    parser.add_argument("--server-alias", default="qwen3.8-flash-next")
    parser.add_argument("--stage-context", type=int, default=FULL_CONTEXT)
    parser.add_argument("--base-gpu-split", default="38,10")
    parser.add_argument("--base-ncmoe", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--threads-batch", type=int, default=0)
    parser.add_argument("--context-values", default=str(FULL_CONTEXT))
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=MIN_FREE_VRAM_DEFAULT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--best-config", type=Path)
    parser.add_argument("--check-gpu", action="store_true", help="query GPU UUIDs in dry-run mode")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="skip completed cases in an existing result (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="rerun all selected cases")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.stage_context < 1 or args.batch_size < 1 or args.ubatch_size < 1:
        raise SystemExit("context, batch, and ubatch values must be positive")
    if args.stage_context != FULL_CONTEXT:
        raise SystemExit(
            f"--stage-context must remain {FULL_CONTEXT}; vary workload length, not KV allocation"
        )
    context_values = parse_int_list(args.context_values, "--context-values")
    if any(value != FULL_CONTEXT for value in context_values):
        raise SystemExit(
            f"--context-values must contain only {FULL_CONTEXT}; reduced-context runs are invalid"
        )
    if args.ubatch_size > args.batch_size:
        raise SystemExit("--ubatch-size cannot exceed --batch-size")
    if args.threads < 0 or args.threads_batch < 0:
        raise SystemExit("thread values cannot be negative")
    parse_split_list(args.base_gpu_split)
    if args.base_ncmoe < 0 or args.base_ncmoe > 48:
        raise SystemExit("--base-ncmoe must be between 0 and 48")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("timeouts must be positive")
    if args.telemetry_interval <= 0 or args.min_free_vram_mib < 0:
        raise SystemExit("telemetry interval and VRAM floor are invalid")
    if args.max_cases < 0:
        raise SystemExit("--max-cases cannot be negative")
    if args.reuse_server_pid < 0:
        raise SystemExit("--reuse-server-pid cannot be negative")
    if args.model_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.model_sha256):
        raise SystemExit("--model-sha256 must be a 64-character hexadecimal SHA-256")


def plan_document(args: argparse.Namespace, cases: list[Case], workload_ids: list[str], run: bool) -> dict[str, Any]:
    return {
        "generated_utc": utc_now(),
        "run_requested": run,
        "model_load_requested": run and not bool(args.reuse_server_pid),
        "model_load_performed": False,
        "model": "Qwen3.8-Flash-Next",
        "model_quant": args.model_quant,
        "runtime": str(args.runtime),
        "runtime_source": str(args.runtime_source),
        "llama_sha": None,
        "model_sha256": args.model_sha256,
        "gpu0_uuid": args.gpu0_uuid,
        "gpu1_uuid": args.gpu1_uuid,
        "server_context": SERVER_CONTEXT,
        "server_parallel": SERVER_PARALLEL,
        "slot_context": FULL_CONTEXT,
        "test_axes": {
            "executors": list(EXECUTOR_COUNTS),
            "prompts": ["short", "mid"],
            "repetitions": FIXED_REPETITIONS,
        },
        "min_free_vram_mib": args.min_free_vram_mib,
        "reuse_server_pid": args.reuse_server_pid or None,
        "stage": args.stage,
        "workloads": [asdict(WORKLOADS[item]) for item in workload_ids],
        "planned_cases": [asdict(case) for case in cases],
        "case_workload_compatibility": [
            {
                "case_id": case.id,
                "compatible": nominally_compatible_workloads(case, workload_ids),
                "nominally_skipped": [
                    workload_id
                    for workload_id in workload_ids
                    if workload_id not in nominally_compatible_workloads(case, workload_ids)
                ],
            }
            for case in cases
        ],
        "cases": [],
        "parity_checks": [],
        "notes": [
            "The profiler never runs a model-loading command in plan-only mode.",
            "The fixed matrix is executors 1/2 by prompts short/mid, repeated exactly three times.",
            "The current repository DFlash2 runtime is not the default Flash-Next runtime path.",
            "Architecture support still requires an explicit load validation after a Qwen4Exp SHA is pinned.",
            "Q8 MTP is intentionally absent; evaluate it only after an F16/BF16 MTP baseline passes.",
        ],
    }


def main() -> int:
    args = parse_args()
    validate_arguments(args)
    args.runtime = workspace_path(args.runtime, "--runtime")
    args.runtime_source = workspace_path(args.runtime_source, "--runtime-source")
    args.model = workspace_path(args.model, "--model")
    args.mtp_model = workspace_path(args.mtp_model, "--mtp-model")
    output_path = workspace_path(args.output, "--output")
    if output_path.exists() and (not args.run or not args.resume):
        raise SystemExit(
            f"refusing to overwrite existing result: {output_path}; choose a new --output or use --resume"
        )
    if args.best_config is not None:
        args.best_config = workspace_path(args.best_config, "--best-config")
    cases = build_cases(args)
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no profiling cases were selected")
    if args.reuse_server_pid and len(cases) != 1:
        raise SystemExit("--reuse-server-pid requires exactly one fixed deployment case")
    workload_ids = select_workload_ids(args)
    incompatible_cases = [
        case.id for case in cases if not nominally_compatible_workloads(case, workload_ids)
    ]
    if incompatible_cases:
        raise SystemExit(
            "selected workloads do not fit these case contexts: "
            + ", ".join(incompatible_cases)
        )
    document = plan_document(args, cases, workload_ids, args.run)

    if not args.run:
        if args.best_config is not None:
            raise SystemExit("--best-config requires --run and measured results")
        if args.check_gpu:
            rows = validate_gpu_mapping(args.gpu0_uuid, args.gpu1_uuid)
            document["gpu_snapshot"] = rows
        document["planned_commands"] = [display_command(command_for(case, args)) for case in cases]
        write_outputs(document, output_path)
        print(f"DRY RUN: planned {len(cases)} case(s); no server was started and no model was loaded.")
        return 0

    if not args.runtime.is_file():
        raise SystemExit(f"--runtime is missing: {args.runtime}")
    validate_model_shards(args.model)
    if any(case.mtp for case in cases) and not args.mtp_model.is_file():
        raise SystemExit(f"--mtp-model is missing for the selected MTP cases: {args.mtp_model}")
    validate_gpu_mapping(args.gpu0_uuid, args.gpu1_uuid)
    runtime_info = validate_runtime(
        args.runtime,
        args.gpu0_uuid,
        args.gpu1_uuid,
        require_mtp=any(case.mtp for case in cases),
    )
    document["runtime_info"] = runtime_info
    document["llama_sha"] = runtime_info.get("commit") or git_sha(args.runtime_source)
    document["model_shards"] = validate_model_shards(args.model)
    if args.resume and output_path.is_file():
        document = load_resume_document(output_path, document, cases)
        document["runtime_info"] = runtime_info
        document["llama_sha"] = runtime_info.get("commit") or git_sha(args.runtime_source)
        document["model_shards"] = validate_model_shards(args.model)
    prompt_catalog = load_prompt_catalog(document) if args.resume else {}
    completed_case_ids = {
        case_result["case"]["id"]
        for case_result in document.get("cases", [])
        if case_result.get("status") in {"ok", "rejected_vram", "correctness_failed", "no_compatible_workload"}
    }
    write_outputs(document, output_path)
    for case in cases:
        if case.id in completed_case_ids:
            print(f"{case.id}: skipped completed case from {output_path}", flush=True)
            continue
        case_result = run_case(case, workload_ids, args, prompt_catalog, output_path)
        document["cases"].append(case_result)
        if case_result.get("model_load_performed"):
            document["model_load_performed"] = True
        document["parity_checks"] = apply_parity(document["cases"])
        write_outputs(document, output_path)
    if args.best_config is not None:
        write_best_config(document, args.best_config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
