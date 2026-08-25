"""Compare Gemma 4 target-only decoding with the Google MTP assistant.

The profiler is intentionally dry-run by default.  Pass --run explicitly before
it starts llama-server and loads the RTX 5090 model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
RUNTIME_DEFAULT = WORKSPACE / "runtime" / "llama.cpp-dflash2" / "build-dflash2" / "bin" / "Release" / "llama-server.exe"
TARGET_DEFAULT = WORKSPACE / "models" / "Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf"
DRAFTER_DEFAULT = WORKSPACE / "models" / "mtp-gemma-4-31B-it-Q8_0.gguf"
OUTPUT_DEFAULT = WORKSPACE / "benchmarks" / "gemma4" / "mtp-profile.json"
GPU_UUID_DEFAULT = "GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0"
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
        description="Profile Gemma 4 target-only versus Google gemma-4-31B-it MTP decoding."
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--run",
        action="store_true",
        help="start llama-server and run the benchmark; omitted by default for safety",
    )
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="print validated commands without starting a server (the default)",
    )
    parser.add_argument("--runtime", type=Path, default=RUNTIME_DEFAULT)
    parser.add_argument("--target", type=Path, default=TARGET_DEFAULT)
    parser.add_argument("--drafter", type=Path, default=DRAFTER_DEFAULT)
    parser.add_argument("--gpu-uuid", default=GPU_UUID_DEFAULT)
    parser.add_argument("--bind-address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--context-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--draft-n-max", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> dict[str, Path]:
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    for name in ("context_size", "batch_size", "ubatch_size", "draft_n_max", "repetitions", "max_tokens"):
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
    if args.prompt_file is not None and not args.prompt_file.is_file():
        raise SystemExit(f"Prompt file is missing: {args.prompt_file}")
    return paths


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


def build_command(
    runtime: Path,
    target: Path,
    drafter: Path,
    args: argparse.Namespace,
    mode: str,
) -> list[str]:
    alias = f"gemma4-31b-isometry-fabled-persona-5090-{mode}"
    command = [str(runtime), "--model", str(target)]
    if mode == "mtp":
        command.extend(
            [
                "--spec-draft-model",
                str(drafter),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(args.draft_n_max),
                "--spec-draft-device",
                "CUDA0",
                "--spec-draft-ngl",
                "all",
                "--spec-draft-type-k",
                "f16",
                "--spec-draft-type-v",
                "f16",
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
            "q8_0",
            "--cache-type-v",
            "q8_0",
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


def log_tail(path: Path, limit: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "<log unavailable>"
    return text[-limit:]


def wait_for_server(process: subprocess.Popen[bytes], port: int, timeout: float, log_path: Path) -> dict[str, Any]:
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
    raise TimeoutError(f"Timed out waiting for llama-server on port {port}; last response {last_status}; log {log_path}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def run_mode(
    mode: str,
    command: list[str],
    args: argparse.Namespace,
    prompts: list[str],
    output_path: Path,
) -> dict[str, Any]:
    log_path = output_path.with_name(f"{output_path.stem}-{mode}.log")
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

        for warmup_index in range(args.warmup):
            warmup_prompt = prompts[0] + f"\n\nWarm-up request {warmup_index + 1}; answer in one sentence."
            stream_chat(args.port, f"gemma4-31b-isometry-fabled-persona-5090-{mode}", warmup_prompt, min(args.max_tokens, 64), args.request_timeout)

        records: list[dict[str, Any]] = []
        model_name = f"gemma4-31b-isometry-fabled-persona-5090-{mode}"
        for repetition, prompt in enumerate(prompts, start=1):
            record = stream_chat(args.port, model_name, prompt, args.max_tokens, args.request_timeout)
            record["mode"] = mode
            record["repetition"] = repetition
            records.append(record)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "repetition": repetition,
                        "ttft_seconds": record["ttft_seconds"],
                        "tg_tokens_per_second": record["tg_tokens_per_second"],
                        "acceptance_ratio": record["acceptance_ratio"],
                        "mean_accepted_per_verification": record["mean_accepted_per_verification"],
                    },
                    sort_keys=True,
                )
            )
        return {"health": health, "models": models, "records": records, "log": str(log_path)}
    finally:
        stop_server(process)


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
        "median_mean_accepted_per_verification": median_field(records, "mean_accepted_per_verification"),
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def validate_gpu(uuid: str) -> None:
    nvidia = shutil.which("nvidia-smi")
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
    matches = [line for line in completed.stdout.splitlines() if "RTX 5090" in line and uuid in line]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one RTX 5090 with UUID {uuid}; matches={matches}")


def main() -> int:
    args = parse_args()
    paths = validate_args(args)
    prompt = DEFAULT_PROMPT if args.prompt_file is None else args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise SystemExit("The benchmark prompt cannot be empty")
    prompts = [prompt + f"\n\nBenchmark repetition {index}." for index in range(1, args.repetitions + 1)]

    commands = {
        mode: build_command(paths["runtime"], paths["target"], paths["drafter"], args, mode)
        for mode in ("target-only", "mtp")
    }
    version = runtime_version(paths["runtime"])

    if not args.run:
        print("Dry run: no llama-server process was started and no model was loaded.")
        print(f"Runtime: {version}")
        print(f"Target: {paths['target']} ({paths['target'].stat().st_size} bytes)")
        print(f"Google MTP drafter: {paths['drafter']} ({paths['drafter'].stat().st_size} bytes)")
        for mode, command in commands.items():
            print(f"{mode}: {display_command(command)}")
        print("Pass --run when the RTX 5090 is available to execute the comparison.")
        return 0

    validate_gpu(args.gpu_uuid)
    output_path = paths["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for mode in ("target-only", "mtp"):
        try:
            results[mode] = run_mode(mode, commands[mode], args, prompts, output_path)
        except Exception as exc:  # preserve baseline results and the failing mode's log
            errors[mode] = f"{type(exc).__name__}: {exc}"
            print(f"{mode} failed: {errors[mode]}", file=sys.stderr)

    target_summary = summarize(results.get("target-only", {}).get("records", []))
    mtp_summary = summarize(results.get("mtp", {}).get("records", []))
    profile = {
        "generated_at": utc_now(),
        "executed": True,
        "workspace": str(WORKSPACE),
        "gpu_uuid": args.gpu_uuid,
        "runtime": {"path": str(paths["runtime"]), "version": version},
        "assets": {
            "target": {"path": str(paths["target"]), "bytes": paths["target"].stat().st_size},
            "google_mtp_drafter": {"path": str(paths["drafter"]), "bytes": paths["drafter"].stat().st_size},
        },
        "configuration": {
            "port": args.port,
            "context_size": args.context_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "draft_n_max": args.draft_n_max,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_k": 1,
            "seed": 1234,
        },
        "commands": {mode: display_command(command) for mode, command in commands.items()},
        "results": results,
        "errors": errors,
        "summary": {"target-only": target_summary, "mtp": mtp_summary},
        "comparison": {
            "tg_speedup_mtp_over_target_only": ratio(
                mtp_summary["median_tg_tokens_per_second"],
                target_summary["median_tg_tokens_per_second"],
            ),
            "wall_time_ratio_mtp_over_target_only": ratio(
                mtp_summary["median_wall_seconds"],
                target_summary["median_wall_seconds"],
            ),
        },
    }
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Profile written: {output_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
