#!/usr/bin/env python3
"""Run the approved Gemma 4 Q8+ short/long combined performance matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "runtime" / "llama.cpp-dflash2" / "build-dflash2" / "bin" / "Release" / "llama-server.exe"
TARGET = ROOT / "models" / "Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf"
DRAFTER = ROOT / "models" / "mtp-gemma-4-31B-it-Q8_0.gguf"
GPU_UUID = "GPU-eed52936-813f-8d68-1654-bfb56cb42bc3"

SHORT_TARGET = 4_000
LONG_TARGET = 48_000
DEFAULT_MAX_TOKENS = 256

BASE_CONFIG = {
    "batch_size": 256,
    "ubatch_size": 128,
    "parallel": 1,
    "flash_attn": "on",
    "draft_n_max": 3,
    "draft_device": "CUDA0",
    "draft_ngl": "all",
    "draft_cache_k": "q8_0",
    "draft_cache_v": "q8_0",
}

CONFIGS = [
    {
        "id": "baseline-control",
        "candidate": False,
        "context_size": 56_320,
        "target_cache_k": "q8_0",
        "target_cache_v": "q8_0",
    },
    {
        "id": "n1-target-q8-f16",
        "candidate": True,
        "context_size": 56_320,
        "target_cache_k": "q8_0",
        "target_cache_v": "f16",
    },
    {
        "id": "n2-target-f16-q8",
        "candidate": True,
        "context_size": 56_320,
        "target_cache_k": "f16",
        "target_cache_v": "q8_0",
    },
    {
        "id": "n3-q8-context-73728",
        "candidate": True,
        "context_size": 73_728,
        "target_cache_k": "q8_0",
        "target_cache_v": "q8_0",
    },
    {
        "id": "n4-q8-context-88064",
        "candidate": True,
        "context_size": 88_064,
        "target_cache_k": "q8_0",
        "target_cache_v": "q8_0",
    },
]

FILLER = (
    "The archive note records a stable calibration sentence for the Gemma four "
    "context benchmark. It is intentionally ordinary, repeats no hidden answer, "
    "and exists only to make the prompt length measurable. "
)


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def tokenize(port: int, text: str) -> int:
    result = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": text})
    tokens = result.get("tokens", [])
    return int(result.get("count", len(tokens)))


def make_prompt(repetitions: int) -> str:
    first = (
        "Benchmark instructions: retain the exact anchor facts while processing this "
        "long context. Anchor A is cobalt-otter-417. "
    )
    middle = (
        "The middle anchor is maple-quartz-628. Do not confuse it with the other "
        "labels. Continue reading all surrounding calibration text. "
    )
    last = (
        "Final retrieval anchor: saffron-lantern-903. In the response, state the three "
        "anchor labels in order, followed by the word COMPLETE. "
    )
    half = max(1, repetitions // 2)
    return first + (FILLER * half) + middle + (FILLER * (repetitions - half)) + last


def calibrate_prompt(port: int, target_tokens: int) -> tuple[str, int]:
    low = 0
    high = max(32, target_tokens // 8)
    while tokenize(port, make_prompt(high)) < target_tokens:
        high *= 2
        if high > 1_000_000:
            raise RuntimeError(f"could not calibrate {target_tokens} tokens")

    best_text = make_prompt(0)
    best_count = tokenize(port, best_text)
    best_distance = abs(best_count - target_tokens)
    while low <= high:
        mid = (low + high) // 2
        text = make_prompt(mid)
        count = tokenize(port, text)
        distance = abs(count - target_tokens)
        if distance < best_distance:
            best_text, best_count, best_distance = text, count, distance
        if count < target_tokens:
            low = mid + 1
        elif count > target_tokens:
            high = mid - 1
        else:
            break
    return best_text, best_count


def metrics(port: int) -> dict[str, float]:
    try:
        raw = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode("utf-8")
    except Exception:
        return {}
    result: dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        try:
            result[name] = float(value)
        except ValueError:
            pass
    return result


def gpu_snapshot() -> dict[str, int] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except Exception:
        return None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[0] != GPU_UUID:
            continue
        try:
            used = int(fields[1])
            total = int(fields[2])
        except ValueError:
            return None
        return {"used_mib": used, "total_mib": total, "free_mib": total - used}
    return None


class VRAMPoller:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            sample = gpu_snapshot()
            if sample is not None:
                self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        sample = gpu_snapshot()
        if sample is not None:
            self.samples.append(sample)

    def summary(self) -> dict[str, int | None]:
        if not self.samples:
            return {"peak_used_mib": None, "minimum_free_mib": None, "total_mib": None}
        return {
            "peak_used_mib": max(sample["used_mib"] for sample in self.samples),
            "minimum_free_mib": min(sample["free_mib"] for sample in self.samples),
            "total_mib": self.samples[-1]["total_mib"],
        }


def wait_for_server(port: int, process: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.monotonic() + 180
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            except Exception:
                pass
            raise RuntimeError(f"server exited with code {process.returncode}: {tail}")
        try:
            response = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if response.status == 200:
                return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def stream_chat(
    port: int,
    alias: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    before = metrics(port)
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "top_k": 1,
        "seed": 1234,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_time: float | None = None
    usage: dict[str, Any] = {}
    chunks = 0
    with urllib.request.urlopen(request, timeout=1800) as response:
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
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    chunks += 1
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
    end = time.perf_counter()
    after = metrics(port)

    def delta(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    prompt_tokens = int(usage.get("prompt_tokens") or delta("llamacpp:prompt_tokens_total") or 0)
    predicted_tokens = int(usage.get("completion_tokens") or delta("llamacpp:tokens_predicted_total") or 0)
    prompt_seconds = delta("llamacpp:prompt_seconds_total")
    predicted_seconds = delta("llamacpp:tokens_predicted_seconds_total")
    return {
        "wall_seconds": end - start,
        "ttft_seconds": (first_token_time - start) if first_token_time is not None else None,
        "prompt_tokens": prompt_tokens,
        "predicted_tokens": predicted_tokens,
        "prompt_tps": (prompt_tokens / prompt_seconds) if prompt_seconds > 0 else None,
        "generation_tps": (predicted_tokens / predicted_seconds) if predicted_seconds > 0 else None,
        "chunks": chunks,
        "usage": usage,
    }


def command_for(config: dict[str, Any], port: int, alias: str) -> list[str]:
    return [
        str(SERVER),
        "--model", str(TARGET),
        "--spec-draft-model", str(DRAFTER),
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(BASE_CONFIG["draft_n_max"]),
        "--spec-draft-device", BASE_CONFIG["draft_device"],
        "--spec-draft-ngl", BASE_CONFIG["draft_ngl"],
        "--cache-type-k", config["target_cache_k"],
        "--cache-type-v", config["target_cache_v"],
        "--spec-draft-type-k", BASE_CONFIG["draft_cache_k"],
        "--spec-draft-type-v", BASE_CONFIG["draft_cache_v"],
        "--alias", alias,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--device", "CUDA0",
        "--split-mode", "none",
        "--gpu-layers", "all",
        "--ctx-size", str(config["context_size"]),
        "--parallel", str(BASE_CONFIG["parallel"]),
        "--kv-unified",
        "--flash-attn", BASE_CONFIG["flash_attn"],
        "--batch-size", str(BASE_CONFIG["batch_size"]),
        "--ubatch-size", str(BASE_CONFIG["ubatch_size"]),
        "--fit", "off",
        "--no-mmproj",
        "--no-context-shift",
        "--jinja",
        "--reasoning", "auto",
        "--reasoning-preserve",
        "--metrics",
    ]


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def run_case(
    config: dict[str, Any],
    port: int,
    output_path: Path,
    repetitions: int,
    max_tokens: int,
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    alias = f"gemma4-{config['id']}"
    log_path = output_path.with_name(f"combined-{config['id']}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    command = command_for(config, port, alias)
    poller = VRAMPoller()
    process: subprocess.Popen[Any] | None = None
    result: dict[str, Any] = {
        **config,
        "started_utc": started,
        "command": command,
        "log": str(log_path),
        "status": "error",
        "workloads": {},
    }
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        poller.start()
        wait_for_server(port, process, log_path)
        short_prompt, short_count = calibrate_prompt(port, SHORT_TARGET)
        long_prompt, long_count = calibrate_prompt(port, LONG_TARGET)
        result["prompt_targets"] = {
            "short": {"target_tokens": SHORT_TARGET, "actual_tokens": short_count},
            "long": {"target_tokens": LONG_TARGET, "actual_tokens": long_count},
        }

        for workload, prompt, actual_count in (
            ("short", short_prompt, short_count),
            ("long", long_prompt, long_count),
        ):
            stream_chat(port, alias, prompt + "\nWarmup.", min(32, max_tokens))
            runs = []
            for repetition in range(1, repetitions + 1):
                run = stream_chat(
                    port,
                    alias,
                    prompt + f"\nCombined benchmark repetition {repetition}.",
                    max_tokens,
                )
                run["repetition"] = repetition
                runs.append(run)
            result["workloads"][workload] = {
                "actual_prompt_tokens": actual_count,
                "runs": runs,
                "median_wall_seconds": median_or_none([run["wall_seconds"] for run in runs]),
                "median_ttft_seconds": median_or_none(
                    [run["ttft_seconds"] for run in runs if run["ttft_seconds"] is not None]
                ),
                "median_prompt_tps": median_or_none(
                    [run["prompt_tps"] for run in runs if run["prompt_tps"] is not None]
                ),
                "median_generation_tps": median_or_none(
                    [run["generation_tps"] for run in runs if run["generation_tps"] is not None]
                ),
            }

        short_summary = result["workloads"]["short"]
        long_summary = result["workloads"]["long"]
        result["combined_median_wall_seconds"] = (
            short_summary["median_wall_seconds"] + long_summary["median_wall_seconds"]
        )
        result["combined_median_generation_tps"] = median_or_none(
            [
                run["generation_tps"]
                for workload in result["workloads"].values()
                for run in workload["runs"]
                if run["generation_tps"] is not None
            ]
        )
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = repr(exc)
        try:
            result["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except Exception:
            pass
    finally:
        if process is not None:
            stop_server(process)
        poller.stop()
        result["vram"] = poller.summary()
        result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    print(
        f"{config['id']}: {result['status']} "
        f"combined_wall={result.get('combined_median_wall_seconds')} "
        f"min_free_mib={result['vram'].get('minimum_free_mib')}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", default=",".join(config["id"] for config in CONFIGS))
    parser.add_argument("--custom-context-sizes", default="")
    parser.add_argument("--custom-id-prefix", default="custom")
    parser.add_argument("--custom-target-cache-k", choices=("q8_0", "f16"), default="q8_0")
    parser.add_argument("--custom-target-cache-v", choices=("q8_0", "f16"), default="q8_0")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()

    if not SERVER.exists():
        raise FileNotFoundError(SERVER)
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    if not DRAFTER.exists():
        raise FileNotFoundError(DRAFTER)
    if args.custom_context_sizes:
        sizes = [int(item.strip()) for item in args.custom_context_sizes.split(",") if item.strip()]
        if not sizes:
            raise ValueError("--custom-context-sizes was empty")
        selected = [
            {
                "id": f"{args.custom_id_prefix}-{context_size}",
                "candidate": True,
                "context_size": context_size,
                "target_cache_k": args.custom_target_cache_k,
                "target_cache_v": args.custom_target_cache_v,
            }
            for context_size in sizes
        ]
    else:
        requested = {item.strip() for item in args.case.split(",") if item.strip()}
        selected = [config for config in CONFIGS if config["id"] in requested]
        unknown = requested - {config["id"] for config in CONFIGS}
        if unknown:
            raise ValueError(f"unknown case(s): {sorted(unknown)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "server": str(SERVER),
        "target": str(TARGET),
        "drafter": str(DRAFTER),
        "gpu_uuid": GPU_UUID,
        "short_target_tokens": SHORT_TARGET,
        "long_target_tokens": LONG_TARGET,
        "repetitions": args.repetitions,
        "max_tokens": args.max_tokens,
        "cases": [],
    }
    for index, config in enumerate(selected):
        document["cases"].append(
            run_case(config, args.port + index, output_path, args.repetitions, args.max_tokens)
        )
        output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
