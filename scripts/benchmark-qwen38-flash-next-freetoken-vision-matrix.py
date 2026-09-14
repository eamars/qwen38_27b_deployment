#!/usr/bin/env python3
"""Run the four Qwen3.8-Flash-Next FreeToken text/vision startup cases.

Every case uses the same shared WSL venv, a 262,144-token context, and the
retained approximately 4K retrieval prompt.  Vision cases build the vision
tower but deliberately send text-only requests; this measures the cost of
enabling vision without introducing image preprocessing into the benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "benchmarks" / "qwen38_flash_next" / "2026-09-02" / "freetoken-4k-prompt.txt"
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "raw" / "qwen38_flash_next" / "2026-09-14"
    / "freetoken-vision-262k-4k.json"
)
RUNTIME = ROOT / "runtime" / "freetoken-a80b4d3"
CONTEXT_TOKENS = 262144
PROMPT_TOKEN_MIN = 4000
PROMPT_TOKEN_MAX = 4096

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "freetoken_qwen38_benchmark_helpers",
    Path(__file__).with_name("benchmark-freetoken-qwen38-next.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError("could not load the existing FreeToken benchmark helpers")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER)


CONFIGS = (
    {"label": "official-no-visual", "variant": "Official", "vision": False},
    {"label": "uncensored-no-visual", "variant": "Uncensored", "vision": False},
    {"label": "official-visual", "variant": "Official", "vision": True},
    {"label": "uncensored-visual", "variant": "Uncensored", "vision": True},
)


def served_model_name(config: dict[str, Any]) -> str:
    if config["variant"] == "Official":
        return "qwen38-next-freetoken-vision" if config["vision"] else "qwen38-next-freetoken"
    return (
        "qwen38-next-uncensored-freetoken-vision"
        if config["vision"]
        else "qwen38-next-uncensored-freetoken"
    )


def launcher_path(config: dict[str, Any]) -> Path:
    if config["variant"] == "Official":
        filename = (
            "start-qwen38-flash-next-freetoken-vision.ps1"
            if config["vision"]
            else "start-qwen38-flash-next-freetoken.ps1"
        )
    else:
        filename = (
            "start-qwen38-flash-next-uncensored-freetoken-vision.ps1"
            if config["vision"]
            else "start-qwen38-flash-next-uncensored-freetoken.ps1"
        )
    return ROOT / "scripts" / filename


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def runtime_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(RUNTIME), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(RUNTIME), "status", "--short"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return {"source_commit": commit, "source_dirty": status}


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"expected a Windows drive path: {resolved}")
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


class TestServer:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]):
        self.args = args
        self.config = config
        self.port = args.port
        pid_slug = "freetoken" if config["variant"] == "Official" else "uncensored-freetoken"
        self.pid_file = f"/tmp/qwen38-flash-next-{pid_slug}-{self.port}.pid"
        self.telemetry_file = f"/tmp/qwen38-vision-matrix-{config['label']}-{self.port}.jsonl"
        self.log_path = args.output.resolve().parent / "logs" / f"{config['label']}.log"
        self.process: subprocess.Popen[Any] | None = None
        self.sampler: subprocess.Popen[Any] | None = None
        self.log_handle: Any = None

    def launcher_command(self, stop: bool = False) -> list[str]:
        command = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(launcher_path(self.config)),
            "-Profile", "Native256K",
            "-Port", str(self.port),
        ]
        if stop:
            command.append("-Stop")
        return command

    def start(self) -> tuple[float, dict[str, Any]]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        subprocess.run(
            ["wsl.exe", "rm", "-f", "--", self.telemetry_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        started = time.perf_counter()
        self.process = subprocess.Popen(
            self.launcher_command(), stdout=self.log_handle,
            stderr=subprocess.STDOUT, text=True,
        )
        sampler_script = windows_to_wsl(Path(__file__))
        self.sampler = subprocess.Popen([
            "wsl.exe", "python3", sampler_script, "--wsl-sampler",
            "--pid-file", self.pid_file, "--sampler-output", self.telemetry_file,
            "--telemetry-interval", str(self.args.telemetry_interval),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.monotonic() + self.args.server_timeout
        last_error = "not ready"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_handle.flush()
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                raise RuntimeError(f"FreeToken exited during load ({self.process.returncode}):\n{tail}")
            try:
                health = _HELPER.http_json(f"http://127.0.0.1:{self.port}/health")
            except Exception as exc:
                last_error = str(exc)
            else:
                if health.get("status") == "error":
                    raise RuntimeError(f"FreeToken load failed: {health.get('message')}")
                if health.get("status") == "ok":
                    stats = _HELPER.http_json(f"http://127.0.0.1:{self.port}/v1/stats")
                    self.assert_contract(stats)
                    return time.perf_counter() - started, {"health": health, "stats": stats}
            time.sleep(2)
        raise TimeoutError(f"FreeToken was not ready after {self.args.server_timeout}s: {last_error}")

    def assert_contract(self, stats: dict[str, Any]) -> None:
        model = stats.get("model") or {}
        resolved_context = int(model.get("ctx") or 0)
        if resolved_context != CONTEXT_TOKENS:
            raise RuntimeError(
                f"{self.config['label']}: expected resolved context {CONTEXT_TOKENS}, "
                f"got {resolved_context}"
            )
        modalities = model.get("input_modalities")
        expected = sorted(["text", "image"] if self.config["vision"] else ["text"])
        if modalities is not None and sorted(modalities) != expected:
            raise RuntimeError(
                f"{self.config['label']}: expected input_modalities={expected}, got {modalities}"
            )

    def stop(self) -> dict[str, Any]:
        try:
            subprocess.run(
                self.launcher_command(stop=True),
                capture_output=True, text=True, timeout=120, check=False,
            )
        finally:
            if self.process is not None:
                try:
                    self.process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
            if self.sampler is not None:
                try:
                    self.sampler.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.sampler.terminate()
            if self.log_handle is not None:
                self.log_handle.close()
        cpu_rows = _HELPER.read_json_lines(self.telemetry_file)
        return {
            "cpu": _HELPER.summarize_cpu(cpu_rows),
            "log": str(self.log_path),
            "log_tail": self.log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            if self.log_path.exists() else "",
        }


def run_request(
    args: argparse.Namespace,
    server: TestServer,
    prompt: str,
    max_tokens: int,
    label: str,
) -> dict[str, Any]:
    gpu = _HELPER.GpuPoller(args.gpu_uuid, args.telemetry_interval)
    gpu.start()
    try:
        row = _HELPER.stream_chat(
            server.port, served_model_name(server.config),
            prompt, max_tokens, args.request_timeout,
        )
    finally:
        row_gpu = gpu.stop()
    prompt_tokens = int(row.get("prompt_tokens") or 0)
    if not PROMPT_TOKEN_MIN <= prompt_tokens <= PROMPT_TOKEN_MAX:
        raise RuntimeError(
            f"{server.config['label']}: prompt token count {prompt_tokens} is outside "
            f"the required 4K range [{PROMPT_TOKEN_MIN}, {PROMPT_TOKEN_MAX}]"
        )
    row["label"] = label
    row["gpu"] = row_gpu
    start = dt.datetime.fromisoformat(row["started_utc"])
    finish = dt.datetime.fromisoformat(row["finished_utc"])
    cpu_rows = [
        sample for sample in _HELPER.read_json_lines(server.telemetry_file)
        if start <= dt.datetime.fromisoformat(sample["timestamp"]) <= finish
    ]
    row["cpu"] = _HELPER.summarize_cpu(cpu_rows)
    return row


def run_case(args: argparse.Namespace, config: dict[str, Any], prompt: str) -> dict[str, Any]:
    server = TestServer(args, config)
    result: dict[str, Any] = {
        "label": config["label"],
        "model_variant": config["variant"],
        "vision_enabled": config["vision"],
        "image_input_sent": False,
        "context_tokens_requested": CONTEXT_TOKENS,
        "status": "not_run",
    }
    try:
        load_seconds, ready = server.start()
        result["load_seconds"] = load_seconds
        result["ready"] = ready
        result["warmup"] = run_request(
            args, server, prompt, min(args.warmup_tokens, args.max_tokens),
            f"{config['label']}-warmup",
        )
        result["runs"] = [
            run_request(args, server, prompt, args.max_tokens, f"{config['label']}-{i + 1}")
            for i in range(args.repetitions)
        ]
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["server_telemetry"] = server.stop()
    runs = result.get("runs") or []
    if runs:
        result["summary"] = {
            "runs": len(runs),
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in runs),
            "median_ttft_seconds": statistics.median(row["ttft_seconds"] for row in runs),
            "median_estimated_pp_tps": statistics.median(row["estimated_pp_tps"] for row in runs),
            "median_estimated_tg_tps": statistics.median(row["estimated_tg_tps"] for row in runs),
            "all_anchor_checks_passed": all(row["retrieval_anchors_in_order"] for row in runs),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--gpu-uuid", default="GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0")
    parser.add_argument("--server-timeout", type=float, default=3600)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--telemetry-interval", type=float, default=0.5)
    parser.add_argument("--wsl-sampler", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pid-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sampler-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.wsl_sampler:
        if args.pid_file is None or args.sampler_output is None:
            raise SystemExit("sampler paths are required")
        return _HELPER.wsl_sampler(args.pid_file, args.sampler_output, args.telemetry_interval)

    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    prompt = args.prompt.resolve().read_text(encoding="utf-8")
    document: dict[str, Any] = {
        "schema": 1,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "objective": "compare Qwen3.8-Flash-Next official/uncensored FreeToken with vision disabled/enabled",
        "runtime": runtime_state(),
        "protocol": {
            "context_tokens": CONTEXT_TOKENS,
            "prompt": str(args.prompt.resolve()),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_token_target": "4K",
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_k": 1,
            "ignore_eos": True,
            "cache_type": "radix",
            "concurrency": 1,
            "image_input_sent": False,
            "repetitions": args.repetitions,
        },
        "configs": [],
    }
    _HELPER.atomic_json(args.output, document)

    for config in CONFIGS:
        print(f"=== {config['label']} ===", flush=True)
        result = run_case(args, config, prompt)
        document["configs"].append(result)
        document["updated_utc"] = utc_now()
        _HELPER.atomic_json(args.output, document)
        print(json.dumps({
            "label": config["label"],
            "status": result["status"],
            "summary": result.get("summary"),
            "error": result.get("error"),
        }, indent=2), flush=True)

    passed = sum(result.get("status") == "passed" for result in document["configs"])
    document["summary"] = {
        "passed": passed,
        "failed": len(document["configs"]) - passed,
        "all_passed": passed == len(CONFIGS),
    }
    document["updated_utc"] = utc_now()
    _HELPER.atomic_json(args.output, document)
    print(json.dumps(document["summary"], indent=2))
    return 0 if document["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
