#!/usr/bin/env python3
"""Benchmark the retained FreeToken GPU-only profile for Qwen3.8-Flash-Next.

The controller runs on Windows and launches the pinned FreeToken environment in
WSL.  ``--wsl-sampler`` is an internal Linux-only mode used to collect CPU/RAM
telemetry for the complete FreeToken process tree without spawning a WSL
process for every sample.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT = (
    ROOT / "benchmarks" / "qwen38_flash_next" / "2026-09-02"
    / "freetoken-4k-prompt.txt"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "qwen38_flash_next" / "2026-09-02"
    / "freetoken-4k-gpu-only-winner.json"
)
DEFAULT_MODEL = "/home/rba90/models/Qwen3.8-Flash-Next-NVFP4"
DEFAULT_VENV = "/home/rba90/.freetoken-qwen38/venv"
DEFAULT_GPU_UUID = "GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0"
ANCHORS = (
    "FLASH-NEXT-BEGIN-ORCHID-5090",
    "FLASH-NEXT-MIDDLE-QUARTZ-4090",
    "FLASH-NEXT-END-LANTERN-262144",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class GpuPoller:
    def __init__(self, uuid: str, interval: float = 0.5):
        self.uuid = uuid
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        queries = (
            "uuid,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu,pcie.rx_throughput,pcie.tx_throughput",
            "uuid,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu",
        )
        completed = None
        for fields in queries:
            attempt = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if attempt.returncode == 0:
                completed = attempt
                break
        if completed is None:
            raise RuntimeError("nvidia-smi telemetry queries failed")
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if parts and parts[0] == self.uuid:
                self.samples.append({
                    "timestamp": utc_now(), "uuid": parts[0], "name": parts[1],
                    "gpu_utilization": numeric(parts[2]),
                    "memory_utilization": numeric(parts[3]),
                    "memory_used_mb": numeric(parts[4]),
                    "memory_free_mb": numeric(parts[5]),
                    "power_w": numeric(parts[6]), "temperature_c": numeric(parts[7]),
                    "pcie_rx_kb_s": numeric(parts[8]) if len(parts) > 8 else None,
                    "pcie_tx_kb_s": numeric(parts[9]) if len(parts) > 9 else None,
                })

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._sample()
            except Exception as exc:  # telemetry must not abort inference
                self.errors.append(str(exc))
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=15)
        try:
            self._sample()
        except Exception as exc:
            self.errors.append(str(exc))
        rows = self.samples
        values = lambda key: [float(row[key]) for row in rows if row.get(key) is not None]
        return {
            "sample_count": len(rows), "errors": list(dict.fromkeys(self.errors)),
            "average_gpu_utilization": statistics.fmean(values("gpu_utilization")) if values("gpu_utilization") else None,
            "p95_gpu_utilization": percentile(values("gpu_utilization"), 0.95),
            "peak_gpu_utilization": max(values("gpu_utilization"), default=None),
            "peak_used_mb": max(values("memory_used_mb"), default=None),
            "minimum_free_mb": min(values("memory_free_mb"), default=None),
            "peak_power_w": max(values("power_w"), default=None),
            "peak_pcie_rx_kb_s": max(values("pcie_rx_kb_s"), default=None),
            "peak_pcie_tx_kb_s": max(values("pcie_tx_kb_s"), default=None),
        }


def proc_snapshot(
    root_pid: int, prior: tuple[float, float, float, float] | None,
) -> tuple[dict[str, Any], tuple[float, float, float, float]]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    clock_ticks = os.sysconf("SC_CLK_TCK")
    processes: dict[int, tuple[int, float, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            processes[int(entry.name)] = (int(fields[3]), int(fields[13]) + int(fields[14]), int(fields[23]))
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _ticks, _rss) in processes.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    proc_ticks = sum(processes[pid][1] for pid in descendants if pid in processes)
    rss_bytes = sum(processes[pid][2] * page_size for pid in descendants if pid in processes)
    cpu_fields = [float(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    total_ticks = sum(cpu_fields)
    idle_ticks = cpu_fields[3] + (cpu_fields[4] if len(cpu_fields) > 4 else 0)
    now = time.monotonic()
    proc_cpu = system_cpu = None
    if prior is not None and now > prior[0]:
        proc_cpu = max(0.0, (proc_ticks - prior[1]) / clock_ticks / (now - prior[0]) * 100.0)
        total_delta = total_ticks - prior[2]
        idle_delta = idle_ticks - prior[3]
        system_cpu = max(0.0, (total_delta - idle_delta) / total_delta * 100.0) if total_delta > 0 else None
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    sample = {
        "timestamp": utc_now(), "root_pid": root_pid, "process_count": len(descendants),
        "process_cpu_percent": proc_cpu, "system_cpu_percent": system_cpu,
        "process_rss_mb": rss_bytes / 1048576,
        "ram_available_mb": meminfo.get("MemAvailable", 0) / 1048576,
    }
    return sample, (now, proc_ticks, total_ticks, idle_ticks)


def wsl_sampler(pid_file: Path, output: Path, interval: float) -> int:
    prior = None
    with output.open("w", encoding="utf-8", buffering=1) as handle:
        while True:
            try:
                root_pid = int(pid_file.read_text().strip())
                if not Path(f"/proc/{root_pid}").exists():
                    return 0
                sample, prior = proc_snapshot(root_pid, prior)
                handle.write(json.dumps(sample) + "\n")
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(interval)


def read_json_lines(path: str) -> list[dict[str, Any]]:
    converted = subprocess.run(["wsl.exe", "wslpath", "-w", path], capture_output=True, text=True, check=True).stdout.strip()
    local = Path(converted)
    if not local.is_file():
        return []
    rows = []
    for line in local.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError:
            continue
    return rows


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"expected a Windows drive path: {resolved}")
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def summarize_cpu(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = lambda key: [float(row[key]) for row in rows if row.get(key) is not None]
    return {
        "sample_count": len(rows),
        "average_process_cpu_percent": statistics.fmean(values("process_cpu_percent")) if values("process_cpu_percent") else None,
        "peak_process_cpu_percent": max(values("process_cpu_percent"), default=None),
        "average_system_cpu_percent": statistics.fmean(values("system_cpu_percent")) if values("system_cpu_percent") else None,
        "peak_system_cpu_percent": max(values("system_cpu_percent"), default=None),
        "peak_process_rss_mb": max(values("process_rss_mb"), default=None),
        "minimum_ram_available_mb": min(values("ram_available_mb"), default=None),
    }


def stream_chat(port: int, model: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0.0, "top_k": 1, "max_tokens": max_tokens,
        "ignore_eos": True,
    }
    before = http_json(f"http://127.0.0.1:{port}/v1/stats")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST",
    )
    started_utc = utc_now()
    started = time.perf_counter()
    first_token = None
    output: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    first_token = first_token or time.perf_counter()
                    output.append(piece)
    finished = time.perf_counter()
    after = http_json(f"http://127.0.0.1:{port}/v1/stats")
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    ttft = None if first_token is None else first_token - started
    decode_seconds = None if ttft is None else max(0.0, finished - first_token)
    text = "".join(output)
    cursor = -1
    anchors_in_order = True
    for anchor in ANCHORS:
        cursor = text.find(anchor, cursor + 1)
        if cursor < 0:
            anchors_in_order = False
            break
    return {
        "started_utc": started_utc, "finished_utc": utc_now(),
        "wall_seconds": finished - started, "ttft_seconds": ttft,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "estimated_pp_tps": prompt_tokens / ttft if ttft and ttft > 0 else None,
        "estimated_tg_tps": max(0, completion_tokens - 1) / decode_seconds if decode_seconds and decode_seconds > 0 else None,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "output_chars": len(text),
        "retrieval_anchors_in_order": anchors_in_order,
        "output_head": text[:500], "output_tail": text[-500:],
        "stats_before": before, "stats_after": after,
    }


class FreeTokenServer:
    def __init__(self, args: argparse.Namespace, backend: str, label: str):
        self.args, self.backend, self.label = args, backend, label
        self.pid_file = f"/tmp/freetoken-qwen38-{label}.pid"
        self.telemetry_file = f"/tmp/freetoken-qwen38-{label}-telemetry.jsonl"
        self.log_path = args.output.resolve().parent / "logs" / f"freetoken-{label}.log"
        self.process: subprocess.Popen[Any] | None = None
        self.sampler: subprocess.Popen[Any] | None = None
        self.log_handle: Any = None

    def command(self) -> list[str]:
        command = [
            f"{self.args.venv}/bin/ft", "serve", "--model", self.args.model,
            "--served-model-name", "qwen38-next-freetoken", "--gpu", self.args.gpu_uuid,
            "--host", "0.0.0.0", "--port", str(self.args.port),
            "--max-running-requests", "1", "--memory-ratio", str(self.args.memory_ratio),
            "--moe-backend", self.backend, "--moe-cpu-layers", "0", "--moe-cache-auto",
            "--ple-backend", "disk",
            "--kv-reserve-tokens", "8192", "--num-tokens", "8192",
            "--max-prefill-length", "8192", "--max-output-tokens", str(self.args.max_tokens),
            "--cache-type", "naive", "--sampling-defaults", "none",
            "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        ]
        return command

    def start(self) -> tuple[float, dict[str, Any]]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        launch_script = windows_to_wsl(ROOT / "scripts" / "launch-freetoken-wsl.sh")
        subprocess.run(
            ["wsl.exe", "rm", "-f", "--", self.telemetry_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        started = time.perf_counter()
        self.process = subprocess.Popen(
            ["wsl.exe", "bash", launch_script, self.pid_file, *self.command()], stdout=self.log_handle,
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
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(f"FreeToken exited during load ({self.process.returncode}):\n{tail}")
            try:
                health = http_json(f"http://127.0.0.1:{self.args.port}/health")
            except Exception as exc:
                last_error = str(exc)
            else:
                if health.get("status") == "error":
                    raise RuntimeError(f"FreeToken load failed: {health.get('message')}")
                if health.get("status") == "ok":
                    stats = http_json(f"http://127.0.0.1:{self.args.port}/v1/stats")
                    stats["health"] = health
                    return time.perf_counter() - started, stats
                time.sleep(2)
        raise TimeoutError(f"FreeToken was not ready after {self.args.server_timeout}s: {last_error}")

    def stop(self) -> dict[str, Any]:
        try:
            pid_text = subprocess.run(
                ["wsl.exe", "cat", self.pid_file], capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if re.fullmatch(r"[1-9][0-9]*", pid_text):
                stopped = subprocess.run(
                    ["wsl.exe", "kill", "-TERM", "--", f"-{pid_text}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                )
                if stopped.returncode != 0:
                    subprocess.run(
                        ["wsl.exe", "kill", "-TERM", pid_text],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                    )
        finally:
            if self.process is not None:
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=15)
            if self.sampler is not None:
                try:
                    self.sampler.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.sampler.terminate()
            if self.log_handle is not None:
                self.log_handle.close()
        cpu_rows = read_json_lines(self.telemetry_file)
        return {
            "cpu": summarize_cpu(cpu_rows), "log": str(self.log_path),
            "log_tail": self.log_path.read_text(encoding="utf-8", errors="replace")[-8000:],
        }


def run_one(args: argparse.Namespace, server: FreeTokenServer, prompt: str, max_tokens: int, label: str) -> dict[str, Any]:
    gpu = GpuPoller(args.gpu_uuid, args.telemetry_interval)
    gpu.start()
    try:
        row = stream_chat(args.port, "qwen38-next-freetoken", prompt, max_tokens, args.request_timeout)
    finally:
        row_gpu = gpu.stop()
    row["label"] = label
    row["gpu"] = row_gpu
    start = dt.datetime.fromisoformat(row["started_utc"])
    finish = dt.datetime.fromisoformat(row["finished_utc"])
    cpu_rows = [
        sample for sample in read_json_lines(server.telemetry_file)
        if start <= dt.datetime.fromisoformat(sample["timestamp"]) <= finish
    ]
    row["cpu"] = summarize_cpu(cpu_rows)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--venv", default=DEFAULT_VENV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--probe-repetitions", type=int, default=1)
    parser.add_argument("--winner-repetitions", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--memory-ratio", type=float, default=0.90)
    parser.add_argument("--gpu-uuid", default=DEFAULT_GPU_UUID)
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
        return wsl_sampler(args.pid_file, args.sampler_output, args.telemetry_interval)
    prompt = args.prompt.resolve().read_text(encoding="utf-8")
    document: dict[str, Any] = {
        "schema": 1, "created_utc": utc_now(), "updated_utc": utc_now(),
        "objective": "measure the retained FreeToken RTX 5090 GPU-only offload profile",
        "runtime": {"source_commit": "a80b4d308a81986fa086ec173d7faa70ba737b2d", "checkpoint": args.model},
        "protocol": {
            "prompt": str(args.prompt.resolve()), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_tokens": args.max_tokens, "temperature": 0, "top_k": 1,
            "ignore_eos": True, "cache_type": "naive", "concurrency": 1,
            "probe_repetitions": args.probe_repetitions, "winner_repetitions": args.winner_repetitions,
        },
        "probes": [], "winner": None, "winner_runs": [], "summaries": {},
    }
    atomic_json(args.output, document)

    for backend in ("offload",):
        server = FreeTokenServer(args, backend, f"probe-{backend}")
        try:
            load_seconds, ready = server.start()
            warmup = run_one(args, server, prompt, min(args.warmup_tokens, args.max_tokens), f"{backend}-warmup")
            runs = [run_one(args, server, prompt, args.max_tokens, f"{backend}-probe-{i + 1}") for i in range(args.probe_repetitions)]
            document["probes"].append({
                "backend": backend, "command": server.command(), "load_seconds": load_seconds,
                "ready_stats": ready, "warmup": warmup, "runs": runs,
            })
        finally:
            stop = server.stop()
            if document["probes"] and document["probes"][-1].get("backend") == backend:
                document["probes"][-1]["server_telemetry"] = stop
            document["updated_utc"] = utc_now()
            atomic_json(args.output, document)

    candidates = [
        (statistics.median(row["wall_seconds"] for row in probe["runs"]), probe["backend"])
        for probe in document["probes"] if probe.get("runs")
    ]
    if not candidates:
        raise RuntimeError("no successful backend probes")
    _wall, winner = min(candidates)
    document["winner"] = winner
    winner_probe = next(probe for probe in document["probes"] if probe["backend"] == winner)
    # Reuse every already-collected winner run. This matters for a single-backend
    # confirmation pass: three repetitions should require one model load, not two.
    document["winner_runs"] = list(winner_probe["runs"][:args.winner_repetitions])
    remaining = max(0, args.winner_repetitions - len(document["winner_runs"]))
    if remaining:
        server = FreeTokenServer(args, winner, f"winner-{winner}")
        try:
            load_seconds, ready = server.start()
            warmup = run_one(args, server, prompt, min(args.warmup_tokens, args.max_tokens), f"{winner}-winner-warmup")
            document["winner_reload"] = {"load_seconds": load_seconds, "ready_stats": ready, "warmup": warmup}
            for i in range(remaining):
                document["winner_runs"].append(run_one(args, server, prompt, args.max_tokens, f"{winner}-winner-{i + 2}"))
                document["updated_utc"] = utc_now()
                atomic_json(args.output, document)
        finally:
            document.setdefault("winner_reload", {})["server_telemetry"] = server.stop()

    for backend_probe in document["probes"]:
        rows = backend_probe["runs"]
        document["summaries"][f"probe_{backend_probe['backend']}"] = {
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
            "median_estimated_pp_tps": statistics.median(row["estimated_pp_tps"] for row in rows),
            "median_estimated_tg_tps": statistics.median(row["estimated_tg_tps"] for row in rows),
        }
    rows = document["winner_runs"]
    document["summaries"]["winner"] = {
        "backend": winner, "runs": len(rows),
        "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
        "median_estimated_pp_tps": statistics.median(row["estimated_pp_tps"] for row in rows),
        "median_estimated_tg_tps": statistics.median(row["estimated_tg_tps"] for row in rows),
    }
    document["updated_utc"] = utc_now()
    atomic_json(args.output, document)
    print(json.dumps(document["summaries"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
