"""Measure real Strata streams, cold prefill, long decode, queuing, and GPU memory.

Keeps complete requests and SSE events in the ignored raw output directory.
VRAM is sampled, never used as a request-abort threshold.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get("STRATA_API_KEY"):
            self.headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
        args.output.mkdir(parents=True, exist_ok=True)
        self.health = self.request("/health")
        assert self.health["loaded"], self.health
        assert self.health["max_context"] == 262144, self.health
        self.model = self.health["model"]
        self.origin = time.perf_counter()
        self.stop = threading.Event()
        self.samples = []

    def open(self, path, body=None):
        req = urllib.request.Request(
            self.args.base_url + path, headers=self.headers,
            data=None if body is None else json.dumps(body).encode())
        try:
            return urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc

    def request(self, path, body=None):
        with self.open(path, body) as response:
            return json.load(response)

    def sample(self):
        while not self.stop.is_set():
            sample = {"t": time.perf_counter() - self.origin}
            try:
                raw = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                sample["gpus"] = []
                for line in raw.splitlines():
                    index, name, used, free, busy = [s.strip() for s in line.split(",")]
                    sample["gpus"].append({"index": int(index), "name": name,
                        "used_mib": int(used), "free_mib": int(free), "utilization": int(busy)})
            except Exception as exc:
                sample["error"] = str(exc)
            self.samples.append(sample)
            self.stop.wait(0.5)

    def payload(self, target, label, output_tokens):
        topics = ["memory locality", "request scheduling", "cache eviction", "storage bandwidth",
                  "latency measurement", "resource accounting", "failure recovery", "queue fairness"]

        def render(count, padding=0):
            lines = [f"Record {i:06d}: review {topics[i % len(topics)]} with measured examples and documented assumptions."
                     for i in range(count)]
            for fraction, name, code in zip((0.05, 0.5, 0.95), "ABC",
                                            ("cobalt-falcon-174", "amber-otter-826", "violet-heron-593")):
                lines[min(count - 1, int(count * fraction))] = f"Special record {name}: code {code}."
            handbook = ("designing a local inference service" if self.args.handbook_topic == "inference"
                        else "operating a community garden through all four seasons")
            coverage = ("scheduling, memory, measurement, troubleshooting, deployment, and reliability"
                        if self.args.handbook_topic == "inference"
                        else "soil, planting, watering, composting, crop rotation, pests, harvesting, and volunteers")
            text = (f"Benchmark case {self.args.run_id} {label}. Read the following reference records.\n"
                    + "\n".join(lines) + "\nPadding:" + " x" * padding
                    + "\nFirst state the exact codes from special records A, B, and C in order. "
                    f"Then write a comprehensive practical handbook about {handbook}. "
                    "Write at least 80 numbered sections, each with a detailed paragraph and a concrete example. "
                    f"Cover {coverage}. "
                    "Aim for at least 5000 words; keep developing distinct useful sections until all 80 are done. "
                    "Do not stop after an outline or a summary.")
            if self.args.avoid_control_tokens:
                text += (" Do not write or quote any model control tokens, chat delimiters, or stop markers. "
                         "Refer to them only in ordinary words, and skip sections about their syntax.")
            return [{"role": "user", "content": text}]

        def count(messages):
            return self.request("/v1/messages/count_tokens", {"model": self.model,
                "messages": messages, "thinking": {"type": "disabled"}, "max_tokens": output_tokens})["input_tokens"]

        lines = max(3, (target - 180) // 22)
        messages = render(lines)
        actual = count(messages)
        lines = max(3, lines + (target - actual) // 22)
        messages = render(lines)
        actual = count(messages)
        while actual > target:
            lines -= max(1, (actual - target + 20) // 21)
            messages = render(lines)
            actual = count(messages)
        padding = target - actual
        messages = render(lines, padding)
        actual = count(messages)
        if actual != target:
            padding += target - actual
            messages = render(lines, padding)
            actual = count(messages)
        return {"model": self.model, "messages": messages, "temperature": 0,
                "reasoning_effort": "none", "max_tokens": output_tokens, "stream": True,
                "stream_options": {"include_usage": True}}, actual

    def stream(self, label, body, target, trigger=None):
        path = self.args.output / label
        save(path.with_suffix(".request.json"), body)
        start = time.perf_counter()
        result = {"label": label, "target_prompt_tokens": target,
                  "requested_output_tokens": body["max_tokens"],
                  "start_s": start - self.origin,
                  "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                  "done": False, "events": []}
        try:
            with self.open("/v1/chat/completions", body) as response:
                result["headers_s"] = time.perf_counter() - start
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        result["done"] = True
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content:
                            result["events"].append({"t": time.perf_counter() - start,
                                "kind": "reasoning" if delta.get("reasoning_content") else "content", "text": content})
                            if trigger is not None and len(result["events"]) == 128:
                                trigger.set()
                        if choice.get("finish_reason"):
                            result["finish_reason"] = choice["finish_reason"]
                    for key in ("usage", "timings"):
                        if event.get(key):
                            result[key] = event[key]
            result["total_s"] = time.perf_counter() - start
            events = result["events"]
            assert result["done"] and events and result.get("usage"), result
            first, last = events[0]["t"], events[-1]["t"]
            gaps = [b["t"] - a["t"] for a, b in zip(events, events[1:])]
            output = "".join(e["text"] for e in events)
            result.update(ttft_s=first, decode_interval_s=last - first,
                client_decode_tok_s=(result["usage"]["completion_tokens"] - 1) / (last - first),
                effective_prefill_tok_s=result["usage"]["prompt_tokens"] / first,
                nonempty_events=len(events), max_event_gap_s=max(gaps, default=0),
                p95_event_gap_s=statistics.quantiles(gaps, n=100)[94] if len(gaps) > 1 else None,
                codes_correct=all(code in output[:600] for code in (
                    "cobalt-falcon-174", "amber-otter-826", "violet-heron-593")))
            path.with_suffix(".output.txt").write_text(output, encoding="utf-8")
        except Exception as exc:
            result["error"] = repr(exc)
            result["total_s"] = time.perf_counter() - start
            raise
        finally:
            result["end_s"] = time.perf_counter() - self.origin
            save(path.with_suffix(".result.json"), result)
        summary = {k: v for k, v in result.items() if k != "events"}
        print(json.dumps(summary), flush=True)
        return result

    def run(self):
        report = {"started_at": datetime.now(timezone.utc).isoformat(),
                  "label": self.args.label, "health": self.health,
                  "props": {k: v for k, v in self.request("/props").items() if k != "chat_template"},
                  "method": "SSE first/last nonempty event; API actual token usage. Client decode excludes TTFT. "
                            "Event gaps are delivery gaps, not per-token GPU timings. Cold means zero reused KV tokens; "
                            "expert cache and OS file cache stay warm. VRAM is observational, not a stop condition.",
                  "cases": []}
        sampler = threading.Thread(target=self.sample)
        sampler.start()
        try:
            for repeat in range(self.args.repeats):
                for target in self.args.prompts:
                    label = f"p{target}-o{self.args.output_tokens}-r{repeat + 1}"
                    body, counted = self.payload(target, label, self.args.output_tokens)
                    result = self.stream(label, body, target)
                    result["count_endpoint_tokens"] = counted
                    report["cases"].append({k: v for k, v in result.items() if k != "events"})
                    save(self.args.output / "report.json", report)
            if self.args.concurrency:
                a, _ = self.payload(4096, "concurrent-a", self.args.output_tokens)
                b, _ = self.payload(131072, "concurrent-b", 128)
                trigger = threading.Event()
                statuses = []
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fa = pool.submit(self.stream, "concurrent-a", a, 4096, trigger)
                    if not trigger.wait(300):
                        raise RuntimeError("A did not reach 128 nonempty events")
                    fb = pool.submit(self.stream, "concurrent-b", b, 131072)
                    while not (fa.done() and fb.done()):
                        statuses.append({"t": time.perf_counter() - self.origin, "status": self.request("/status")})
                        time.sleep(0.5)
                    ar, br = fa.result(), fb.result()
                save(self.args.output / "concurrency-status.json", statuses)
                report["concurrency"] = {"a": {k: v for k, v in ar.items() if k != "events"},
                    "b": {k: v for k, v in br.items() if k != "events"},
                    "b_first_event_after_a_last_s": br["start_s"] + br["ttft_s"]
                        - ar["start_s"] - ar["events"][-1]["t"]}
        finally:
            self.stop.set()
            sampler.join()
            save(self.args.output / "vram.json", self.samples)
            report["gpu_memory"] = []
            for index in sorted({g["index"] for s in self.samples for g in s.get("gpus", [])}):
                rows = [g for s in self.samples for g in s.get("gpus", []) if g["index"] == index]
                report["gpu_memory"].append({"index": index, "name": rows[0]["name"], "samples": len(rows),
                    "minimum_free_mib": min(g["free_mib"] for g in rows),
                    "maximum_used_mib": max(g["used_mib"] for g in rows)})
            save(self.args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="single-gpu")
    parser.add_argument("--run-id", default="strata-20261003-long-output")
    parser.add_argument("--prompts", type=int, nargs="*", default=[4096, 131072])
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", action="store_true")
    parser.add_argument("--avoid-control-tokens", action="store_true",
                        help="Avoid quoting chat stop markers in the generated handbook")
    parser.add_argument("--handbook-topic", choices=["inference", "gardening"], default="inference")
    Benchmark(parser.parse_args()).run()
