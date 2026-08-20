"""Run small client-side streaming checks against the two local llama.cpp APIs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def metrics(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(None, 1)
        try:
            values[name] = float(value)
        except ValueError:
            continue
    return values


def stream_chat(port: int, model: str, prompt: str, max_tokens: int) -> dict:
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
    before = metrics(port)
    sent = time.perf_counter()
    first_token = None
    events = 0
    output_parts: list[str] = []
    usage = None
    with urllib.request.urlopen(request, timeout=900) as response:
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                continue
            event = json.loads(raw.decode("utf-8"))
            events += 1
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first_token is None:
                        first_token = time.perf_counter()
                    output_parts.append(piece)
    finished = time.perf_counter()
    after = metrics(port)
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in set(before) | set(after)}
    prompt_seconds = delta.get("llamacpp:prompt_seconds_total", 0.0)
    predicted_seconds = delta.get("llamacpp:tokens_predicted_seconds_total", 0.0)
    prompt_tokens = delta.get("llamacpp:prompt_tokens_total", 0.0)
    predicted_tokens = delta.get("llamacpp:tokens_predicted_total", 0.0)
    drafts = delta.get("llamacpp:spec_decode_num_drafts_total", 0.0)
    drafted_tokens = delta.get("llamacpp:spec_decode_num_draft_tokens_total", 0.0)
    accepted_tokens = delta.get("llamacpp:spec_decode_num_accepted_tokens_total", 0.0)
    return {
        "sent_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "port": port,
        "model": model,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "events": events,
        "output_chars": len("".join(output_parts)),
        "usage": usage,
        "ttft_seconds": None if first_token is None else first_token - sent,
        "wall_seconds": finished - sent,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_cached": delta.get("llamacpp:prompt_tokens_cached_total", 0.0),
        "pp_tokens_per_second": None if prompt_seconds <= 0 else prompt_tokens / prompt_seconds,
        "predicted_tokens": predicted_tokens,
        "tg_tokens_per_second": None if predicted_seconds <= 0 else predicted_tokens / predicted_seconds,
        "draft_verification_steps": drafts,
        "drafted_tokens": drafted_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "acceptance_ratio": None if drafted_tokens <= 0 else accepted_tokens / drafted_tokens,
        "mean_accepted_per_verification": None if drafts <= 0 else accepted_tokens / drafts,
        "metrics_delta": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/api-smoke.json")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    endpoints = [(8080, "qwen3.8-27b-dflash2-5090"), (8081, "qwen3.8-27b-dflash2-4090")]
    results = []
    for port, model in endpoints:
        health = http_json(f"http://127.0.0.1:{port}/health")
        models = http_json(f"http://127.0.0.1:{port}/v1/models")
        print(f"port {port}: {health}; model {model}")
        for repetition in range(args.repetitions):
            nonce = uuid.uuid4().hex
            prompt = (
                "Write a compact Python function that validates a UTF-8 JSON document, "
                f"then explain the edge cases. Unique run nonce: {nonce}."
            )
            result = stream_chat(port, model, prompt, args.max_tokens)
            result["repetition"] = repetition + 1
            result["health"] = health
            result["models"] = models
            results.append(result)
            print(json.dumps({key: result[key] for key in (
                "port", "repetition", "ttft_seconds", "wall_seconds", "prompt_tokens",
                "pp_tokens_per_second", "tg_tokens_per_second", "drafted_tokens",
                "accepted_draft_tokens", "mean_accepted_per_verification"
            )}, sort_keys=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "results": results}, indent=2), encoding="utf-8")
    print(f"API profile written: {output}")


if __name__ == "__main__":
    main()
