"""Build a tokenizer-calibrated deep prompt and profile one local backend."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.request
import uuid
from pathlib import Path


FACTS = (
    "BEGIN-FACT-5090-ORCHID-7241",
    "MIDDLE-FACT-QWEN38-AX9-5520",
    "END-FACT-DFLASH-9137-LANTERN",
)
FILLER = (
    "Repository context record: preserve the existing Windows deployment constraints, "
    "keep target KV at Q8_0 or better, keep draft KV at F16, and do not introduce CPU "
    "offload. This record is intentionally repetitive for tokenizer calibration.\n"
)


def post_json(port: int, path: str, payload: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_metrics(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(None, 1)
        try:
            values[name] = float(value)
        except ValueError:
            pass
    return values


def make_prompt(repetitions: int) -> str:
    first_half = repetitions // 2
    second_half = repetitions - first_half
    return (
        "Deep-context retrieval test. The first immutable fact is " + FACTS[0] + ".\n"
        + FILLER * first_half
        + "The middle immutable fact is " + FACTS[1] + ".\n"
        + FILLER * second_half
        + "The far-end immutable fact is " + FACTS[2] + ".\n"
        + "Answer the final task using all three immutable facts. For the sustained profile, "
        "continue with a detailed technical explanation after listing them. Unique run: "
        + uuid.uuid4().hex
        + "\nFinal task: list the three immutable facts in their original order, exactly."
    )


def calibrate(port: int, target_tokens: int) -> tuple[str, int, int]:
    low, high = 0, max(100, target_tokens // 8)
    while len(post_json(port, "/tokenize", {"content": make_prompt(high)})["tokens"]) < target_tokens:
        high *= 2
    best = ("", 0, 10**9)
    while low <= high:
        middle = (low + high) // 2
        prompt = make_prompt(middle)
        count = len(post_json(port, "/tokenize", {"content": prompt})["tokens"])
        distance = abs(count - target_tokens)
        if distance < best[2]:
            best = (prompt, count, distance)
        if count < target_tokens:
            low = middle + 1
        elif count > target_tokens:
            high = middle - 1
        else:
            break
    return best[0], best[1], best[2]


def stream_request(port: int, model: str, prompt: str, max_tokens: int) -> dict:
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
    before = get_metrics(port)
    sent = time.perf_counter()
    first_token = None
    output: list[str] = []
    with urllib.request.urlopen(request, timeout=1800) as response:
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
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first_token is None:
                        first_token = time.perf_counter()
                    output.append(piece)
    finished = time.perf_counter()
    after = get_metrics(port)
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in set(before) | set(after)}
    prompt_seconds = delta.get("llamacpp:prompt_seconds_total", 0.0)
    predicted_seconds = delta.get("llamacpp:tokens_predicted_seconds_total", 0.0)
    prompt_tokens = delta.get("llamacpp:prompt_tokens_total", 0.0)
    predicted_tokens = delta.get("llamacpp:tokens_predicted_total", 0.0)
    drafted = delta.get("llamacpp:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("llamacpp:spec_decode_num_accepted_tokens_total", 0.0)
    drafts = delta.get("llamacpp:spec_decode_num_drafts_total", 0.0)
    text = "".join(output)
    return {
        "ttft_seconds": None if first_token is None else first_token - sent,
        "wall_seconds": finished - sent,
        "output_chars": len(text),
        "output": text,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_cached": delta.get("llamacpp:prompt_tokens_cached_total", 0.0),
        "pp_tokens_per_second": None if prompt_seconds <= 0 else prompt_tokens / prompt_seconds,
        "predicted_tokens": predicted_tokens,
        "tg_tokens_per_second": None if predicted_seconds <= 0 else predicted_tokens / predicted_seconds,
        "drafted_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "acceptance_ratio": None if drafted <= 0 else accepted / drafted,
        "mean_accepted_per_verification": None if drafts <= 0 else accepted / drafts,
        "metrics_delta": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompt, calibrated_tokens, calibration_error = calibrate(args.port, args.target_tokens)
    print(f"Calibrated content tokens: {calibrated_tokens} (target {args.target_tokens}, error {calibration_error})")
    result = stream_request(args.port, args.model, prompt, args.max_tokens)
    result.update(
        {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "port": args.port,
            "model": args.model,
            "target_content_tokens": args.target_tokens,
            "calibrated_content_tokens": calibrated_tokens,
            "calibration_error": calibration_error,
            "max_tokens": args.max_tokens,
            "retrieval_correct": all(fact in result["output"] for fact in FACTS),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("port", "calibrated_content_tokens", "prompt_tokens", "ttft_seconds", "tg_tokens_per_second", "mean_accepted_per_verification", "retrieval_correct")}, sort_keys=True))
    print(f"Deep-context profile written: {output}")


if __name__ == "__main__":
    main()
