"""Exercise the real Strata API: text, streaming, tools, images, and cached retrieval."""

import argparse
import base64
import json
import os
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib


def image_fixture():
    """A PNG with a red circle on the left and a blue square on the right."""
    width, height = 336, 168
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            color = (255, 255, 255)
            if (x - 84) ** 2 + (y - 84) ** 2 < 48 ** 2:
                color = (230, 25, 25)
            elif 208 <= x < 304 and 36 <= y < 132:
                color = (20, 55, 230)
            rows.extend(color)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-tokens", type=int, default=4000)
    parser.add_argument("--only-long", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Content-Type": "application/json"}
    if os.environ.get("STRATA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]

    def open_request(path, payload=None):
        request = urllib.request.Request(args.base_url + path, headers=headers, data=None if payload is None else json.dumps(payload).encode())
        try:
            return urllib.request.urlopen(request, timeout=900)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error

    def request(path, payload=None):
        with open_request(path, payload) as response:
            return json.load(response)

    health = request("/health")
    assert health["loaded"] and health["images"], health
    model = health["model"]
    report = {"health": health, "models": request("/v1/models"), "cases": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def payload(messages, **extra):
        return dict(model=model, messages=messages, max_tokens=128, temperature=0, reasoning_effort="none", **extra)

    def chat(messages, **extra):
        return request("/v1/chat/completions", payload(messages, **extra))

    def case(name, run):
        started = time.perf_counter()
        result = {"name": name}
        try:
            result.update(run())
            result["passed"] = True
        except Exception as error:
            result.update(passed=False, error=str(error))
        result["wall_seconds"] = round(time.perf_counter() - started, 3)
        report["cases"].append(result)
        save()
        print(json.dumps(result), flush=True)

    def text():
        response = chat([{"role": "user", "content": "What is 19 + 23? Reply only with the number."}])
        assert response["choices"][0]["message"]["content"].strip() == "42", response
        return {"response": response}

    def stream():
        events, pieces, done = [], [], False
        with open_request("/v1/chat/completions", payload([{"role": "user", "content": "What is 11 + 18? Reply only with the number."}], stream=True)) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                assert "error" not in event, event
                events.append(event)
                for choice in event.get("choices", []):
                    pieces.append(choice.get("delta", {}).get("content") or "")
        output = "".join(pieces).strip()
        assert done and output == "29", {"done": done, "content": output, "events": events}
        return {"content": output, "events": events}

    def tools():
        messages = [{"role": "user", "content": "Use the get_weather tool to check the weather in Wellington."}]
        definitions = [{"type": "function", "function": {"name": "get_weather", "description": "Look up weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        response = chat(messages, tools=definitions)
        message = response["choices"][0]["message"]
        call = message["tool_calls"][0]
        assert call["function"]["name"] == "get_weather", response
        assert json.loads(call["function"]["arguments"])["city"].lower() == "wellington", response
        messages += [message, {"role": "tool", "tool_call_id": call["id"], "content": '{"temperature_c":17,"condition":"sunny"}'}]
        continuation = chat(messages, tools=definitions)
        assert "17" in continuation["choices"][0]["message"]["content"], continuation
        return {"response": response, "continuation": continuation}

    def vision():
        png = image_fixture()
        args.output.with_suffix(".png").write_bytes(png)
        content = [{"type": "text", "text": "Describe the color and shape on the left, then the color and shape on the right. Be brief."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
        response = chat([{"role": "user", "content": content}])
        output = response["choices"][0]["message"]["content"].lower()
        assert all(word in output for word in ["red", "circle", "blue", "square"]), response
        return {"response": response}

    def retrieval():
        codes = ["cobalt-falcon-174", "amber-otter-826", "violet-heron-593"]

        def make_prompt(count):
            lines = [f"Reference item {i:06d}: the ordinary marker is stone." for i in range(count)]
            for fraction, label, code in zip([0.05, 0.5, 0.95], "ABC", codes):
                lines[int(count * fraction)] = f"Special record {label}: the secret code is {code}."
            return "Read these records and retain the three secret codes.\n" + "\n".join(lines) + "\nReturn the secret codes from special records A, B, and C, in that order."

        count = max(10, args.long_tokens // 18)
        prompt = make_prompt(count)
        measured = request("/v1/messages/count_tokens", {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128})["input_tokens"]
        count = max(10, int(count * args.long_tokens / measured))
        prompt = make_prompt(count)
        messages = [{"role": "user", "content": prompt}]
        cold = chat(messages)
        for code in codes:
            assert code in cold["choices"][0]["message"]["content"], cold
        warm = chat(messages)
        for code in codes:
            assert code in warm["choices"][0]["message"]["content"], warm
        assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, warm
        return {"requested_prompt_tokens": args.long_tokens, "cold": cold, "warm": warm}

    if not args.only_long:
        case("text", text)
        case("stream", stream)
        case("tool_round_trip", tools)
        case("image", vision)
    case("retrieval_and_reuse", retrieval)
    report["passed"] = all(c["passed"] for c in report["cases"])
    save()
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
