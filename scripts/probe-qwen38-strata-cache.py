"""Exercise sequential independent histories and real model tool calls on Strata.

The probe executes only its deterministic local lookup fixture; it does not
dispatch model-selected tools into Asuna or modify Asuna conversations.
"""

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request


TOOL = {"type": "function", "function": {"name": "lookup_probe_value",
    "description": "Read the deterministic cache-test value for this agent and round.",
    "parameters": {"type": "object", "properties": {
        "agent": {"type": "string", "enum": ["A", "B"]}, "round": {"type": "integer"}},
        "required": ["agent", "round"], "additionalProperties": False}}}


class Probe:
    def __init__(self, args):
        self.args = args
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get("STRATA_API_KEY"):
            self.headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
        args.output.mkdir(parents=True, exist_ok=True)
        self.health = self.request("/health")
        self.model = args.model or self.health["model"]
        self.report = {"started_at": datetime.now(timezone.utc).isoformat(),
            "label": args.label, "health": self.health, "target_history_tokens": args.tokens,
            "reasoning_effort": args.reasoning_effort, "cases": []}
        self.log_offset = args.engine_log.stat().st_size if args.engine_log.exists() else None
        self.stop = threading.Event()
        self.samples = []

    def sample(self):
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in ("total", "available", "page_total", "page_available",
                                                    "virtual_total", "virtual_available", "extended")]
        while not self.stop.is_set():
            sample = {"at": datetime.now(timezone.utc).isoformat()}
            try:
                if os.name == "nt":
                    memory = MemoryStatus()
                    memory.length = ctypes.sizeof(memory)
                    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                        raise ctypes.WinError()
                    sample["ram_available_mib"] = memory.available / 1048576
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits"], text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                sample["gpu_free_mib"] = {index.strip(): int(free) for index, free in
                    (line.split(",") for line in raw.splitlines())}
            except Exception as exc:
                sample["error"] = str(exc)
            self.samples.append(sample)
            self.stop.wait(0.5)

    def request(self, path, body=None):
        req = urllib.request.Request(self.args.base_url + path, headers=self.headers,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode())
        try:
            with urllib.request.urlopen(req, timeout=1200) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc

    def save(self):
        (self.args.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")

    def history(self, agent):
        secret = {"A": "cobalt-falcon-174", "B": "amber-otter-826"}[agent]
        system = (f"You are independent agent {agent}, benchmark {self.args.run_id}. "
            "Keep your own history separate from the other agent. When asked to call lookup_probe_value, "
            "call it once. After its result, reply with your private code and the exact returned value, briefly. "
            "Never call a tool again unless a new user message asks for it.")

        def messages(lines):
            records = [f"{agent} archive record {i:06d}: routine reference information remains stable." for i in range(lines)]
            records[len(records) // 2] = f"Private code for agent {agent}: {secret}."
            return [{"role": "system", "content": system}, {"role": "user", "content":
                "Read and retain this independent history.\n" + "\n".join(records)
                + f"\nRound 1: call lookup_probe_value with agent {agent} and round 1 now."}]

        def count_tokens(result):
            return self.request("/v1/messages/count_tokens", {"model": self.model,
                "system": system, "messages": result[1:], "max_tokens": 512,
                "tools": [{"name": TOOL["function"]["name"], "description": TOOL["function"]["description"],
                           "input_schema": TOOL["function"]["parameters"]}],
                "thinking": {"type": "disabled"}})["input_tokens"]

        count = max(10, (self.args.tokens - 550) // 18)
        result = messages(count)
        counted = count_tokens(result)
        while counted > self.args.tokens:
            count -= max(1, (counted - self.args.tokens + 17) // 18)
            result = messages(count)
            counted = count_tokens(result)
        result[1]["content"] += " x" * (self.args.tokens - counted)
        assert count_tokens(result) == self.args.tokens
        return result

    def call(self, label, messages, expect_tool=False, **overrides):
        payload = {"model": self.model, "messages": copy.deepcopy(messages), "tools": [TOOL],
                   "max_tokens": 512, "temperature": 0, "reasoning_effort": self.args.reasoning_effort,
                   **overrides}
        (self.args.output / (label + ".request.json")).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        start = time.perf_counter()
        response = self.request("/v1/chat/completions", payload)
        record = {"label": label, "wall_s": time.perf_counter() - start, "response": response}
        self.report["cases"].append(record)
        self.save()
        message = response["choices"][0]["message"]
        usage = response["usage"]
        print(json.dumps({"label": label, "wall_s": round(record["wall_s"], 3), "usage": usage,
                          "timings": response.get("timings"),
                          "tool": bool(message.get("tool_calls")),
                          "content": (message.get("content") or "")[:240]}), flush=True)
        if expect_tool:
            assert len(message.get("tool_calls", [])) == 1, record
            assert message["tool_calls"][0]["function"]["name"] == "lookup_probe_value", record
        messages.append(message)
        return message

    @staticmethod
    def tool_result(messages, agent, round_id):
        call = messages[-1]["tool_calls"][0]
        arguments = json.loads(call["function"]["arguments"])
        assert arguments == {"agent": agent, "round": round_id}, arguments
        value = f"{agent}-round-{round_id}-verified"
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": json.dumps({"agent": agent, "round": round_id, "value": value})})
        return value

    def continuation(self, label, messages, agent, round_id):
        value = self.tool_result(messages, agent, round_id)
        message = self.call(label, messages)
        content = message.get("content") or ""
        secret = {"A": "cobalt-falcon-174", "B": "amber-otter-826"}[agent]
        assert secret in content and value in content, message
        assert not message.get("tool_calls"), message

    def run_cases(self):
        histories = {agent: self.history(agent) for agent in "AB"}
        # A contiguous tool call/result is the cache-hit control.
        self.call("a1-tool", histories["A"], expect_tool=True)
        self.continuation("a1-result-contiguous", histories["A"], "A", 1)
        self.call("b1-tool-switch", histories["B"], expect_tool=True)
        for round_id in range(2, self.args.rounds + 1):
            histories["A"].append({"role": "user", "content":
                f"Round {round_id}: call lookup_probe_value for agent A and round {round_id}."})
            self.call(f"a{round_id}-tool-switch", histories["A"], expect_tool=True)
            self.continuation(f"b{round_id - 1}-result-switch", histories["B"], "B", round_id - 1)
            self.continuation(f"a{round_id}-result-switch", histories["A"], "A", round_id)
            histories["B"].append({"role": "user", "content":
                f"Round {round_id}: call lookup_probe_value for agent B and round {round_id}."})
            self.call(f"b{round_id}-tool-switch", histories["B"], expect_tool=True)
        self.continuation(f"b{self.args.rounds}-result-contiguous", histories["B"], "B", self.args.rounds)
        if self.args.prefix_controls:
            for kind in ("unchanged", "tool-schema", "system"):
                branch = copy.deepcopy(histories["A"])
                branch.append({"role": "user", "content":
                    f"Round 99: call lookup_probe_value for agent A and round 99."})
                tools = copy.deepcopy([TOOL])
                if kind == "tool-schema":
                    tools[0]["function"]["description"] += " New declaration version."
                elif kind == "system":
                    branch[0]["content"] = "Changed system prefix. " + branch[0]["content"]
                self.call("prefix-" + kind, branch, expect_tool=True, tools=tools)
                cached = self.report["cases"][-1]["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]
                assert (cached > 0) == (kind == "unchanged"), (kind, cached)
        if self.args.third_context:
            # A small unrelated request models an auxiliary title/summary call.
            self.call("third-context", [{"role": "system", "content": "You are unrelated agent C."},
                {"role": "user", "content": "Reply with only C-ready."}], tools=[])
            for agent in "AB":
                histories[agent].append({"role": "user", "content":
                    f"Round 99: call lookup_probe_value for agent {agent} and round 99."})
                self.call(agent.lower() + "-after-third", histories[agent], expect_tool=True)
        for agent, messages in histories.items():
            (self.args.output / (agent + ".history.json")).write_text(json.dumps(messages, indent=2) + "\n", encoding="utf-8")
        self.report["passed"] = True
        self.save()

    def run(self):
        sampler = threading.Thread(target=self.sample, daemon=True)
        sampler.start()
        try:
            self.run_cases()
        finally:
            self.stop.set()
            sampler.join(timeout=6)
            self.report["memory_samples"] = self.samples
            if self.log_offset is not None:
                with self.args.engine_log.open("rb") as source:
                    source.seek(self.log_offset)
                    (self.args.output / "engine.log").write_bytes(source.read())
            self.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="default-cache")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--run-id", default="strata-cache-20261003")
    parser.add_argument("--engine-log", type=Path, default=Path("benchmarks/raw/qwen38-strata/server/engine-1919.log"))
    parser.add_argument("--prefix-controls", action="store_true")
    parser.add_argument("--third-context", action="store_true")
    Probe(parser.parse_args()).run()
