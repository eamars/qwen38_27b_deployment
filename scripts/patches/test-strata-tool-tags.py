r"""Replay check for strata-tool-tags: the tool calls whose parameter tag lost its `>` (Xiaoman, 2026-11-07 to 11-11),
fed to the patched parser as the server streams them (stream_tools, recovery on, undeclared calls kept).

Each case is a raw model output rebuilt from the records' structure (her text replaced by filler). Every case runs
whole, one character at a time and in random chunks. A call must arrive with the expected arguments, and the
streamed argument pieces (what pi-ai joins) must parse as JSON. Text outside a call keeps a slipped-looking tag as
written. With --stock, the release's own parser runs the same cases and the ones it breaks are listed.

Usage, with the runtime's serve environment:
  runtime/<rt>/.venv-serve/Scripts/python.exe scripts/patches/test-strata-tool-tags.py runtime/<rt> [--stock <release.zip>]
Exit 0 when every case passes on the patched parser.
"""
import argparse
import importlib.util
import json
import random
import sys
import tempfile
import zipfile
from pathlib import Path

F = "字"
TOOLS = [
    {"name": "think", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}},
                                     "required": ["thought"]}},
    {"name": "write_file", "parameters": {"type": "object", "properties": {
        "content": {"type": "string"}, "overwrite": {"type": "boolean"}, "path": {"type": "string"}},
        "required": ["path", "content"]}},
    {"name": "write_document", "parameters": {"type": "object", "properties": {
        k: {"type": "string"} for k in ("doc", "do", "heading", "why")}, "required": ["why"]}},
]
THINK = "想一想。\n</think>\n\n"
OPEN = "<tool_call>\n<function={}>\n"
CLOSE = "\n</function>\n</tool_call>"
VALUE = F * 25 + ":" + F * 21 + ":" + F * 5 + ":" + F * 67

# name, raw output, expected calls [(name, args)], expected answer text
CASES = [
    ("well-formed call", THINK + OPEN.format("think") + "<parameter=thought>\n" + VALUE + "\n</parameter>" + CLOSE,
     [("think", {"thought": VALUE})], ""),
    ("B: <parameter=NAME without '>'", THINK + OPEN.format("think") + "<parameter=thought\n" + VALUE + "\n</parameter>"
     + CLOSE, [("think", {"thought": VALUE})], ""),
    ("A: an extra parameter without '>' after good ones", THINK + OPEN.format("write_file")
     + "<parameter=content>\n" + F * 300 + "\n</parameter>\n<parameter=path>\n" + F * 17 + "\n</parameter>\n"
     + "<parameter=content_bk\n</parameter>" + CLOSE,
     [("write_file", {"content": F * 300, "path": F * 17, "content_bk": ""})], ""),
    ("C: </parameter without '>' before </function>", THINK + OPEN.format("write_document")
     + "<parameter=doc>\n" + F * 2 + "\n</parameter>\n<parameter=do>\n" + F * 14 + "\n</parameter>\n"
     + "<parameter=heading>\n" + F * 15 + "\n</parameter" + CLOSE,
     [("write_document", {"doc": F * 2, "do": F * 14, "heading": F * 15})], ""),
    ("both slips in one call", THINK + OPEN.format("think") + "<parameter=thought\n" + VALUE + "\n</parameter" + CLOSE,
     [("think", {"thought": VALUE})], ""),
    ("slipped-looking tags in the answer stay text", THINK + "写成 <parameter=name\n然后 </parameter\n就错了。",
     [], "写成 <parameter=name\n然后 </parameter\n就错了。"),
]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(frontend, chunks):
    parser = frontend.OutputParser(thinking=True, tools=TOOLS, stream_tools=True, recover=True, deferred=True)
    events = []
    for piece in chunks:
        events += parser.feed(piece)
    events += parser.finish("stop")
    pieces = "".join(e.text for e in events if e.kind == "tool_args")
    calls = [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]
    content = "".join(e.text for e in events if e.kind == "content").strip("\n")
    return pieces, calls, content


def splits(raw: str, seed: int):
    rng = random.Random(seed)
    chunks, i = [], 0
    while i < len(raw):
        n = rng.randint(1, 9)
        chunks.append(raw[i:i + n])
        i += n
    return {"whole": [raw], "char": list(raw), "random": chunks}


def check(frontend) -> list[str]:
    failures = []
    for k, (name, raw, calls, answer) in enumerate(CASES):
        for mode, chunks in splits(raw, k).items():
            pieces, got, content = run(frontend, chunks)
            problems = []
            if got != calls:
                problems.append(f"calls {str(got)[:120]} != {str(calls)[:120]}")
            if calls:
                try:
                    if json.loads(pieces) != calls[0][1]:
                        problems.append("streamed arguments differ from the call")
                except ValueError as e:
                    problems.append(f"streamed arguments are not JSON ({e}): ...{pieces[-50:]!r}")
            if content != answer:
                problems.append(f"answer {content[:60]!r} != {answer[:60]!r}")
            if problems:
                failures.append(f"{name} [{mode}]: " + "; ".join(problems))
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runtime", type=Path, help="the patched runtime folder (holds serve/frontend.py)")
    ap.add_argument("--stock", type=Path, help="the release zip: also run its unpatched parser, for comparison")
    a = ap.parse_args()
    patched = check(load(a.runtime / "serve" / "frontend.py", "frontend_patched"))
    if a.stock:
        with zipfile.ZipFile(a.stock) as release, tempfile.TemporaryDirectory() as tmp:
            stock_path = Path(tmp) / "frontend.py"
            stock_path.write_bytes(release.read("strata-nvfp4/serve/frontend.py"))
            stock = check(load(stock_path, "frontend_stock"))
        wrong = sorted({f.split(" [", 1)[0] for f in stock})
        print(f"stock parser: {len(wrong)} of {len(CASES)} cases wrong" + "".join(f"\n  - {w}" for w in wrong))
    print(f"patched parser: {len(CASES) * 3 - len(patched)} of {len(CASES) * 3} runs pass")
    for f in patched:
        print("  FAIL", f)
    sys.exit(1 if patched else 0)


if __name__ == "__main__":
    main()
