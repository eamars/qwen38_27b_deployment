r"""Replay check for strata-think-echo: the reply shapes that leaked on this server, fed to the patched parser.

Each case is a raw model output (thinking, `</think>`, answer) of a shape recorded from Qwen3.8 on Strata between
0.1.37 and 0.2.0 (upstream #537, #1053; docs/qwen38-flash-next-strata.md). Every case runs three ways: whole, one
character at a time, and in random chunks; the events must agree and match the expected reasoning/answer split. With
--stock, the release's own parser runs the same cases and the cases it gets wrong are listed (the reason the patch
stays).

Usage, with the runtime's serve environment (the parser imports jinja2):
  runtime/<rt>/.venv-serve/Scripts/python.exe scripts/patches/test-strata-think-echo.py runtime/<rt> [--stock <release.zip>]
Exit 0 when every case passes on the patched parser.
"""
import argparse
import importlib.util
import random
import sys
import tempfile
import zipfile
from pathlib import Path

TOOLS = [{"name": "read", "description": "read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
CALL = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
LONG = "长" * 600

# name, raw output, expected answer (content), text the reasoning must hold (or None), expected tool calls
CASES = [
    ("plain close", "想一想。\n</think>\n\n你好呀。", "你好呀。", "想一想。", []),
    ("template close after text", "想一想。</think>\n\n你好呀。", "你好呀。", "想一想。", []),
    ("no thinking text", "\n</think>\n\n在呢。", "在呢。", None, []),
    ("thinking quotes the tag mid-line", "用户问 </think> 是什么标签。\n应该解释。\n</think>\n\n是思考结束标记。",
     "是思考结束标记。", "用户问 </think> 是什么标签。", []),
    ("thinking quotes the tag after a quote mark", '他写了"</think>"。\n</think>\n\n收到。', "收到。", '他写了"</think>"', []),
    ("thinking quotes the tag at a line start", "要不要输出\n</think> 这个词？不用。\n</think>\n\n不用。", "不用。",
     "</think> 这个词？不用。", []),
    ("answer written twice around a second close", "先想。\n</think>\n\n我在呢，主人。\n</think>\n\n我在呢，主人。",
     "我在呢，主人。", "我在呢，主人。", []),
    ("second close, then a different answer", "先想。\n</think>\n\n草稿回答。\n</think>\n\n最终回答。",
     "最终回答。", "草稿回答。", []),
    ("answer quotes the tag mid-line", "先想。\n</think>\n\n知道呀，</think> 是标签。", "知道呀，</think> 是标签。", None, []),
    ("answer line starts with the tag and text", "先想。\n</think>\n\n前半句。\n</think> 是手滑", "前半句。\n</think> 是手滑",
     None, []),
    ("second close before a tool call", "先想。\n</think>\n\n我去读文件。\n</think>\n\n" + CALL, "我去读文件。", None,
     ["read"]),
    ("tool call right after the close", "要读文件。\n</think>\n\n" + CALL, "", "要读文件。", ["read"]),
    ("call written in the thinking, then the close", CALL + "</think>\n", "", None, ["read"]),
    ("a fence the thinking opened, a repeated close, then a call",
     "内容：\n```\n样例\n</think>\n答案样例\n```\n现在调用。\n</think>\n\n" + CALL, "答案样例\n```\n现在调用。", "样例", ["read"]),
    ("second close after a long answer", "先想。\n</think>\n\n" + LONG + "\n</think>\n\n尾", LONG + "尾", None, []),
    ("ends still thinking after a quoted tag", "答案就是</think>四十二", "四十二", None, []),
]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(frontend, raw: str, chunks: list[str]):
    parser = frontend.OutputParser(thinking=True, tools=TOOLS, recover=True)
    events = []
    for piece in chunks:
        events += parser.feed(piece)
    events += parser.finish("stop")
    reasoning = "".join(e.text for e in events if e.kind == "reasoning")
    content = "".join(e.text for e in events if e.kind == "content")
    calls = [e.call.name for e in events if e.kind == "tool_call"]
    return reasoning, content.strip("\n"), calls


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
    for k, (name, raw, answer, thought, calls) in enumerate(CASES):
        results = {mode: run(frontend, raw, chunks) for mode, chunks in splits(raw, k).items()}
        for mode, (reasoning, content, got_calls) in results.items():
            problems = []
            if content != answer:
                problems.append(f"answer {content[:80]!r} != {answer[:80]!r}")
            if thought is not None and thought not in reasoning:
                problems.append(f"reasoning lacks {thought!r}: {reasoning[:80]!r}")
            if got_calls != calls:
                problems.append(f"calls {got_calls} != {calls}")
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
