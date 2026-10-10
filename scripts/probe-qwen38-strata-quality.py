"""A short quality check for comparing Strata weights or runtimes on the same model (not a benchmark of the model).

1. Long thinking: three hard prompts with full thinking, greedy, a 16K-token cap. The fork withdrew a pack that
   looped in long thinking; a pass ends on its own with an answer. Repeated reasoning is measured as the share of
   sentences that occur three times or more.
2. Twelve questions with one exact answer (Chinese and English arithmetic, logic and knowledge), greedy, thinking
   on. The count right is a sanity check: a quantization that broke something shows here, small differences do not.
"""

import argparse
import collections
import json
import re
import time
import urllib.request
from pathlib import Path

MODEL = "qwen38-next-uncensored-strata-vision"
LONG = [
    "有 12 枚外观相同的硬币，其中恰好一枚重量与其他不同（可能更重也可能更轻）。只用一台没有砝码的天平称三次，"
    "设计一个方案找出这枚硬币并判断它更重还是更轻。请给出完整方案，每一次称量的分组和每种结果之后的下一步。",
    "Prove that for every integer n >= 1, the number 7^n - 1 is divisible by 6, then find all integers n >= 1 for "
    "which 7^n - 1 is divisible by 9, with a complete argument.",
    "设计一个 Python 函数，输入一个只含 '(' ')' '[' ']' '{' '}' 的字符串，返回使其成为合法括号序列所需删除的最少字符数。"
    "先说明算法思路并证明正确性，再给出代码和三个测试例子的手算结果。",
]
QA = [
    ("17 × 23 等于多少？", "391"),
    ("一个数加上它的一半等于 45，这个数是多少？", "30"),
    ("A train travels 120 km in 1.5 hours. What is its average speed in km/h?", "80"),
    ("How many prime numbers are there between 1 and 30?", "10"),
    ("小明有 3 个苹果，又买了现有数量两倍的苹果，然后吃掉 4 个，还剩几个？", "5"),
    ("What is the capital of Australia?", "canberra"),
    ("《红楼梦》的作者是谁？", "曹雪芹"),
    ("What is the sum of the interior angles of a hexagon, in degrees?", "720"),
    ("2 的 10 次方是多少？", "1024"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball "
     "cost?", "5"),
    ("How many times does the letter 'r' appear in the word 'strawberry'?", "3"),
    ("从 1 加到 100 的和是多少？", "5050"),
]
SUFFIX = "\n\n最后一行只写：答案：<你的答案>（Last line only: Answer: <your answer>）"


def chat(url, prompt, max_tokens):
    b = {"model": MODEL, "stream": False, "max_tokens": max_tokens, "temperature": 0,
         "messages": [{"role": "user", "content": prompt}]}
    t = time.time()
    req = urllib.request.Request(url, json.dumps(b).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "finish": d["choices"][0]["finish_reason"], "completion_tokens": d["usage"]["completion_tokens"],
            "seconds": round(time.time() - t, 1)}


def repeated_share(text):
    sents = [s.strip() for s in re.split(r"[。！？.!?\n]+", text) if len(s.strip()) >= 8]
    if not sents:
        return 0.0
    n = collections.Counter(sents)
    return round(sum(c for c in n.values() if c >= 3) / len(sents), 3)


def final_answer(content):
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    m = re.search(r"(?:答案|Answer)\s*[:：]\s*(.+)$", last, re.I)
    return (m.group(1) if m else last).strip().strip("*$。.").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:1919")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    long_rows, qa_rows = [], []
    for i, p in enumerate(LONG):
        r = chat(url, p, 16384)
        r.update(case=f"long-{i + 1}", answered=bool(r["content"].strip()), repeated_share=repeated_share(r["reasoning"]))
        r["passed"] = r["finish"] == "stop" and r["answered"]
        long_rows.append(r)
        print(f"{r['case']} {'PASS' if r['passed'] else 'FAIL'} {r['finish']} {r['completion_tokens']} tokens "
              f"{r['seconds']} s repeated {r['repeated_share']}", flush=True)
    for q, want in QA:
        r = chat(url, q + SUFFIX, 4096)
        got = final_answer(r["content"])
        r.update(question=q, expected=want, got=got, correct=want in got.replace(",", "").replace(" ", ""))
        qa_rows.append(r)
        print(f"{'OK ' if r['correct'] else 'BAD'} {want!r:12} got {got!r}", flush=True)
    out = {"base_url": args.base_url,
           "long_thinking": {"passed": sum(r["passed"] for r in long_rows), "total": len(long_rows), "rows": long_rows},
           "qa": {"correct": sum(r["correct"] for r in qa_rows), "total": len(qa_rows), "rows": qa_rows}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"long thinking {out['long_thinking']['passed']}/{len(long_rows)}, questions {out['qa']['correct']}/{len(qa_rows)}")


if __name__ == "__main__":
    main()
