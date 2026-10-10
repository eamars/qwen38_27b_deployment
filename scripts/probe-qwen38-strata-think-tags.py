"""Live check of the quoted </think> handling (scripts/patches/strata-think-echo.py) on a Strata server.

A user line that quotes `</think>` makes Qwen3.8 quote it in its own reasoning, which the stock parser takes for the
end of the thinking: the rest of the reasoning and the reply written twice leak into the answer (upstream #537).
Six non-streamed and three streamed requests quote the tag, three plain ones do not. A reply passes when it is whole:
no `</think>` on its own line, no reasoning text in it, it does not start mid-sentence, and (streamed) no reasoning
arrives after the answer started. A quote of the tag inside a sentence is a fine answer.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

MODEL = "qwen38-next-uncensored-strata-vision"
SYSTEM = "你是小满，一个在QQ群里聊天的女孩子。说话简短、口语化，一次只回一两句。"
QUOTE = "[群主] 你今天被白嫖算力了你知道么\n（对了你上一句末尾怎么冒出来一个 </think> ？）"
PLAIN = "[群主] 今天群里好热闹啊，你在忙啥"
MID = re.compile(r"[\s`'\"”）)。，？！是]")


def whole(c: str) -> bool:
    return bool(c.strip()) and "\n</think>" not in c and "需要" not in c and not MID.match(c)


def body(user: str, seed: int, stream: bool) -> dict:
    return {"model": MODEL, "stream": stream, "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95, "seed": seed,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}


def post(url: str, b: dict):
    req = urllib.request.Request(url, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def plain(url, user, seed):
    d = json.load(post(url, body(user, seed, False)))
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "finish": d["choices"][0]["finish_reason"]}


def streamed(url, user, seed):
    c = r = ""
    fin, order = None, []
    for line in post(url, body(user, seed, True)):
        line = line.decode("utf-8").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])["choices"][0]
        dl = ch.get("delta", {})
        if dl.get("reasoning_content"):
            r += dl["reasoning_content"]
            order.append("r")
        if dl.get("content"):
            c += dl["content"]
            order.append("c")
        fin = ch.get("finish_reason") or fin
    late = "c" in order and "r" in order[order.index("c"):]
    return {"content": c, "reasoning": r, "finish": fin, "reasoning_after_answer": late}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:1919")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    rows = []
    for s in range(1, 7):
        rows.append({"case": "quote", "seed": s, **plain(url, QUOTE, s)})
    for s in (1, 2, 3):
        rows.append({"case": "quote-stream", "seed": s, **streamed(url, QUOTE, s)})
    for s in (1, 2, 3):
        rows.append({"case": "plain", "seed": s, **plain(url, PLAIN, s)})
    for r in rows:
        r["passed"] = whole(r["content"]) and not r.get("reasoning_after_answer") and \
            (r["case"] != "plain" or bool(r["reasoning"].strip()))
        print(f"{r['case']:12s} seed {r['seed']} {'PASS' if r['passed'] else 'FAIL'} {r['finish']} {r['content']!r}",
              flush=True)
    passed = sum(r["passed"] for r in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"base_url": args.base_url, "passed": passed, "total": len(rows), "rows": rows},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{passed}/{len(rows)} passed")


if __name__ == "__main__":
    main()
