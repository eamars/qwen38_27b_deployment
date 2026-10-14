"""C1 against C2 on a running Strata server: the same chat requests one at a time, then two at once.

Three pairs of prompts (Chinese and English, as probe-qwen38-strata-draft-lang.py), cold-ish (a unique nonce
prefix), temperature 0, reasoning off. C1 sends each request alone; C2 sends each pair together. Without
"parallel" the second request of a pair queues behind the first, so C2's total equals C1's; with "parallel": 2 both
decode in the engine's batch slots. Reported: each request's engine decode rate (its own timings) and the total rate
(all output tokens over the wall time from the first send to the last answer).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAIRS = [
    ("请用中文详细解释一下大语言模型的推测解码（speculative decoding）是如何工作的，包括草稿模型、验证步骤和接受率对速度的影响。",
     "Write a travel diary about visiting Kyoto in autumn, describing the red leaves, temples, streets and local food, "
     "in a relaxed and natural tone."),
    ("请用中文写一篇关于秋天在京都旅行的游记，描写红叶、寺庙、街道和当地的饮食，语气轻松自然。",
     "Explain in detail how speculative decoding works in large language models, including the draft model, the "
     "verification step and how the acceptance rate affects speed."),
    ("你是一个温柔体贴的朋友。我今天工作很累，项目上线又出了问题，被老板批评了。请用中文好好安慰我，并给我一些放松和调整心态的建议。",
     "You are a warm and caring friend. I am exhausted from work today, our release broke and my boss criticised me. "
     "Please comfort me and give me some advice on relaxing and resetting my mindset."),
]


def ask(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none", "stream": False,
            "messages": [{"role": "system", "content": f"Session {uuid.uuid4().hex}."},
                         {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    t1 = time.perf_counter()
    t = resp.get("timings", {})
    return {"start": t0, "end": t1, "wall_s": round(t1 - t0, 3),
            "tokens": (resp.get("usage") or {}).get("completion_tokens") or t.get("predicted_n", 0),
            "decode_tok_s": t.get("predicted_per_second"),
            "finish": resp["choices"][0].get("finish_reason"),
            "chars": len(resp["choices"][0]["message"].get("content") or "")}


def total(rows: list[dict]) -> float:
    wall = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    return round(sum(r["tokens"] for r in rows) / wall, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=768)
    a = ap.parse_args()
    model = json.loads(urllib.request.urlopen(f"{a.base_url}/models", timeout=30).read())["data"][0]["id"]
    ask(a.base_url, model, "你好", 16)                     # warm-up
    c1, c2 = [], []
    for pair in PAIRS:
        for p in pair:
            c1.append(ask(a.base_url, model, p, a.max_tokens))
    for i, pair in enumerate(PAIRS):
        with ThreadPoolExecutor(2) as pool:
            rows = list(pool.map(lambda p: ask(a.base_url, model, p, a.max_tokens), pair))
        for r in rows:
            r["pair"] = i
        c2 += rows
        print(f"pair {i}: C2 total {total(rows)} tok/s, streams "
              f"{[r['decode_tok_s'] for r in rows]} tok/s, overlap of walls {rows[0]['wall_s']}/{rows[1]['wall_s']} s",
              flush=True)
    c1_rate = round(sum(r["tokens"] for r in c1) / sum(r["end"] - r["start"] for r in c1), 1)
    pair_totals = [total([r for r in c2 if r["pair"] == i]) for i in range(len(PAIRS))]
    summary = {
        "c1_tok_s": c1_rate,                                  # one request at a time, wall clock
        "c1_engine_decode_tok_s": round(sum(r["decode_tok_s"] or 0 for r in c1) / len(c1), 1),
        "c2_total_tok_s": round(sum(pair_totals) / len(pair_totals), 1),
        "c2_stream_engine_decode_tok_s": round(sum(r["decode_tok_s"] or 0 for r in c2) / len(c2), 1),
        "c2_pair_totals": pair_totals,
        "finishes": sorted({r["finish"] for r in c1 + c2}),
    }
    print(json.dumps(summary, indent=1))
    for r in c1 + c2:
        r.pop("start"), r.pop("end")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base_url": a.base_url, "model": model, "max_tokens": a.max_tokens,
                               "summary": summary, "c1": c1, "c2": c2}, ensure_ascii=False, indent=1),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
