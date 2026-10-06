"""Compare MTP draft acceptance and decode speed by output language on a running Strata server.

Each prompt is sent once per repeat, cold-ish (a unique nonce prefix defeats prefix reuse), at temperature 0 with
reasoning disabled. Strata's own per-request timings supply draft counts; the result keeps every response text so a
language mismatch (e.g. an English answer to a Chinese prompt) is visible. Used to decide whether the MTP draft
vocabulary needs CJK rows (tools/draft_vocab.py --add cjk).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path

PROMPTS = {
    "zh": [
        "请用中文详细解释一下大语言模型的推测解码（speculative decoding）是如何工作的，包括草稿模型、验证步骤和接受率对速度的影响。",
        "请用中文写一篇关于秋天在京都旅行的游记，描写红叶、寺庙、街道和当地的饮食，语气轻松自然。",
        "你是一个温柔体贴的朋友。我今天工作很累，项目上线又出了问题，被老板批评了。请用中文好好安慰我，并给我一些放松和调整心态的建议。",
    ],
    "en": [
        "Explain in detail how speculative decoding works in large language models, including the draft model, the verification step and how the acceptance rate affects speed.",
        "Write a travel diary about visiting Kyoto in autumn, describing the red leaves, temples, streets and local food, in a relaxed and natural tone.",
        "You are a warm and caring friend. I am exhausted from work today, our release broke and my boss criticised me. Please comfort me and give me some advice on relaxing and resetting my mindset.",
    ],
}
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    model = json.loads(urllib.request.urlopen(f"{a.base_url}/models", timeout=30).read())["data"][0]["id"]
    rows = []
    for rep in range(a.repeats):
        for lang, prompts in PROMPTS.items():
            for i, prompt in enumerate(prompts):
                body = {
                    "model": model,
                    "messages": [{"role": "system", "content": f"Session {uuid.uuid4().hex}."},
                                 {"role": "user", "content": prompt}],
                    "max_tokens": a.max_tokens, "temperature": 0, "reasoning_effort": "none", "stream": False,
                }
                t0 = time.perf_counter()
                resp = post(f"{a.base_url}/chat/completions", body, 600)
                wall = time.perf_counter() - t0
                text = resp["choices"][0]["message"].get("content") or ""
                t = resp.get("timings", {})
                gen, acc, drafted = t.get("predicted_n", 0), t.get("draft_n_accepted", 0), t.get("draft_n", 0)
                rounds = max(1, gen - acc)
                row = {"lang": lang, "prompt": i, "repeat": rep, "wall_s": round(wall, 3),
                       "generated": gen, "drafted": drafted, "accepted": acc,
                       "acceptance": round(acc / drafted, 3) if drafted else None,
                       "tokens_per_round": round(gen / rounds, 3), "ms_per_round": round(t.get("predicted_ms", 0) / rounds, 3),
                       "decode_tok_s": t.get("predicted_per_second"),
                       "cjk_share": round(len(CJK.findall(text)) / max(1, len(text)), 3),
                       "finish_reason": resp["choices"][0].get("finish_reason"), "text": text}
                rows.append(row)
                print(f"{lang} p{i} r{rep}: {gen} tok, {row['tokens_per_round']} tok/round, accept {row['acceptance']}, "
                      f"{row['decode_tok_s']} tok/s, cjk {row['cjk_share']}", flush=True)
    summary = {}
    for lang in PROMPTS:
        sel = [r for r in rows if r["lang"] == lang]
        gen = sum(r["generated"] for r in sel); acc = sum(r["accepted"] for r in sel)
        drafted = sum(r["drafted"] for r in sel)
        ms = sum(r["ms_per_round"] * max(1, r["generated"] - r["accepted"]) for r in sel)
        summary[lang] = {"requests": len(sel), "generated": gen, "acceptance": round(acc / max(1, drafted), 3),
                         "tokens_per_round": round(gen / max(1, gen - acc), 3),
                         "ms_per_round": round(ms / max(1, gen - acc), 3), "decode_tok_s": round(gen / ms * 1000, 1)}
    print(json.dumps(summary, indent=1))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": a.label, "base_url": a.base_url, "model": model, "max_tokens": a.max_tokens,
                               "summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
