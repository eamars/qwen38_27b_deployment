"""Same-session A/B of the production Strata runtime against a staged candidate, one at a time on 127.0.0.1:1921.

Stage the candidate beside production first (unzip the release into runtime/strata-nvfp4-<version>, create its
.venv-serve with requirements-serve.txt, pillow and psutil, and apply the alias and strata-think-echo as
setup-strata-runtime.ps1 does). Both arms start from the production launcher's own config (its -DryRun, so model,
profile, KV, cache, context, vision, draft settings and reserve match what Asuna uses); the candidate changes only
the runtime folder. Both must carry the workspace's local server patches, so they differ only in the release.

Probes, in order (scripts/ in this workspace): probe-qwen38-strata-draft-lang.py (3 zh + 3 en replies x2),
benchmark-qwen38-strata.py 4K in / 4K out and 128K in / 4K out (prompt read, decode, retrieval),
probe-qwen38-strata-cache.py (two agents alternating 32K histories with tool calls), probe-qwen38-strata.py
(smoke), probe-qwen38-strata-think-tags.py, probe-qwen38-strata-quality.py (long-thinking loops, 12 exact answers).
An arm that fails to load or a probe that fails is recorded and the run goes on. Nothing may run on the GPU
meanwhile: the run refuses to start while a Strata process exists.
"""

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = WORKSPACE / "scripts"
LAUNCHER = SCRIPTS / "start-qwen38-flash-next-uncensored-strata-vision.ps1"
PRODUCTION = "strata-nvfp4"
PORT = 1921
BASE = f"http://127.0.0.1:{PORT}"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def production_config() -> dict:
    out = subprocess.run(["pwsh", "-NoProfile", "-File", str(LAUNCHER), "-Port", str(PORT), "-BindAddress", "127.0.0.1",
                          "-DryRun"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out[out.index("{"):out.rindex("}") + 1])


def arm_config(template: dict, runtime: str, engine_log: Path) -> dict:
    cfg = json.loads(json.dumps(template))
    rt = WORKSPACE / "runtime" / runtime
    cfg["exe"], cfg["cwd"], cfg["log"] = str(rt / "engine" / "strata.exe"), str(rt), str(engine_log)
    cfg["vision"]["exe"] = str(rt / "engine" / "strata-vision.exe")
    a = cfg["args"]
    a[a.index("--expert-profile") + 1] = str(rt / "data" / "expert-profile.bin")
    return cfg


def check_runtime(runtime: str) -> list:
    rt = WORKSPACE / "runtime" / runtime
    problems = [f"{runtime}: missing {need}" for need in
                ("engine/strata.exe", "engine/strata-vision.exe", ".venv-serve/Scripts/python.exe", "data/expert-profile.bin")
                if not (rt / need).exists()]
    read = lambda p: p.read_text(encoding="utf-8") if p.exists() else ""   # noqa: E731
    if 'req.get("thinking_token_budget")' not in read(rt / "serve" / "server.py"):
        problems.append(f"{runtime}: no thinking_token_budget alias")
    if "# local: strata-think-echo v4" not in read(rt / "serve" / "frontend.py"):
        problems.append(f"{runtime}: no strata-think-echo v4")
    return problems


def strata_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True, text=True).stdout
    return "strata.exe" in out


def port_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def wait_ready(proc, limit_s=900) -> bool:
    end = time.time() + limit_s
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            if any(m.get("status", {}).get("value") == "loaded" for m in get("/v1/models")["data"]):
                return True
        except Exception:  # noqa: BLE001 - not listening yet
            pass
        time.sleep(2)
    return False


def stop(proc) -> None:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    end = time.time() + 120
    while time.time() < end and (strata_running() or not port_free()):
        time.sleep(1)


def run_probe(name, cmd, out_dir: Path, timeout_s) -> dict:
    t = time.time()
    with (out_dir / f"{name}.log").open("w", encoding="utf-8") as f:
        try:
            rc = subprocess.run(cmd, cwd=WORKSPACE, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    log(f"  {name}: exit {rc} in {time.time() - t:.0f} s")
    return {"exit": rc, "seconds": round(time.time() - t, 1)}


def probes(out: Path, engine_log: Path) -> dict:
    py, s = sys.executable, lambda name: str(SCRIPTS / name)   # noqa: E731
    return {
        "lang": run_probe("lang", [py, s("probe-qwen38-strata-draft-lang.py"), "--base-url", BASE + "/v1",
                                   "--output", str(out / "lang.json"), "--repeats", "2"], out, 1200),
        "bench4k": run_probe("bench4k", [py, s("benchmark-qwen38-strata.py"), "--base-url", BASE,
                                         "--output", str(out / "bench4k"), "--prompts", "4096", "--repeats", "1"], out, 900),
        "bench128k": run_probe("bench128k", [py, s("benchmark-qwen38-strata.py"), "--base-url", BASE,
                                             "--output", str(out / "bench128k"), "--prompts", "131072", "--repeats", "1"],
                               out, 1500),
        "cache32k": run_probe("cache32k", [py, s("probe-qwen38-strata-cache.py"), "--base-url", BASE,
                                           "--output", str(out / "cache32k"), "--tokens", "32768", "--rounds", "3",
                                           "--engine-log", str(engine_log)], out, 1800),
        "smoke": run_probe("smoke", [py, s("probe-qwen38-strata.py"), "--base-url", BASE,
                                     "--output", str(out / "smoke.json")], out, 900),
        "think_tags": run_probe("think_tags", [py, s("probe-qwen38-strata-think-tags.py"), "--base-url", BASE,
                                               "--output", str(out / "think-tags.json")], out, 1200),
        "quality": run_probe("quality", [py, s("probe-qwen38-strata-quality.py"), "--base-url", BASE,
                                         "--output", str(out / "quality.json")], out, 3600),
    }


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a probe that failed leaves nothing to read
        return None


def summarize(out: Path) -> dict:
    s = {}
    lang = load(out / "lang.json")
    if lang:
        for k in ("zh", "en"):
            s[f"{k}_decode_tok_s"] = lang["summary"][k]["decode_tok_s"]
    for case in ("bench4k", "bench128k"):
        for f in glob.glob(str(out / case / "*.result.json")):
            r = load(f)
            if r and r.get("timings"):
                s[f"{case}_prefill_tok_s"] = r["timings"]["prompt_per_second"]
                s[f"{case}_decode_tok_s"] = r["timings"]["predicted_per_second"]
                s[f"{case}_retrieval"] = r.get("codes_correct")
    cache = load(out / "cache32k" / "report.json")
    if cache:
        sw = [c for c in cache["cases"] if "switch" in c["label"]]
        s["cache_passed"] = cache.get("passed")
        s["cache_switch_wall_s_mean"] = round(sum(c["wall_s"] for c in sw) / len(sw), 2) if sw else None
    smoke = load(out / "smoke.json")
    if smoke:
        s["smoke_passed"] = smoke.get("passed")
    tt = load(out / "think-tags.json")
    if tt:
        s["think_tags"] = f"{tt['passed']}/{tt['total']}"
    q = load(out / "quality.json")
    if q:
        s["long_thinking"] = f"{q['long_thinking']['passed']}/{q['long_thinking']['total']}"
        s["exact_answers"] = f"{q['qa']['correct']}/{q['qa']['total']}"
    server_log = out / "server.log"
    if server_log.exists():
        m = re.findall(r"filling the GPU's expert cache \((\d+) experts", server_log.read_text(encoding="utf-8", errors="replace"))
        s["gpu_expert_slots"] = int(m[-1]) if m else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="the staged runtime folder under runtime\\, e.g. strata-nvfp4-0.1.42")
    ap.add_argument("--output", type=Path, default=WORKSPACE / "benchmarks/raw/qwen38-strata" / time.strftime("%Y-%m-%d") / "ab")
    ap.add_argument("--dry-run", action="store_true", help="write both configs and check them, start nothing")
    args = ap.parse_args()
    args.output = args.output.resolve()      # the server runs from its runtime folder: every path it gets is absolute
    args.output.mkdir(parents=True, exist_ok=True)
    arms = {"production": PRODUCTION, "candidate": args.candidate}
    template = production_config()
    problems = sorted({p for rt in arms.values() for p in check_runtime(rt)})
    if problems:
        raise SystemExit("not ready:\n  " + "\n  ".join(problems))
    if not args.dry_run and (strata_running() or not port_free()):
        raise SystemExit(f"a Strata process or port {PORT} is in use: stop every Strata server first")
    results = {}
    for arm, runtime in arms.items():
        out = args.output / arm
        out.mkdir(parents=True, exist_ok=True)
        engine_log = out / "engine.log"
        cfg_path = out / "config.json"
        cfg_path.write_text(json.dumps(arm_config(template, runtime, engine_log), indent=1), encoding="utf-8")
        log(f"{arm}: runtime\\{runtime}")
        if args.dry_run:
            continue
        python = WORKSPACE / "runtime" / runtime / ".venv-serve" / "Scripts" / "python.exe"
        t = time.time()
        with (out / "server.log").open("w", encoding="utf-8") as server_log:
            proc = subprocess.Popen([str(python), "-u", "-m", "serve.server", "--engine", "strata", "--config",
                                     str(cfg_path), "--port", str(PORT)], cwd=WORKSPACE / "runtime" / runtime,
                                    stdout=server_log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            try:
                if not wait_ready(proc):
                    results[arm] = {"error": "the server did not load (see server.log, engine.log)"}
                    log(f"{arm}: did not load")
                    continue
                status = get("/v1/status")
                log(f"{arm}: ready in {time.time() - t:.0f} s, engine {status.get('engine')}")
                results[arm] = {"engine": status.get("engine"), "probes": probes(out, engine_log)}
            finally:
                stop(proc)
        results.setdefault(arm, {}).update(summarize(out))
        (args.output / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    keys = ["engine", "gpu_expert_slots", "zh_decode_tok_s", "en_decode_tok_s", "bench4k_prefill_tok_s",
            "bench4k_decode_tok_s", "bench128k_prefill_tok_s", "bench128k_decode_tok_s", "bench128k_retrieval",
            "cache_passed", "cache_switch_wall_s_mean", "smoke_passed", "think_tags", "long_thinking", "exact_answers",
            "error"]
    lines = ["| | production | candidate |", "|---|---:|---:|"]
    lines += [f"| {k} | " + " | ".join(str(results.get(a, {}).get(k, "")) for a in arms) + " |" for k in keys]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
