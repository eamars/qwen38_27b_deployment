"""Measure a cold text prompt on the resident PR447 server, without unloading it."""
import argparse
import json
from pathlib import Path
import re
import threading
import time

import httpx
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input-tokens', type=int, default=131072)
parser.add_argument('--output-tokens', type=int, default=1024)
args = parser.parse_args()
root = Path('/mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38_flash_next/pr447-speed-2026-09-30')
out = root / f'resident-{args.input_tokens}-{time.strftime("%H%M%S")}'
out.mkdir()
base = 'http://127.0.0.1:19447'
client = httpx.Client(timeout=10, trust_env=False)
health = client.get(base + '/health').json()
assert health['status'] == 'ok', health
before = client.get(base + '/v1/stats').json()
assert before['requests']['active'] == 0, before
assert before['limits']['max_seq_len'] >= args.input_tokens + args.output_tokens
(out / 'health-before.json').write_text(json.dumps(health, indent=2))
(out / 'stats-before.json').write_text(json.dumps(before, indent=2))
tokenizer = Tokenizer.from_file('/home/rba90/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json')
prefix = ('Zircon long-context cold-prefill experiment. Read this technical archive, then '
          'continue it with detailed engineering recommendations.\n')
paragraph = ('Distributed storage uses replication, consistency checks, cache admission, '
             'transaction logs, background repair, observability and recovery procedures. '
             'Explain throughput, latency, data integrity, and operational tradeoffs.\n')
text = prefix + ''.join(f'Archive record {i}: {paragraph}' for i in range(args.input_tokens // 20 + 1))
ids = tokenizer.encode(text, add_special_tokens=False).ids[:args.input_tokens]
prompt = tokenizer.decode(ids, skip_special_tokens=False)
actual = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
assert actual == args.input_tokens, (actual, args.input_tokens)
body = {'model': health['model'], 'prompt': prompt, 'max_tokens': args.output_tokens,
        'temperature': 0, 'ignore_eos': True, 'stream': True,
        'stream_options': {'include_usage': True}}
(out / 'request.json').write_text(json.dumps(body))
result = {'input_tokens': actual, 'requested_output_tokens': args.output_tokens,
          'instance_id': health['instance_id'], 'server_pid': int((root/'server.pid').read_text()),
          'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'decode_windows': [],
          'prefill_chunks': [], 'model_kept_resident': True, 'threshold_tps': 40}
done = threading.Event()
started = time.perf_counter()
first = last = None
log_path = root / 'server-128k-continuation.log'
if not log_path.exists():
    log_path = root / 'server.log'
log_offset = log_path.stat().st_size


def monitor():
    first_decode = True
    with log_path.open() as source, (out / 'server-request.log').open('w', buffering=1) as log:
        source.seek(log_offset)
        while not done.is_set():
            line = source.readline()
            if not line:
                time.sleep(0.02)
                continue
            log.write(line)
            if 'Prefill batch,' in line:
                first_decode = True
                result['prefill_chunks'].append({'elapsed_s': time.perf_counter()-started, 'log': line.strip()})
                print(line.strip(), flush=True)
            match = re.search(r'gen throughput \(token/s\): ([0-9.]+)', line)
            if match:
                rate = float(match.group(1))
                result['decode_windows'].append({'tps': rate, 'clean': not first_decode, 'log': line.strip()})
                print(line.strip(), flush=True)
                first_decode = False


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
print(f'REQUEST: {actual} cold input / {args.output_tokens} output; artifacts {out}', flush=True)
try:
    with httpx.Client(timeout=httpx.Timeout(1800, connect=10), trust_env=False) as stream_client:
        with stream_client.stream('POST', base + '/v1/completions', json=body) as response:
            if response.is_error:
                (out/'http-error.txt').write_bytes(response.read())
            response.raise_for_status()
            with (out / 'stream.jsonl').open('w', buffering=1) as stream:
                for line in response.iter_lines():
                    if not line.startswith('data: ') or line == 'data: [DONE]':
                        continue
                    elapsed = time.perf_counter()-started
                    data = json.loads(line[6:])
                    stream.write(json.dumps({'elapsed_s': elapsed, 'data': data})+'\n')
                    if any(c.get('text') for c in data.get('choices', [])):
                        if first is None:
                            first = elapsed
                            print(f'FIRST TOKEN: {first:.3f}s; effective prefill {actual/first:.2f} token/s', flush=True)
                        last = elapsed
                    if data.get('usage'):
                        result['usage'] = data['usage']
                    if data.get('error'):
                        raise RuntimeError(data['error'])
    result['status'] = 'completed'
except Exception as exc:
    result['exception'] = repr(exc)
    result['status'] = 'request_error'
finally:
    result.update(total_s=time.perf_counter()-started, ttft_s=first, last_token_s=last)
    if first is not None:
        result['effective_prefill_tps_input_over_ttft'] = actual/first
    for _ in range(50):
        after = client.get(base + '/v1/stats').json()
        if after['requests']['active'] == 0:
            break
        time.sleep(0.2)
    (out/'stats-after.json').write_text(json.dumps(after, indent=2))
    result['actual_prompt_tokens'] = after['requests']['prompt_tokens_total']-before['requests']['prompt_tokens_total']
    result['actual_completion_tokens'] = after['requests']['completion_tokens_total']-before['requests']['completion_tokens_total']
    result['cached_tokens'] = after['prefix_cache']['cached_tokens_total']-before['prefix_cache']['cached_tokens_total']
    if result['status'] == 'completed' and first is not None and last > first:
        result['decode_tps_average'] = (result['actual_completion_tokens']-1)/(last-first)
        if result['decode_tps_average'] < 40:
            result['status'] = 'completed_below_40'
            result['stop_reason'] = 'completed request average below 40'
    final_health = client.get(base+'/health').json()
    result['health_after'] = final_health
    assert final_health['instance_id'] == health['instance_id'], 'server instance changed'
    done.set()
    thread.join(timeout=2)
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['prefill_chunks','decode_windows']},indent=2), flush=True)
    print('RESULT:',out/'result.json',flush=True)
