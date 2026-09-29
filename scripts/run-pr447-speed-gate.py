"""Run TP2+EP2 and judge completed-request average decode; keep the model resident."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import httpx
from tokenizers import Tokenizer

ROOT = Path('/mnt/c/workspace/qwen38_27b')
OUT = ROOT / 'benchmarks/raw/qwen38_flash_next/pr447-speed-2026-09-30'
MODEL = Path('/home/rba90/models/Qwen3.8-Flash-Next-NVFP4')
OUT.mkdir(parents=True, exist_ok=True)
# Expose library headers without shadowing CUDA 13.3's compiler/runtime headers.
venv = Path(sys.executable).parent.parent
include = venv / 'include/cublas-compat'
include.mkdir(parents=True, exist_ok=True)
cuda_headers = venv / 'lib/python3.12/site-packages/nvidia/cu13/include'
for pattern in ('cublas*.h', 'curand*.h'):
    for header in cuda_headers.glob(pattern):
        shutil.copy2(header, include / header.name)
env = os.environ.copy()
env.update(CUDA_HOME='/usr/local/cuda-13.3', TVM_FFI_CUDA_ARCH_LIST='8.9 12.0',
           PYTHONUNBUFFERED='1', NCCL_P2P_DISABLE='1', FREETOKEN_PIN_BUDGET_GB='64',
           FLASHINFER_USE_CUDA_NORM='1', CPATH=str(include),
           PATH='/home/rba90/.venvs/freetoken-pr447/bin:/usr/local/cuda-13.3/bin:' + env['PATH'],
           LD_LIBRARY_PATH='/usr/local/cuda-13.3/targets/x86_64-linux/lib:' + env.get('LD_LIBRARY_PATH', ''))
env.pop('CUDA_VISIBLE_DEVICES', None)
cmd = [str(Path(sys.executable).with_name('ft')), 'serve', '--model', str(MODEL),
       '--served-model-name', 'qwen38-next-pr447-speed', '--gpu', '1,0',
       '--host', '127.0.0.1', '--port', '19447', '--tp-size', '2', '--moe-ep-size', '2',
       '--max-running-requests', '1', '--dtype', 'bfloat16', '--memory-ratio', '0.90',
       '--moe-strategy', 'offload', '--moe-cache-auto',
       '--ple-backend', 'disk', '--max-seq-len-override', '262144',
       '--kv-reserve-tokens', '262144', '--num-tokens', '262144',
       '--max-prefill-length', '8192', '--max-output-tokens', '1024',
       '--cache-type', 'radix', '--enable-cache-report', '--decode-log-interval', '40',
       '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder']
result = {'command': cmd, 'model': str(MODEL), 'threshold_tps': 40,
          'gate': 'completed-request average decode only; individual windows never stop a request',
          'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'requests': [], 'decode_windows': []}
(OUT / 'launch.json').write_text(json.dumps({'command': cmd, 'environment': {k: env[k] for k in
    ['CUDA_HOME', 'TVM_FFI_CUDA_ARCH_LIST', 'NCCL_P2P_DISABLE', 'FREETOKEN_PIN_BUDGET_GB', 'FLASHINFER_USE_CUDA_NORM',
     'CPATH', 'PATH', 'LD_LIBRARY_PATH']}}, indent=2))
proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, start_new_session=True)
(OUT / 'server.pid').write_text(str(proc.pid))
stopped = threading.Event()
stop_lock = threading.Lock()


def stop(reason):
    with stop_lock:
        if stopped.is_set():
            return
        stopped.set()
        result['stop_reason'] = reason
        print('STOP:', reason, flush=True)
        if result.get('ready_at') and proc.poll() is None:
            # The server writes to a pipe; retain its reader when this controller exits.
            fd = proc.stdout.fileno()
            pump_code = (
                'import os,sys\n'
                'with os.fdopen(int(sys.argv[1]), "rb", buffering=0) as src, '
                'open(sys.argv[2], "ab", buffering=0) as dst:\n'
                ' while block := src.read(65536): dst.write(block)\n'
            )
            pump = subprocess.Popen(
                [sys.executable, '-c', pump_code, str(fd), str(OUT / 'server.log')],
                pass_fds=(fd,), start_new_session=True, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            (OUT / 'resident-log-pump.pid').write_text(str(pump.pid))
            result['model_kept_resident'] = True
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def monitor():
    first_decode = True
    with (OUT / 'server.log').open('w', buffering=1) as log:
        for line in proc.stdout:
            log.write(line)
            if 'Prefill batch,' in line:
                first_decode = True
            match = re.search(r'gen throughput \(token/s\): ([0-9.]+)', line)
            if match:
                tps = float(match.group(1))
                result['decode_windows'].append({'tps': tps, 'clean': not first_decode, 'log': line.strip()})
                first_decode = False
            if any(word in line for word in ['INFO ', 'WARNING ', 'ERROR ', 'Error:', 'Exception:', 'Traceback']):
                print(line.rstrip(), flush=True)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
base = 'http://127.0.0.1:19447'
try:
    with httpx.Client(timeout=5, trust_env=False) as client:
        deadline = time.monotonic() + 1800
        while not stopped.is_set() and proc.poll() is None:
            try:
                health = client.get(base + '/health').json()
                if health.get('status') == 'ok':
                    break
                if health.get('status') == 'error':
                    raise RuntimeError(str(health))
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('startup exceeded 30 minutes')
            time.sleep(1)
        else:
            raise RuntimeError('server exited before ready')
        result['ready_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        (OUT / 'stats-ready.json').write_text(client.get(base + '/v1/stats').text)

    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    for name, lead, count in [('warmup', 'Warmup engineering notes.', 128),
                               ('formal', 'Independent formal speed measurement.', 1024)]:
        if stopped.is_set():
            break
        prompt = (lead + '\n' + ('Explain the design of a reliable distributed storage system. '
            'Consider replication, consistency, caching, recovery, monitoring and throughput.\n' * 240))
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids[:4096]
        prompt_text = tokenizer.decode(ids, skip_special_tokens=False)
        body = {'model': 'qwen38-next-pr447-speed', 'prompt': prompt_text, 'max_tokens': count,
                'temperature': 0, 'ignore_eos': True, 'stream': True,
                'stream_options': {'include_usage': True}}
        (OUT / f'{name}-request.json').write_text(json.dumps(body))
        row = {'phase': name, 'input_tokens': len(ids), 'requested_output_tokens': count}
        result['requests'].append(row)
        print(f'REQUEST {name}: {len(ids)} input, {count} output', flush=True)
        started = time.perf_counter()
        first = last = None
        with httpx.Client(timeout=600, trust_env=False) as client:
            with client.stream('POST', base + '/v1/completions', json=body) as response:
                if response.is_error:
                    (OUT / f'{name}-http-error.txt').write_bytes(response.read())
                response.raise_for_status()
                with (OUT / f'{name}-stream.jsonl').open('w', buffering=1) as stream:
                    for line in response.iter_lines():
                        if not line.startswith('data: ') or line == 'data: [DONE]':
                            continue
                        elapsed = time.perf_counter() - started
                        data = json.loads(line[6:])
                        stream.write(json.dumps({'elapsed_s': elapsed, 'data': data}) + '\n')
                        if any(c.get('text') for c in data.get('choices', [])):
                            first = elapsed if first is None else first
                            last = elapsed
                        if data.get('usage'):
                            row['usage'] = data['usage']
                        if stopped.is_set():
                            break
        row.update(total_s=time.perf_counter() - started, ttft_s=first, last_token_s=last)
        if row.get('usage') and first is not None and last > first:
            row['decode_tps'] = (row['usage']['completion_tokens'] - 1) / (last - first)
            if row['decode_tps'] < 40:
                stop(f"{name} average {row['decode_tps']:.2f} token/s < 40")
        print(json.dumps(row), flush=True)
        if not stopped.is_set():
            with httpx.Client(timeout=5, trust_env=False) as client:
                (OUT / f'stats-{name}.json').write_text(client.get(base + '/v1/stats').text)
    if not stopped.is_set():
        result['status'] = 'completed'
except Exception as exc:
    result['exception'] = repr(exc)
    print('EXCEPTION:', repr(exc), flush=True)
finally:
    stop(result.get('stop_reason', result.get('status', 'startup_or_request_failure')))
    if not result.get('model_kept_resident'):
        proc.wait(timeout=10)
    thread.join(timeout=3)
    result['server_exit_code'] = proc.returncode
    result['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (OUT / 'result.json').write_text(json.dumps(result, indent=2))
    print('RESULT:', OUT / 'result.json', flush=True)
