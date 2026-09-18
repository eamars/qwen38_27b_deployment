import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

out = Path(__file__).resolve().parent


def post(endpoint, body):
    req = urllib.request.Request(
        'http://127.0.0.1:8094/' + endpoint,
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


for _ in range(90):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8094/health', timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError('Server not ready')

prefix = post('tokenize', {'content': 'Read the following records. The secret key is ORCHID.\n', 'add_special': True})['tokens']
filler = post('tokenize', {'content': '\n'.join(
    f'Record {i}: The project includes testing, documentation, and review.' for i in range(1000))})['tokens']
ending = post('tokenize', {'content': '\nWhat is the secret key stated at the start? The secret key is'})['tokens']
n = 260096 - len(prefix) - len(ending)
tokens = prefix + (filler * (n // len(filler) + 1))[:n] + ending
stop = threading.Event()
samples = []


def monitor():
    while not stop.is_set():
        samples.append(subprocess.check_output([
            'nvidia-smi', '--id=GPU-eed52936-813f-8d68-1654-bfb56cb42bc3',
            '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        stop.wait(2)


thread = threading.Thread(target=monitor)
thread.start()
try:
    print('Starting 260096-token capacity probe', flush=True)
    result = post('completion', {'prompt': tokens, 'n_predict': 128, 'temperature': 0, 'cache_prompt': True})
    result.pop('prompt', None)  # The deterministic input is reconstructed above.
    (out / 'long-context.json').write_text(json.dumps(result, indent=2))
    assert result['tokens_evaluated'] == 260096 and not result['truncated']
    assert 'ORCHID' in result['content']
    assert result['timings']['draft_n_accepted'] > 0
    print(json.dumps({key: result.get(key) for key in (
        'content', 'timings', 'tokens_evaluated', 'tokens_predicted', 'truncated')}), flush=True)
finally:
    stop.set()
    thread.join()
    (out / 'vram-samples.json').write_text(json.dumps(samples, indent=2))
