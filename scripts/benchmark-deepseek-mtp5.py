"""One warmup, one paired benchmark; no tuning after a slower result."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

root=Path('/home/rba90/deepseek-freetoken-probe')
prompt='Write a Python function that merges overlapping intervals. Explain its time complexity and give three test cases.'
payload=dict(model='deepseek-v4-flash-gpu-probe',messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=64,stream=False)
request=urllib.request.Request('http://127.0.0.1:1920/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
print('Warming new draft kernels (not timed).',flush=True)
with urllib.request.urlopen(request,timeout=600) as response:
    warmup=json.load(response)
(root/'mtp5-warmup.json').write_text(json.dumps(warmup,indent=2))
assert warmup.get('choices') and warmup['choices'][0]['message'].get('content'),warmup
print('Running the paired 256-token benchmark.',flush=True)
subprocess.run([sys.executable,'/mnt/c/workspace/qwen38_27b/scripts/benchmark-deepseek-freetoken.py','--name','spec-mtp5-paired','--tokens','256'],check=True)
baseline=json.loads((root/'corrected-mtp5-paired-baseline.json').read_text())
mtp=json.loads((root/'spec-mtp5-paired.json').read_text())
result=dict(baseline_tps=baseline['client_decode_tps'],mtp5_tps=mtp['client_decode_tps'])
result['slower']=result['mtp5_tps']<result['baseline_tps']
result['change_percent']=(result['mtp5_tps']/result['baseline_tps']-1)*100
(root/'mtp5-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
if result['slower']:
    # Kill only this experiment's server, whose command line carries both flags.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try: cmd=(process/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if b'freetoken.cli' in cmd and b'--speculative-dspark' in cmd and b'deepseek-v4-flash-gpu-probe' in cmd:
            import os,signal
            os.kill(int(process.name),signal.SIGTERM)
    print('MTP5 is slower: stopped. No further tuning or benchmarks.',flush=True)
