"""Measure streamed completions at the client and retain the returned text."""
import argparse
import json
import pathlib
import time
import urllib.request
import subprocess
import threading
import signal

p = argparse.ArgumentParser()
p.add_argument('--prompt',default='Write a Python function that merges overlapping intervals. Explain its time complexity and give three test cases.')
p.add_argument('--tokens',type=int,default=256)
p.add_argument('--name',default='short-baseline')
p.add_argument('--filler-lines', type=int, default=0)
p.add_argument('--reference', default='ORCHID-7319')
p.add_argument('--copy-test', action='store_true')
p.add_argument('--long-output', action='store_true')
p.add_argument('--reasoning-effort', '--thinking-effort', dest='reasoning_effort',
                choices=('none', 'off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'),
                default=None,
                help='Per-request DeepSeek thinking effort; omit to preserve the model default.')
a=p.parse_args()
samples=[]
stop_sampling=threading.Event()
def sample_resources():
    while not stop_sampling.is_set():
        mem={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())}
        vm=dict(line.split() for line in pathlib.Path('/proc/vmstat').read_text().splitlines())
        gpu=subprocess.run(['nvidia-smi','-i','GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0',
                            '--query-gpu=memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
        samples.append({'time':time.time(),'mem_available_kib':mem['MemAvailable'],
                        'swap_used_kib':mem['SwapTotal']-mem['SwapFree'],
                        'pswpin':int(vm['pswpin']),'pswpout':int(vm['pswpout']),'gpu':gpu})
        stop_sampling.wait(2)
sampler=threading.Thread(target=sample_resources,daemon=True)
sampler.start()
if a.filler_lines:
    a.prompt = (f'Remember this reference code: {a.reference}.\n' +
                'The archive contains ordinary records for review.\n' * a.filler_lines +
                '\nWhat was the reference code at the beginning? Reply with only the code.')
    if a.long_output:
        a.prompt=a.prompt.rsplit('\n',1)[0]+'\nFirst state the reference code from the beginning, then explain in detail how to validate an archive of records. Give a numbered checklist with at least twelve items.'
copy_expected=None
if a.copy_test:
    words='alpha cedar maple river orchid stone silver forest amber cloud'
    copy_expected='\n'.join(f'Record {i}: {words}.' for i in range(1,17))
    a.prompt='Copy the following records exactly. Output only the records, without code fences or commentary.\n\n'+copy_expected
payload={'model':'deepseek-v4-flash-gpu-probe','messages':[{'role':'user','content':a.prompt}],
         'temperature':0,'max_tokens':a.tokens,'stream':True,'stream_options':{'include_usage':True}}
if a.reasoning_effort is not None:
    payload['reasoning_effort'] = a.reasoning_effort
req=urllib.request.Request('http://127.0.0.1:1919/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
start=time.perf_counter()
def interrupted(signum, frame):
    stop_sampling.set()
    result={'name':a.name,'payload':payload,'wall_s':time.perf_counter()-start,
            'error':f'benchmark interrupted with signal {signum}',
            'client_decode_tps':None,'resources':samples}
    (pathlib.Path('/home/rba90/deepseek-freetoken-probe')/(a.name+'.json')).write_text(json.dumps(result,indent=2))
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
first=None
last=None
chunks=[]
usage=None
events=[]
with urllib.request.urlopen(req,timeout=7200) as response:
    for line in response:
        if not line.startswith(b'data: '):
            continue
        data=line[6:].strip()
        if data==b'[DONE]':
            break
        event=json.loads(data)
        events.append(event)
        if event.get('usage'):
            usage=event['usage']
        for choice in event.get('choices',[]):
            delta=choice.get('delta',{})
            text=delta.get('content') or delta.get('reasoning_content') or ''
            if text:
                now=time.perf_counter()
                if first is None:
                    first=now
                last=now
                chunks.append(text)
elapsed=time.perf_counter()-start
stop_sampling.set()
sampler.join(timeout=5)
count=(usage or {}).get('completion_tokens')
result={'name':a.name,'payload':payload,'wall_s':elapsed,'ttft_s':first-start if first else None,
        'usage':usage,'client_decode_tps':(count-1)/(last-first) if count and last and last>first else None,
        'text':''.join(chunks),'events':events,'resources':samples}
if copy_expected is not None:
    result['copy_exact_match']=result['text'].strip()==copy_expected
path=pathlib.Path('/home/rba90/deepseek-freetoken-probe')/(a.name+'.json')
path.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('events','payload','resources')}),flush=True)
