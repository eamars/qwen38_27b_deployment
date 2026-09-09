"""Export compact benchmark evidence and the exact experimental runtime patches."""
import json
import pathlib
import shutil
import subprocess

source=pathlib.Path('/home/rba90/deepseek-freetoken-probe')
root=pathlib.Path('/mnt/c/workspace/qwen38_27b')
target=root/'artifacts/deepseek-freetoken'
target.mkdir(parents=True,exist_ok=True)
benchmarks=[]
for path in sorted(source.glob('*.json')):
    data=json.loads(path.read_text())
    if not isinstance(data,dict) or 'client_decode_tps' not in data:
        continue
    result={k:v for k,v in data.items() if k not in ('events','payload','resources')}
    result['pre_tokenizer_correction']=not data['name'].startswith(('corrected-','spec-'))
    resources=data.get('resources',[])
    if resources:
        result['resources']={
            'samples':len(resources),
            'min_available_ram_gib':min(s['mem_available_kib'] for s in resources)/1048576,
            'max_gpu_used_mib':max(int(s['gpu'].split(',')[0]) for s in resources if s['gpu']),
            'swap_in_pages':resources[-1]['pswpin']-resources[0]['pswpin'],
            'swap_out_pages':resources[-1]['pswpout']-resources[0]['pswpout'],
        }
    benchmarks.append(result)
(target/'benchmarks.json').write_text(json.dumps(benchmarks,indent=2))
for name in ('download-verified.json','gpu-kernel-probe.json','host-bank-probe.json',
             'offload-copy-probe.json','loader-metadata-probe.json',
             'final-cache-status.json','final-rebuild.json','mtp5-comparison.json',
             'mtp5-weight-probe.json','mtp5-quant-probe.json','mtp5-warmup.json'):
    if (source/name).exists():
        shutil.copyfile(source/name,target/name)
for path in source.glob('server-*.log'):
    shutil.copyfile(path,target/path.name)
for folder in ('freetoken-deepseek-experiment','freetoken-deepseek-spec'):
    patch=subprocess.run(['/mnt/c/Program Files/Git/cmd/git.exe','-C',
                          'C:/workspace/qwen38_27b/runtime/'+folder,'diff','--binary','HEAD'],
                         check=True,capture_output=True).stdout
    (target/(folder+'.patch')).write_bytes(patch)
print(json.dumps({'benchmarks':len(benchmarks),'export':str(target)}))
sidecar=root/'runtime/freetoken-deepseek-spec/python/freetoken/models/deepseek_v4/mtp_sidecar.py'
if sidecar.exists(): shutil.copyfile(sidecar,target/'mtp_sidecar.py')
