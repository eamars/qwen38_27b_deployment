"""Reconcile real checkpoint tensor headers with the experimental model on meta."""
import json
import pathlib
import re
import urllib.request
import torch
from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.models.deepseek_v4.gguf import parse_gguf_config, _LAYER_MAP, _GLOBAL_MAP, _EXPERT_SUFFIXES
from freetoken.models.deepseek_v4.model import DeepseekV4ForCausalLM
from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes,dsv4_pool_bytes
from freetoken.distributed.info import set_tp_info

root = pathlib.Path('/home/rba90/deepseek-freetoken-probe/metadata-only')
root.mkdir(exist_ok=True)
inventory = json.loads(pathlib.Path('/home/rba90/deepseek-freetoken-probe/header-inventory.json').read_text())[0]
for f in inventory['files']:
    p = root/pathlib.Path(f['path']).name
    if p.exists() and p.stat().st_size == f['size']:
        continue
    req = urllib.request.Request(f'https://huggingface.co/{inventory["repo"]}/resolve/{inventory["revision"]}/{f["path"]}',headers={'Range':'bytes=0-8388607'})
    with urllib.request.urlopen(req) as r,p.open('wb') as out:
        out.write(r.read(8388608))
        out.truncate(f['size'])
path = str(sorted(root.glob('*.gguf'))[0])
config = parse_gguf_config(build_gguf_shim(path))
set_tp_info(0,1)
with torch.device('meta'):
    model = DeepseekV4ForCausalLM(config)
params = model.state_dict()
mapped = {}
for tensor in iter_gguf_tensors(path):
    m = re.match(r'^blk\.(\d+)\.(.+)$',tensor.name)
    if m:
        layer,suffix = int(m[1]),m[2]
        if suffix in _EXPERT_SUFFIXES:
            continue
        if suffix == 'ffn_gate_tid2eid.weight':
            dest,kind = 'ffn.gate.tid2eid','i64'
        else:
            dest,kind = _LAYER_MAP[suffix]
        dest = f'layers.{layer}.{dest}'
    elif tensor.name == 'token_embd.weight':
        dest,kind = 'embed.weight','packed'
    else:
        dest,kind = _GLOBAL_MAP[tensor.name]
    shape = (tensor.rows,tensor.row_bytes) if kind == 'packed' else tensor.shape
    assert tuple(shape) == tuple(params[dest].shape),(tensor.name,dest,shape,params[dest].shape)
    mapped[dest] = tensor.name
assert set(mapped)==set(params),sorted(set(params)-set(mapped))
result = {'matched_parameters':len(mapped),'layers':config.num_layers,
          'dense_parameter_gib':sum(p.numel()*p.element_size() for p in params.values())/2**30,
          'kv_gib_at_262144':{str(r):dsv4_pool_bytes(dsv4_pool_sizes(2048,config.dsv4_args,r),config.dsv4_args,2)/2**30 for r in [.2,.02,.005]}}
print(json.dumps(result),flush=True)
pathlib.Path('/home/rba90/deepseek-freetoken-probe/loader-metadata-probe.json').write_text(json.dumps(result,indent=2))
