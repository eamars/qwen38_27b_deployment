"""Exercise CUDA LRU admission and gather from real-size pinned expert banks."""
import json
import pathlib
import time
import torch
from freetoken.moe.host_banks import HostBank
from freetoken.moe.offload_cache import OffloadMoeCache

torch.set_num_threads(8)
held=[]
sources={}
for name,shape in [('gate_up',(256,4096,1568)),('down',(256,4096,784))]:
    b=HostBank(shape,torch.uint8)
    for expert in range(256):
        b.tensor[expert].fill_(expert)
    b.pin()
    held.append(b)
    sources[name]=[b.tensor,b.tensor]
cache=OffloadMoeCache(num_layers=2,num_experts=256,cache_size=256,device=torch.device('cuda'),quant_format='gguf',gguf_expert_types=((18,18),(18,18)))
cache.set_bank_sources(sources)
ids=torch.tensor([[1,17,33,65,129,255]],device='cuda',dtype=torch.int32)
cache.ensure_experts(0,ids)
cache.copy_missing()
torch.cuda.synchronize()
for views in cache.bank_views():
    selected=views.index_select(0,ids.long().reshape(-1))
    expected=torch.tensor([1,17,33,65,129,255],device='cuda',dtype=torch.uint8)
    assert torch.equal(selected[:,0,0],expected)
    assert torch.equal(selected[:,-1,-1],expected)
route=torch.arange(6,device='cuda',dtype=torch.int32).reshape(1,6)
for _ in range(3):
    ids.copy_(route)
    cache.ensure_experts(0,ids)
    cache.copy_missing()
torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for layer in range(2):
        ids.copy_(route)
        cache.ensure_experts(layer,ids)
        cache.copy_missing()
torch.cuda.synchronize()
start=time.perf_counter()
for _ in range(100):
    route.add_(6).remainder_(256)
    graph.replay()
torch.cuda.synchronize()
elapsed=time.perf_counter()-start
result={'byte_check_passed':True,'cuda_graph_passed':True,'cpu_executor':cache.cpu_executor is not None,
        'gather_gb_per_s':12*(4096*1568+4096*784)*100/elapsed/1e9,'two_layer_step_ms':elapsed*10}
print(json.dumps(result),flush=True)
pathlib.Path('/home/rba90/deepseek-freetoken-probe/offload-copy-probe.json').write_text(json.dumps(result,indent=2))
