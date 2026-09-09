"""Real CUDA IQ3 execution and pinned-host transfer probe; not a model benchmark."""
import json
import time
import pathlib
import numpy as np
import torch
import gguf
from freetoken.kernel.gguf import ggml_moe_a8_vec
from freetoken.moe.fused_q4_0 import fused_experts_gguf

torch.set_num_threads(8)
torch.manual_seed(71)
rng = np.random.default_rng(71)
device = torch.cuda.get_device_name()
# Exercise valid IQ3_XXS blocks and compare against separately dequantized weights.
e, m, n = 2, 64, 256
raw = rng.integers(0, 256, (e, m, 98), dtype=np.uint8)
raw[..., :2] = np.array([0.02], np.float16).view(np.uint8)
weights = torch.from_numpy(raw).cuda()
x = torch.randn(1, n, device='cuda', dtype=torch.bfloat16)
ids = torch.tensor([[0, 1]], device='cuda', dtype=torch.int32)
got = ggml_moe_a8_vec(x, weights, ids, 2, 18, m, 1)
dense = gguf.dequantize(raw.reshape(e*m, 98), gguf.GGMLQuantizationType.IQ3_XXS)
dense = torch.from_numpy(dense.reshape(e, m, n)).cuda().to(torch.bfloat16)
ref = torch.stack([x[0].float() @ dense[k].float().T for k in range(e)])
relative_rmse = ((got.float().reshape(e,m)-ref).square().mean().sqrt()/ref.square().mean().sqrt()).item()
assert torch.isfinite(got).all() and relative_rmse < .03, relative_rmse

def make_bank(rows):
    b = rng.integers(0, 256, (2, rows, 98), dtype=np.uint8)
    b[..., :2] = np.array([0.1], np.float16).view(np.uint8)
    d = gguf.dequantize(b.reshape(2*rows,98), gguf.GGMLQuantizationType.IQ3_XXS)
    return torch.from_numpy(b).cuda(), torch.from_numpy(d.reshape(2,rows,256)).cuda().to(torch.bfloat16)

gu, gu_dense = make_bank(512)
down, down_dense = make_bank(256)
large_x = x * 10
route_weights = torch.tensor([[.4,.6]],device='cuda')
clamped = fused_experts_gguf(large_x,gu,down,route_weights,ids,'silu',18,swiglu_limit=10)
reference = []
for expert in range(2):
    projections = (large_x.float() @ gu_dense[expert].float().T).to(torch.bfloat16).float()
    gate, up = projections.chunk(2,dim=-1)
    inter = (torch.nn.functional.silu(gate.clamp(max=10))*up.clamp(-10,10)).to(torch.bfloat16)
    reference.append((inter.float()@down_dense[expert].float().T).to(torch.bfloat16))
clamped_ref = (torch.stack(reference,dim=1)*route_weights[:,:,None].to(torch.bfloat16)).sum(dim=1)
clamp_rmse = ((clamped.float()-clamped_ref.float()).square().mean().sqrt()/clamped_ref.float().square().mean().sqrt()).item()
actual_gu = ggml_moe_a8_vec(large_x, gu, ids, 2, 18, 512, 1)
g, u = actual_gu.float().chunk(2,dim=-1)
inter_ref = (torch.nn.functional.silu(g.clamp(max=10))*u.clamp(-10,10)).to(torch.bfloat16)
down_ref = ggml_moe_a8_vec(inter_ref, down, ids, 1, 18, 256, 2)
staged_ref = (down_ref.reshape(1,2,256)*route_weights[:,:,None].to(torch.bfloat16)).sum(dim=1)
print(json.dumps({'clamp_rmse':clamp_rmse,'staged_max_error':(clamped-staged_ref).abs().max().item(),
                  'gu_reference_rmse': ((actual_gu.float().reshape(2,512)-torch.stack([(large_x.float()@w.float().T)[0] for w in gu_dense])).square().mean().sqrt()/actual_gu.float().square().mean().sqrt()).item()}),flush=True)
assert torch.equal(clamped, staged_ref)

size = 256 * 1024**2
host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
host.fill_(71)
gpu = torch.empty_like(host, device='cuda')
for _ in range(3):
    gpu.copy_(host, non_blocking=True)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(32):
    gpu.copy_(host, non_blocking=True)
torch.cuda.synchronize()
seconds = time.perf_counter()-t
result = {'device': device, 'iq3_cuda_relative_rmse': relative_rmse,
          'clamped_moe_relative_rmse': clamp_rmse,
          'h2d_gb_per_second': size*32/seconds/1e9, 'h2d_elapsed_s': seconds}
print(json.dumps(result), flush=True)
pathlib.Path('/home/rba90/deepseek-freetoken-probe/gpu-kernel-probe.json').write_text(json.dumps(result,indent=2))
