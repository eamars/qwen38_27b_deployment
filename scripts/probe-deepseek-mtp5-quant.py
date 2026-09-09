import json
from pathlib import Path
import torch
from safetensors import safe_open
from freetoken.kernel.gguf import ggml_dequantize
root=Path('/home/rba90/deepseek-freetoken-probe/mtp5')
with safe_open(root/'model-00046-of-00048.safetensors',framework='pt') as f:
    w=f.get_tensor('mtp.0.ffn.experts.0.w1.weight').cuda()
    s=f.get_tensor('mtp.0.ffn.experts.0.w1.scale').view(torch.uint8).cuda().float()
lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
x=torch.stack([lut[(w&15).long()],lut[(w>>4).long()]],dim=-1).reshape(2048,4096)
x=(x.reshape(2048,-1,32)*torch.exp2(s-127).unsqueeze(-1)).reshape(2048,4096)
with (root/'gate_up-0.bin').open('rb') as f: data=bytearray(f.read(2048*1568))
q=torch.frombuffer(data,dtype=torch.uint8).reshape(2048,1568).cuda()
y=ggml_dequantize(q,18,2048,4096,torch.bfloat16).float()
rmse=float(torch.sqrt(torch.mean((x-y)**2)/torch.mean(x*x)))
assert torch.isfinite(y).all() and rmse<.25,rmse
result=dict(relative_weight_rmse=rmse,finite=True,source='official MTP layer 0 expert 0 w1',quant='IQ3_XXS')
Path('/home/rba90/deepseek-freetoken-probe/mtp5-quant-probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
