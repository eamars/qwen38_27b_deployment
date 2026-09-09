import os
os.environ['FREETOKEN_MTP_SIDECAR']='/home/rba90/deepseek-freetoken-probe/mtp5'
os.environ['FREETOKEN_DSPARK_LAYER_BASE']='0'
import json
from pathlib import Path
import torch
from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.deepseek_v4.gguf import parse_gguf_config
from freetoken.models.deepseek_v4.args import set_dspark_enabled
from freetoken.models.deepseek_v4.dspark import DSparkDrafter
from freetoken.models.deepseek_v4.mtp_sidecar import weights
from freetoken.distributed.info import set_tp_info
set_tp_info(0,1)
path='/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS/IQ3_XXS/DeepSeek-V4-Flash-0731-IQ3_XXS-00001-of-00004.gguf'
config=parse_gguf_config(build_gguf_shim(path)); args=config.dsv4_args
args.dspark_enabled=True; set_dspark_enabled(True)
with torch.device('meta'): draft=DSparkDrafter(args)
expected=draft.state_dict(); seen=set(); nbytes=0
for name,tensor in weights(args):
    name=name.removeprefix('drafter.')
    assert expected[name].shape==tensor.shape,(name,expected[name].shape,tensor.shape)
    seen.add(name); nbytes+=tensor.numel()*tensor.element_size()
assert seen==set(expected),set(expected)-seen
result=dict(matched_draft_parameters=len(seen),draft_weight_gib=nbytes/2**30,block_size=args.dspark_block_size)
Path('/home/rba90/deepseek-freetoken-probe/mtp5-weight-probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
