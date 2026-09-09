"""Probe the real checkpoint's MTP5 compatibility before allocating another model."""
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request
from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.deepseek_v4.gguf import parse_gguf_config
from freetoken.engine.engine import _adjust_dsv4_config

path = '/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS/IQ3_XXS/DeepSeek-V4-Flash-0731-IQ3_XXS-00001-of-00004.gguf'
model = parse_gguf_config(build_gguf_shim(path))
args = model.dsv4_args
result = dict(requested_draft_tokens=5, n_mtp_layers=args.n_mtp_layers,
              dspark_block_size=args.dspark_block_size, has_dspark=args.has_dspark,
              benchmark_completed=False)
config = SimpleNamespace(tp_info=SimpleNamespace(size=1), model_config=model,
                         max_seq_len=180000, max_running_req=1, speculative_dspark=True)
try:
    _adjust_dsv4_config(config, None)
except ValueError as exc:
    result['startup_error'] = str(exc)
with urllib.request.urlopen('http://127.0.0.1:1920/v1/cache/status') as response:
    status = response.read().decode()
Path('/home/rba90/deepseek-freetoken-probe/final-cache-status.json').write_text(status)
Path('/home/rba90/deepseek-freetoken-probe/mtp5-compatibility.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
