"""Fetch immutable official MTP metadata and identify only the required draft shards."""
import json
from pathlib import Path
import urllib.request

root = Path('/home/rba90/deepseek-freetoken-probe/mtp5')
root.mkdir(parents=True, exist_ok=True)
repo = 'deepseek-ai/DeepSeek-V4-Flash-0731'
revision = '7872f01b1d1fe23eabc4c98b48bffcef5a386062'
def fetch(name):
    with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}') as r:
        data = r.read()
    (root / Path(name).name).write_bytes(data)
    return json.loads(data)
config = fetch('inference/config.json')
index = fetch('model.safetensors.index.json')
draft = {k:v for k,v in index['weight_map'].items() if k.startswith('mtp.')}
print(json.dumps(dict(config=config, draft_tensors=len(draft), shards=sorted(set(draft.values()))), indent=2))
from huggingface_hub import hf_hub_download
for name in sorted(set(draft.values())):
    print('Downloading ' + name, flush=True)
    hf_hub_download(repo, name, revision=revision, local_dir=root)
print('Draft shards downloaded.', flush=True)
