"""Read remote GGUF headers without downloading tensor payloads."""
import collections
import json
import pathlib
import urllib.parse
import urllib.request

import gguf

ROOT = pathlib.Path('/home/rba90/deepseek-freetoken-probe')
ROOT.mkdir(exist_ok=True)

def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)

results = []
for repo, quant in [('tarruda/DeepSeek-V4-Flash-0731-GGUF', 'IQ3_XXS'),
                    ('unsloth/DeepSeek-V4-Flash-0731-GGUF', 'UD-IQ3_XXS')]:
    meta = get_json(f'https://huggingface.co/api/models/{repo}')
    revision = meta['sha']
    entries = get_json(f'https://huggingface.co/api/models/{repo}/tree/{revision}/{quant}?limit=1000')
    summary = {'repo': repo, 'revision': revision, 'quant': quant, 'files': [], 'experts': []}
    for ent in entries:
        if not ent['path'].endswith('.gguf'):
            continue
        path = ROOT / 'headers' / repo.split('/')[0] / pathlib.Path(ent['path']).name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{ent["path"]}'
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-8388607'})
        with urllib.request.urlopen(req) as r, path.open('wb') as f:
            data = r.read(8388608)
            f.write(data)
            f.truncate(ent['size'])
        reader = gguf.GGUFReader(str(path), 'r')
        for tensor in reader.tensors:
            if '_exps.weight' in tensor.name:
                summary['experts'].append({'name': tensor.name, 'type': tensor.tensor_type.name,
                                           'shape': tensor.shape.tolist(), 'bytes': tensor.n_bytes})
        summary['files'].append({'path': ent['path'], 'size': ent['size']})
        del reader
        path.unlink()
    summary['type_counts'] = dict(collections.Counter(x['type'] for x in summary['experts']))
    summary['expert_gib'] = sum(x['bytes'] for x in summary['experts']) / 2**30
    results.append(summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('experts','files')}), flush=True)
(ROOT / 'header-inventory.json').write_text(json.dumps(results, indent=2))
