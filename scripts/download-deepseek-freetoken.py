"""Fetch the header-verified uniform IQ3 checkpoint at an immutable revision."""
import json
import pathlib
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
repo = 'tarruda/DeepSeek-V4-Flash-0731-GGUF'
revision = 'ad2ca0244d0e62868d0b79d67d14a42d6c41278d'
target = pathlib.Path('/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS')
print(json.dumps({'event': 'download_start', 'repo': repo, 'revision': revision, 'target': str(target)}), flush=True)
snapshot_download(repo, revision=revision, local_dir=target,
                  allow_patterns=['IQ3_XXS/*.gguf', 'README.md'], max_workers=2)
for name in ('tokenizer.json', 'tokenizer_config.json'):
    hf_hub_download('deepseek-ai/DeepSeek-V4-Flash-0731', name,
                    revision='7872f01b1d1fe23eabc4c98b48bffcef5a386062')
sizes = (5251072, 49260045920, 49822244864, 14783041600)
for i, size in enumerate(sizes, 1):
    path=target/'IQ3_XXS'/f'DeepSeek-V4-Flash-0731-IQ3_XXS-{i:05d}-of-00004.gguf'
    assert path.stat().st_size == size, str(path)
result = {'event': 'download_complete', 'repo': repo, 'revision': revision, 'files': [
    {'path': str(p), 'bytes': p.stat().st_size} for p in target.rglob('*.gguf')]}
marker=pathlib.Path('/home/rba90/deepseek-freetoken-probe/download-verified.json')
marker.parent.mkdir(parents=True,exist_ok=True)
marker.write_text(json.dumps(result,indent=2))
print(json.dumps(result), flush=True)
