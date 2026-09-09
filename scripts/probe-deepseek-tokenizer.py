import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from freetoken.models.gguf.tokenizer import load_gguf_tokenizer, gguf_eos_token_ids
from freetoken.models.gguf.reader import load_gguf_metadata

model='/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS/IQ3_XXS/DeepSeek-V4-Flash-0731-IQ3_XXS-00001-of-00004.gguf'
official='/home/rba90/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062'
a=load_gguf_tokenizer(model)
b=PreTrainedTokenizerFast.from_pretrained(official)
meta=load_gguf_metadata(model)
vocab=meta['tokenizer.ggml.tokens']
print('vocab mismatch', [(i,t,b.convert_ids_to_tokens(i)) for i,t in enumerate(vocab) if t != b.convert_ids_to_tokens(i)][:20])
print('stop',gguf_eos_token_ids(model,a),'official',b.eos_token_id)
for s in ['ORCHID-7319', 'The code is ORCHID-7319. Repeat the code exactly.', 'The archive contains ordinary records for review.\n', '<｜User｜>Hello<｜Assistant｜></think>']:
    print(json.dumps({'text':s,'gguf':a.encode(s,add_special_tokens=False),'official':b.encode(s,add_special_tokens=False)}))
print('gguf pretokenizer',a.backend_tokenizer.pre_tokenizer.__getstate__())
print('official pretokenizer',b.backend_tokenizer.pre_tokenizer.__getstate__())
