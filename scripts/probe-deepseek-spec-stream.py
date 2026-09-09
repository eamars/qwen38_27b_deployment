from freetoken.models.gguf.tokenizer import load_gguf_tokenizer, gguf_eos_token_ids
from freetoken.tokenizer.detokenize import DetokenizeManager
from freetoken.message import DetokenizeMsg

model='/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS/IQ3_XXS/DeepSeek-V4-Flash-0731-IQ3_XXS-00001-of-00004.gguf'
tokenizer=load_gguf_tokenizer(model)
expected='Record 1: alpha cedar maple river orchid stone silver forest amber cloud.\nRecord 2: alpha cedar maple.'
ids=tokenizer.encode(expected,add_special_tokens=False)
messages=[DetokenizeMsg(uid=1,next_token=t,finished=i==len(ids)-1) for i,t in enumerate(ids)]
manager=DetokenizeManager(tokenizer,frozenset(gguf_eos_token_ids(model,tokenizer)))
out=''.join(text for i in range(0,len(messages),3) for text in manager.detokenize(messages[i:i+3]))
assert out==expected,(out,expected)
print('Three-token speculative streaming matches the original text exactly.')
