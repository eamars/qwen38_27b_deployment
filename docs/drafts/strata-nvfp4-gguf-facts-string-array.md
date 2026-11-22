# Draft: fork issue for sergqwer/strata-nvfp4 (not posted)

**Title:** drafting: `gguf_facts` stops at the first string array, so a ModelOpt NVFP4 GGUF with `general.tags` reads as "another size" and loses DFlash2 / Auto

**Body:**

On 0.2.15-nvfp4.7 (`4442f2d53`), `strata_control/drafting.py` `gguf_facts()` reads a GGUF's first megabyte for
`general.architecture`, `<arch>.block_count`, `<arch>.embedding_length` and the token count. When it meets an array
whose elements are strings before the tokens, it stops (`else: break  # an array of words before the tokens: not
looked into`).

A GGUF written by `tools/nvfp4_convert.py` from a ModelOpt checkpoint carries `general.tags` (an array of 11 strings)
as key 17, before `qwen4exp.block_count` and `qwen4exp.embedding_length`. So the reader returns
`{'arch': 'qwen4exp', 'layers': None, 'hidden': None, 'vocab': None}`, and `file_fit()` answers `family` for a file
that is exactly Qwen3.8-Flash-Next (48 layers, hidden 2560, 248,320 tokens).

What that changes for a run config whose `model_name` is not one setup writes:
- `"drafting": "dflash2"` is refused ("DFlash2 is for Qwen3.8-Flash-Next only") and the start falls back to the MTP.
- With Auto, any DFlash2 flags written in `args` are stripped, and the MTP with Basalt's drafting is not applied.
- Whether Auto rewrites hand-written MTP flags therefore depends on this header detail.

Reproduce (any ModelOpt-converted GGUF; ours is OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4):

```python
from strata_control import drafting as d
print(d.gguf_facts("orca-nvfp4.gguf"))              # layers / hidden / vocab None
print(d.file_fit(["--native", "orca-nvfp4.gguf"], "<model dir>"))   # 'family'
```

A fix that skips string arrays (each element is a u64 length and its bytes) instead of stopping:

```python
elif etype == 8:
    for _ in range(count):
        (slen,) = struct.unpack_from("<Q", head, pos)
        pos += 8 + slen
```

With it the same file reads `{'arch': 'qwen4exp', 'layers': 48, 'hidden': 2560, 'vocab': 248320}` and `fits`.

For what it is worth, on this file on an RTX 5090 (9950X3D, 128 GB, 0.2.15-nvfp4.7), the drafters measured the same
session: MTP with Basalt's drafting and Basalt's draft layer 216 / 216 tok/s (Chinese / English chat), with the pack's
own layer 209 / 214, the plain MTP (window 4) 203 / 200, DFlash2 (stock pack, conf 0.5, window 6) 170 / 196. So Auto's
choice of the MTP for this set is right here; only the DFlash2-by-name path is blocked.

---
Notes for the owner (not part of the issue): we run explicit drafter flags with `"drafting": "mtp"`, so production
does not need this fix; the reader was patched only in the staged copy used for the DFlash2 arm on 2026-11-23.
