# Issue for Niko1221/Strata: posted on 2026-11-23 as [Niko1221/Strata#6695](https://github.com/Niko1221/Strata/issues/6695)

Where: **upstream, Niko1221/Strata** (bug report form). The reader is upstream's (dd015b8c3, 2026-11-12, "drafting:
decide DFlash2 by what the model file is when its name is not one setup writes") and is unchanged in v0.2.19
(`terminal/strata_control/drafting.py:580`). sergqwer/strata-nvfp4 carries the same code and will get a fix with
its next port; it needs no issue of its own.

Form fields: Area **[Engine]** · How: **Source** · Reproduction status: **Reproduced locally** · Version: **Strata
0.2.15 via strata-nvfp4 0.2.15-nvfp4.7 (engine 0.1.61); the same function in v0.2.19 read, not run** · Hardware:
**RTX 5090 32 GB · 128 GB RAM · Windows 11**

---

**Title:** [Engine] drafting: `gguf_facts` stops at the first string array, so a renamed llama.cpp GGUF never "fits" and DFlash2 is refused

### What happened

`file_fit()` (dd015b8c3) reads a GGUF's header to decide whether a run config with an unknown `model_name` is
Qwen3.8-Flash-Next. `gguf_facts()` walks the key/value pairs and stops at the first array whose elements are strings:

```python
                else:
                    break                                   # an array of words before the tokens: not looked into
```

GGUFs written by llama.cpp's converter carry `general.tags` (and often `general.languages`) among the `general.*`
keys, before `<arch>.block_count` and `<arch>.embedding_length`. For those files the reader returns no sizes, and
`file_fit()` answers `family` for a file that is exactly the model. The start then logs
`Drafting: MTP (DFlash2 is for Qwen3.8-Flash-Next only)` and runs the MTP, also when DFlash2 is chosen by name.

Every llama.cpp-written GGUF on this PC reads that way. `general.tags` is key 16 in Unsloth's Qwen3.8-27B
`UD-Q4_K_M` and `UD-Q4_K_XL`, and key 17 in a Qwen3.8-Flash-Next ModelOpt NVFP4 file converted by
`tools/nvfp4_convert.py`:

```
Qwen3.8-27B-UD-Q4_K_M.gguf   {'arch': 'qwen35',   'layers': None, 'hidden': None, 'vocab': None}
Qwen3.8-27B-UD-Q4_K_XL.gguf  {'arch': 'qwen35',   'layers': None, 'hidden': None, 'vocab': None}
gemma-4-31B-it-Q4_0.gguf     {'arch': 'gemma4',   'layers': None, 'hidden': None, 'vocab': None}
orca-nvfp4.gguf              {'arch': 'qwen4exp', 'layers': None, 'hidden': None, 'vocab': None}   -> file_fit: family
```

`tests/terminal/data/flash_next_iq3s_header.gguf` has no string array before `tokenizer.ggml.tokens` (its keys run
`general.architecture` ... `general.size_label`, then `qwen4exp.*`), so `test_drafting_file_fit.py` passes.

Expected: the sizes are read past a string array, and a renamed Qwen3.8-Flash-Next GGUF fits.

### Smallest steps

```python
from strata_control import drafting as d
print(d.gguf_facts("<any llama.cpp-converted GGUF with general.tags>"))     # layers / hidden / vocab: None
print(d.file_fit(["--native", "<a Qwen3.8-Flash-Next GGUF with general.tags>"], "<its data folder>"))   # 'family'
```

### A fix

Skip a string array (each element is a u64 length and its bytes) instead of stopping:

```python
                elif etype == 8:
                    for _ in range(count):
                        (slen,) = struct.unpack_from("<Q", head, pos)
                        pos += 8 + slen
                else:
                    break
```

With it the NVFP4 file above reads `{'arch': 'qwen4exp', 'layers': 48, 'hidden': 2560, 'vocab': 248320}` and
`fits`. A header fixture with `general.tags` before the sizes would cover it in `test_drafting_file_fit.py`.

---

Notes for the owner (not part of the issue):
- We do not need the fix: the launcher writes its drafter arguments and sets `"drafting": "mtp"`, which the stock
  policy passes through unchanged (checked on 0.2.15-nvfp4.7). The fix was applied only in the staged copy used for
  the DFlash2 arm on 2026-11-23.
- Left out on purpose: our drafter measurements (fork-specific, and they would read as a request to change Auto), the
  model's name and this workspace's paths.
- The "v0.2.19 read, not run" in the version field is honest: the function is byte-identical between the fork's
  0.2.15-nvfp4.7 (which we ran) and upstream v0.2.19 (which we only diffed).
