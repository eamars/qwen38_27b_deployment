# Model manifest

Generated: 2026-09-26 14:48:01 +12:00

All model artifacts are local under `models/` and are ignored by Git. SHA-256
values below were calculated from the completed files in this workspace. The
Qwen rows are the deployment set; Gemma rows are experimental and are included
when those files are present. Regenerate this file with
`scripts/record-model-manifest.ps1` after replacing an artifact.

The Qwen3.8 Flash-Next deployment uses the complete
`RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint under WSL at
`/home/rba90/models/Qwen3.8-Flash-Next-NVFP4`. Its indexed weight files total
135195303851206 bytes. It is intentionally outside the generated Windows
`models/` table below; the pinned runtime and benchmark record its repository
revision and validate that all indexed shards are present.

The native Windows Strata deployment has a separate ModelOpt NVFP4 checkpoint
under `models/qwen38-flash-next-uncensored-strata/checkpoint/`. All 220 files
(135,253,671,102 bytes) were verified on 2026-10-03 against revision
`f24d2b68ff2814f24455ae86717be276619b5664`. The complete source inventory and
hashes are in [qwen38-strata-checkpoint-manifest.json](qwen38-strata-checkpoint-manifest.json);
generated assets and launch commands are in the [Strata runbook](qwen38-flash-next-strata.md).

| Role | Repository | Filename | Quantization | Size bytes | Size GB | SHA-256 | Download date |
|---|---|---|---|---:|---:|---|---|
| RTX 5090 primary target | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q6_K_M.gguf | UD-Q6_K_M | 23088409504 | 21.503 | 493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb | 2026-08-20 |
| RTX 5090 context fallback | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q6_K.gguf | UD-Q6_K | 21983677344 | 20.474 | c9c206812fbe4ac7b76a729e25928b63f2ae89d37f69da7a71c20aec763cd436 | 2026-08-20 |
| RTX 4090 primary target | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q4_K_XL.gguf | UD-Q4_K_XL | 17559178144 | 16.353 | 3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e | 2026-08-20 |
| RTX 4090 context fallback | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q4_K_M.gguf | UD-Q4_K_M | 16464440224 | 15.334 | 322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482 | 2026-08-20 |
| DFlash2 drafter for both backends | incoai/Qwen3.8-27B-DFlash2-GGUF | Qwen3.8-27B-DFlash2-Q4_K_M.gguf | DFlash2 Q4_K_M | 1143006752 | 1.065 | 18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594 | 2026-08-20 |
| Gemma 4 5090 experimental target | LM Studio local import | Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf | i1-Q4_K_M | 18687066112 | 17.404 | 1a4c20908471ff51916a35915bce73874100331f80a7433f407f77993c5539f3 | 2026-08-17 |
| Gemma 4 4090 experimental target | mradermacher/Gemma-4-31B-Isometry-Fabled-Persona-i1-GGUF | Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf | i1-Q4_K_S | 17763168256 | 16.543 | b4a7c4e3d22f523202806b5c9bf0f2e374cb4e8a0cc17c34457c3b4d66af5dc7 | 2026-08-24 |
| Gemma 4 MTP drafter | ggml-org/gemma-4-31B-it-GGUF | mtp-gemma-4-31B-it-Q8_0.gguf | MTP Q8_0 | 514687104 | 0.479 | 6b52ab20af503aee320dc09e93f886133b18d89ffc9075c7d9dcaf681e20b375 | 2026-08-24 |
| Gemma 4 4090 QAT Instruct target | unsloth/gemma-4-31B-it-qat-GGUF | gemma-4-31B-it-qat-UD-Q4_K_XL.gguf | QAT UD-Q4_K_XL | 17287670048 | 16.1 | 00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71 | 2026-09-09 |
| Gemma 4 QAT MTP drafter | unsloth/gemma-4-31B-it-qat-GGUF | MTP/mtp-gemma-4-31B-it-Q8_0.gguf | MTP Q8_0 | 514705920 | 0.479 | e6e88dea4fcf79a0a71cd7732e61632355126dbbda9b72a08d6301fa06fbaf22 | 2026-09-09 |
| Gemma 4 4090 HauhauCS uncensored target | HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP | Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf | QAT Q4_K_M | 18687062176 | 17.404 | 71667f9e601a4b914a98425c59150b731f6e15d260d661dbd1f1ee07469fc7db | 2026-09-10 |
| Gemma 4 HauhauCS MTP drafter | HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP | mtp-gemma-4-31B-it.gguf | MTP | 279954368 | 0.261 | b5c4e583fc5982439080114bbc1b7edaec361f9d4c9193d6bed606a3de401b62 | 2026-09-10 |
| Gemma 4 HauhauCS vision projector | HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP | mmproj-Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf | BF16 mmproj | 1200726016 | 1.118 | 7bef0d0fb3e85fc2941ec5f1c375febf3742645f158132a43ced557093aea841 | 2026-09-10 |
| RTX 4090 Turbo Fable Cold Fusion vision/MTP target | DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF | Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf | LOW-MTP-IQ4_XS | 15309039136 | 14.258 | fa92183638b045b01447fd657afd6220f0995ca3a31b528078cefdf26a96c820 | 2026-09-26 |
| Qwen3.8 Turbo Fable Cold Fusion vision projector | unsloth/Qwen3.8-27B-GGUF | mmproj-F16.gguf | F16 mmproj | 927607488 | 0.864 | cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e | 2026-09-26 |
