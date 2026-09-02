# Model manifest

Generated: 2026-08-26 11:19:43 +12:00

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

| Role | Repository | Filename | Quantization | Size bytes | Size GB | SHA-256 | Download date |
|---|---|---|---|---:|---:|---|---|
| RTX 5090 primary target | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q6_K_M.gguf | UD-Q6_K_M | 23088409504 | 21.503 | 6629d378ec65deaa772917e9b2b031c97f07aa710f9cf218ca2a0a32e8531fcc | 2026-08-20 |
| RTX 5090 context fallback | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q6_K.gguf | UD-Q6_K | 21983677344 | 20.474 | c9c206812fbe4ac7b76a729e25928b63f2ae89d37f69da7a71c20aec763cd436 | 2026-08-20 |
| RTX 4090 primary target | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q4_K_XL.gguf | UD-Q4_K_XL | 17559178144 | 16.353 | 3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e | 2026-08-20 |
| RTX 4090 context fallback | unsloth/Qwen3.8-27B-GGUF | Qwen3.8-27B-UD-Q4_K_M.gguf | UD-Q4_K_M | 16464440224 | 15.334 | 322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482 | 2026-08-20 |
| DFlash2 drafter for both backends | incoai/Qwen3.8-27B-DFlash2-GGUF | Qwen3.8-27B-DFlash2-Q4_K_M.gguf | DFlash2 Q4_K_M | 1143006752 | 1.065 | 18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594 | 2026-08-20 |
| Gemma 4 5090 experimental target | LM Studio local import | Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf | i1-Q4_K_M | 18687066112 | 17.404 | 1a4c20908471ff51916a35915bce73874100331f80a7433f407f77993c5539f3 | 2026-08-17 |
| Gemma 4 4090 experimental target | mradermacher/Gemma-4-31B-Isometry-Fabled-Persona-i1-GGUF | Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf | i1-Q4_K_S | 17763168256 | 16.543 | b4a7c4e3d22f523202806b5c9bf0f2e374cb4e8a0cc17c34457c3b4d66af5dc7 | 2026-08-24 |
| Gemma 4 MTP drafter | ggml-org/gemma-4-31B-it-GGUF | mtp-gemma-4-31B-it-Q8_0.gguf | MTP Q8_0 | 514687104 | 0.479 | 6b52ab20af503aee320dc09e93f886133b18d89ffc9075c7d9dcaf681e20b375 | 2026-08-24 |
