# Agent kickoff — Qwen3.8 / FreeToken / two-GPU MTP experiment

Read `Qwen38_FreeToken_MTP_Revised_Agent_Plan.md` version 2.0 as the authoritative execution plan. Its paths and current-machine figures are source-reported until verified. Do not treat this kickoff as permission to skip its gates.

## Task

Determine whether the existing local NVFP4 target on the RTX 5090 can be accelerated by its existing BF16 MTP component on the RTX 4090, without changing the checkpoint, target offload policy, existing project launchers, or protected baseline.

Execute the plan's authorized stages G0–G6 in dependency order. Stop at G6 regardless of outcome. Stages G7–G8 require a separate explicit approval. Do not commit, push, publish, contact upstream, download another model, convert weights, change quantization, move target experts to the 4090, or introduce CPU expert computation.

## Begin here

Run the read-only discovery in §4.2. Inspect local repository instructions, baseline launch configuration, actual model/tensor metadata, GPU UUIDs, imported package paths, and existing changes. Do not reset or overwrite existing work.

Create the evidence workspace only inside the verified experiment worktree. Resolve the actual interpreter, model path, target backend, cache/graph settings, and benchmark workload from local evidence. Save them in the execution manifest. Missing required values are blockers, not fields to guess.

Reproduce the protected baseline before modifying execution paths. Then write and run the smallest real multi-row offload-layer probe against the baseline's existing device-side cache path. Do not start by importing a donor PR, designing a generic speculative framework, or implementing multiple transports.

## Critical implementation rules

- A depth-k round verifies **anchor + k candidates**, so the target block has k+1 input rows.
- Accepted count, emitted count, and processed cache length are different quantities. Implement the pending-anchor table from §6.3 exactly or document and prove an equivalent adapter.
- Export the target's correct wide pre-mixer feature. Keep MTP's LM-head representation and recursive representation separate.
- Accepted token IDs do not prove recursive draft KV is canonical. Validate accepted-target-feature catch-up and include its bridge/compute cost.
- Preserve GDN, convolution, PLE, QSA/index/ring, page, and scheduler state at the chosen frontier. The reference replay path is diagnostic only; no target replay may count toward final performance.
- Use the real total context limit and reserve output capacity. Do not label a 261120-input/1024-output run as 262144 input tokens.
- Build a minimal synchronized prefix sidecar before the final cache-enabled benchmark. Missing sidecars preserve the target cache hit and use target-only decoding.
- Keep MTP-off free of draft startup, feature capture, allocations, or synchronization overhead.

## Working method

Use a hypothesis, a minimal executable probe, observed evidence, focused hardening/tests, and then a rerun. Unit tests and static analysis are required supporting checks—not evidence that the hardware path works. Never tune performance around known incorrect output.

Use actual runs and real tensors. Do not replace a hardware probe with mocks, invent missing CLI flags, or present a proposed command as already runnable. A successful oracle-candidate benchmark is not an MTP speed result.

Run only one full target process at a time. Measure the full two-process path under load and keep raw logs. Identify every result by baseline SHA plus experiment diff/untracked-source hashes, because commits are not authorized.

## Report at each gate

Update `.mtp-exp/status.json` and provide: hypothesis; changed files; exact commands; observed output; correctness comparison; timings/memory; PASS/FAIL/INCONCLUSIVE/BLOCKED; unverified assumptions; next authorized action.

Continue only when the prerequisite gate is satisfied. When blocked, deliver the work and evidence already completed with the exact blocker. Do not claim local runtime validation without access to the checkpoint and GPUs.

The final greedy gate requires the locked workload to reach at least 48 token/s and at least 20% improvement over the fresh protected baseline, while satisfying the unchanged correctness, TTFT, MTP-off regression, traffic-explanation, and end-to-end requirements. A passing greedy experiment is not a completed sampled-serving implementation.
