# FreeToken #447 本地实验历史

2026-10-03 cleanup: the official RadixArk checkpoint and dedicated PR #447
benchmark scripts have been removed. Results, launch metadata, patches and
raw logs remain preserved; script references below are historical.

确认日期：2026-09-29。2026-09-30 已完成 128K cold 请求，平均 decode 31.16 token/s；用户随后决定结束本轮探索、记录历史并提交。推理结束时模型保持驻留；提交前用户追加要求直接卸载且不要检查，已向服务进程组发送停止信号，未复查卸载状态。以下最新要求优先于原始范围；时间按 Pacific/Auckland 记录，本轮为 UTC+13。

## 2026-09-30 最新执行要求

- 以速度为目标，不要求相同模型或量化格式，可以载入非 NVFP4 模型。
- 用户明确纠正：40 token/s 指完整请求的平均 decode；必须让请求完成后再判定，不能因为单个窗口低于 40 就中断。
- 继续 PR #447 的 TP2+EP2 实验，保留 262,144-token 上下文与 KV 池、单请求。
- 用户追加：4K 后测试 128K cold prefill + decode，模型不要卸载；即使早先 4K 某窗口不足 40，也继续 128K。
- 平均 decode = `(实际输出 tokens - 1) / (最后非空输出块时间 - 首个非空输出块时间)`，排除 prefill/TTFT；各 40-forward 窗口仅作诊断。
- 驻留服务测速脚本：`scripts/benchmark-pr447-resident.py`；默认 131,072 输入、请求 1,024 输出。`scripts/run-pr447-speed-gate.py` 也已修正为完整请求平均门槛及保持驻留。本轮日志目录：`benchmarks/raw/qwen38_flash_next/pr447-speed-2026-09-30/`。

## 2026-09-30 最终实测

使用 **RadixArk Qwen3.8-Flash-Next-NVFP4**：专家 NVFP4，dense/GDN/attention BF16；没有加载非 NVFP4 专家权重，也不是原 Uncensored 版本。FreeToken 0.1.2，PR commit `81d034d4a2c0f9f4024b8ba820a1cbe65db6d1d5`，仅保留接手前已有的显存预算兼容补丁。

| 指标 | 128K cold 实测 |
|---|---:|
| 实际输入 | 131,072 tokens |
| 实际输出 | 1,023 tokens |
| 前缀缓存命中 | 0 tokens |
| TTFT | 66.0663 秒 |
| decode 计时区间 | 32.8031 秒 |
| 完整请求平均 decode | **31.1556 token/s** |
| 相对 40 token/s | **-22.11%** |
| 有效 prefill（输入 tokens / TTFT） | 1,983.95 token/s |
| 请求总耗时 | 98.8713 秒 |
| 结束原因 | `length`，正常收到末尾 usage |

请求参数为 `max_tokens=1024, ignore_eos=true`，服务实际返回 1,023 tokens；usage、1,023 个非空 SSE 输出块、输出文本重新分词三者一致。本轮不能写成完成了精确 1,024-token 输出。有效 prefill 包含请求处理及首 token 延迟，不冒充纯 GPU prefill 时间；未使用混入空闲时间的首条调度器吞吐日志作为速度。

4K 阶段实际输入 4,096、输出 127 tokens，纯 decode 窗口为 34.11 / 47.78 token/s。最初把单窗口当停止条件是本次执行的错误；用户纠正后已撤销。更换控制器时该 4K 请求正常结束，但原客户端逐 token 计时没有完整保存，因此不据此宣称完整 4K 平均值。随后按用户指示继续 128K，全程保持同一模型实例。

128K 请求运行中收到“必须完整结束再算平均”的纠正：更换日志管道读取器，使旧控制器的单窗口停止监视器不再读取新日志；原 HTTP 请求和模型均未重启。完整原始日志分段保存在 `server.log` 与 `server-128k-continuation.log`。原始 result 的 `stopped_below_40` 表示请求完成后不再做后续实验，不表示中途取消；整理结果明确记录为 `completed_below_40`。

推理结束时服务状态（历史快照）：`http://127.0.0.1:19447/v1`，served model `qwen38-next-pr447-speed`，WSL PID `60429`，instance ID `ca5506d2-eeaf-49d5-b5fd-8d6b0ac3e9e9`。4K 与 128K 前后实例一致；当时 `/health=ok`、active requests=0，权重保持驻留。仅验证了文本请求，未验证 vision 功能。提交前的停止操作见下文。

结果与复现证据：

- [整理结果 JSON](../benchmarks/qwen38_flash_next/2026-09-30/pr447-128k-cold-resident.json)
- 原始请求、SSE 时间戳、前后 stats/health：`benchmarks/raw/qwen38_flash_next/pr447-speed-2026-09-30/resident-131072-011337/`
- 纳入 Git 的复现证据：[实际启动参数及环境](../artifacts/freetoken/pr447-2026-09-30/launch.json)、[版本信息](../artifacts/freetoken/pr447-2026-09-30/runtime-metadata.json)、[继承的显存兼容补丁](../artifacts/freetoken/pr447-2026-09-30/inherited-memory-compat.patch)。原件在上述 raw 根目录。
- raw 请求、流与服务日志留在本机，按仓库规则不纳入 Git。整理 JSON 是最终结果，不能用 raw 根目录中早期失败尝试的 `result.json` 替代。
- 两个脚本保留本机实验路径。最后的平均门槛及驻留控制修改仅做语法检查，没有重新启动模型或重跑推理；本轮实际执行经过上文所述的控制器交接。驻留脚本再次使用同一 prompt 可能命中前缀缓存，是否 cold 必须核对实际 `cached_tokens`。

## 接手与启动处理

接手时发现原文未记载的后续痕迹：`engine.py` 已有异构显存兼容补丁（按跨 rank 最小空闲显存预算，并将显存差错误改为警告）；`pr447-local-2026-09-30/server.log` 00:45 的启动已越过该检查，但 uncensored 权重触发 `qwen4_exp dense TP currently supports the BF16 GDN path only`。保留这些已有修改。

本轮选择本地 `/home/rba90/models/Qwen3.8-Flash-Next-NVFP4`（RadixArk，BF16 dense/GDN + NVFP4 routed experts），其索引引用的 206 个权重文件全部存在。PR 的 owner EP 当前要求 NVFP4 专家；该本地模型可以直接满足其 BF16 dense TP 路径，无需等待额外模型下载。

本轮新增启动发现及处理：

- `FREETOKEN_PIN_BUDGET_GB=64`：覆盖 WSL 默认的 44 GiB 预估检查，实际 63.5 GiB 专家库已经完整加载、锁页成功；没有跳过实际 `cudaHostRegister`。
- `TVM_FFI_CUDA_ARCH_LIST="8.9 12.0"`：为两张异构 GPU 编译扩展。此变量用空格分隔。
- `FLASHINFER_USE_CUDA_NORM=1`：切换 FlashInfer 官方 CUDA norm 实现。默认 CuTe norm 在 rank 1 预热时生成了含 `griddepcontrol` 的 PTX，却以 `sm_89` 编译，报错并中止启动。
- CUDA norm 的 JIT 还需要 `cublas*.h` / `curand*.h`。脚本将隔离 venv 自带的这些库头文件复制到该 venv 的 `include/cublas-compat`，通过 `CPATH` 提供；不把整个 cu13 include 加入 `CPATH`，避免其中 CUDA 13.0 头文件覆盖 13.3 工具链头文件。独立探针已在 4090 与 5090 上执行 `gemma_rmsnorm`，并通过 BF16 参考输出校验（`rtol=atol=0.02`）。
- `NCCL_P2P_DISABLE=1`，GPU rank 顺序沿用 `1,0`（4090、5090）。
- 首次完整分配时，自动专家缓存每 rank 为 3,759 槽，其中前 512 槽供双缓冲 prefill 复用；每卡 KV 为 262,144 tokens / 3.19 GiB。以最终启动日志为准。
- 当前分支不支持 `--linear-state-cache-ratio` CLI 参数，使用其默认值 2.0；不传 `--moe-cpu-layers` 和 `--mm-encoder-weights`。

## 测后核查与讨论（2026-09-30）

### NVFP4 与 CPU 参与范围

- 实测始终使用 NVFP4 专家权重，没有进行其他量化格式的专家测速。用户允许换模型，不代表已经测试了非 NVFP4 模型。
- 日志选择的是 `nvfp4 via triton`。该路径的 decode 内核解包 FP4 权重并进行缩放、乘加归约；不能将“加载 NVFP4 权重”等同于“使用原生 FP4 Tensor Core 指令”。源码依据：`layers/quantization/moe/nvfp4.py`、`kernel/triton/nvfp4_fused_moe.py`。
- 实际参数为 `moe_cpu_layers=None`、`moe_cpu_threads=0`，owner EP 不支持 CPU expert layers；本轮没有把专家矩阵计算分配给 CPU。CPU 仍参与调度、分词、内存中的专家库管理及磁盘 PLE 等主机工作，不能说 CPU 完全没有参与。
- 本轮没有分 rank 的耗时剖析，不能仅凭有效 prefill 接近 2,000、decode 约 31，就证明 4090 是主要瓶颈。专家缺页搬运、跨卡通信、路由负载和长上下文 attention 都未被独立量化。

### 两卡分配与不均分专家的设想

[PR #447](https://github.com/FlashML-org/FreeToken/pull/447) 的当前实现按连续专家编号等量划分，不按异构显卡性能或显存加权。`moe/ownership.py` 的 `ExpertOwnership.local_num_experts` 为全局专家数除以 EP 卡数。

| 项目 | rank 0：RTX 4090 | rank 1：RTX 5090 |
|---|---:|---:|
| 每层专家编号 | 0–255 | 256–511 |
| 每层负责专家数 | 256 | 256 |
| 跨层 GPU 专家缓存槽 | 3,759 | 3,759 |
| KV token 池容量 | 262,144 | 262,144 |
| 日志报告 KV 池占用 | 3.19 GiB | 3.19 GiB |

缓存预算按跨 rank 最小空闲显存求解。继承补丁让异构显存配置可以启动，但没有让 5090 获得更大的独立缓存预算；专家数量均分也不保证实际路由计算量均分。

讨论过让 4090/5090 每层分别负责 **192/320** 个专家，并按每卡显存独立配置缓存。这只是待验证的起始比例，没有实现、没有测速。需要修改专家归属映射、加载和缓存几何，当前没有可直接设置的 CLI 参数，也不支持在线热切换。若只增加 5090 负责的专家数量而不增加其缓存，可能因缓存覆盖率下降而抵消计算收益。

### 单卡 KV 与 attention 的设想

实际模型配置有 24 个 query heads、2 个 KV heads，TP2 下每卡负责 1 个 KV head。两卡同为 262K 容量表示同一上下文的不同 head 分片，并非两份完整 K/V；QSA 的压缩索引另有副本。依据为 `models/qwen4_exp/config.py`、`attention.py` 和 `kvcache/qsa_pool.py`。

- 只把 KV 存储集中到 5090、仍由两卡分别计算 attention，会新增远程 KV 获取。结合本轮 `NCCL_P2P_DISABLE=1`，预期不利于速度，但未测试。
- 同时将完整 attention 和 KV 放到 5090、让 4090 继续分担专家，是另一种可能的架构。可释放 4090 的 KV 池空间用于专家缓存，但会增加 5090 的 attention 负担和显存占用，仍需向另一卡传递结果；不能据此预言超过 40 token/s。
- 该模型使用 QSA 稀疏 attention，不能按每个 decode token 必须读取全部 128K K/V 来估算收益。单卡 attention 方案还须与专家比例联合规划，避免把计算集中到 5090。当前 PR 未提供这种分工，本轮没有修改或运行该方案。

### 本轮结束决定

用户在上述讨论后要求“记录一下历史，然后 commit”。本轮到此结束：保留完整请求平均门槛、实测结果、控制器纠正过程及未验证设想；不继续参数搜索，不修改专家/KV 分配，不重启模型。该结果不作为正式部署升级依据。

提交前用户追加要求“先帮我把模型卸载，不要检查。然后继续之前的操作”，覆盖此前保持驻留的要求。已直接执行 `wsl -d Ubuntu -- kill -TERM -- -60429`，向已知服务进程组发送 SIGTERM；命令返回 0。按要求没有再查询进程、显存或健康状态，因此不记录未经复查的显存释放数值。结果 JSON 中的驻留和 health 字段保留为测试完成时的历史快照。

以下保留原始实验要求及早期记录供追溯；与上文最新用户要求冲突时，以上文为准。

## 原始约定：现有性能基准

直接采用用户提供的当前 FreeToken 0.1.3 单卡部署实测数据：

| 指标 | 基准 |
|---|---:|
| 上下文配置 | 262,144 tokens |
| 平均 decode | 40 token/s |
| Prefill | 约 1,450 token/s |

这组数据作为收益计算依据，无需重新测量单卡。历史 4K 留档中的 50.59 token/s 不用于本次收益计算。

## 原始约定：唯一测试 C

| 项目 | 要求 |
|---|---|
| FreeToken 变更 | [PR #447](https://github.com/FlashML-org/FreeToken/pull/447)，TP2+EP2 |
| GPU | RTX 5090 + RTX 4090 |
| 本地链路条件 | PCIe 5.0 x8 + PCIe 4.0 x8 |
| 部署依据 | [当前 uncensored vision 启动脚本](../scripts/start-qwen38-flash-next-uncensored-freetoken-vision.ps1) |
| 模型 | 当前部署的 `Qwen3.8-Flash-Next-Uncensored-NVFP4` |
| 服务功能 | 保留当前 vision 功能；本次请求使用文本 prompt |
| 上下文容量 | 262,144 tokens |
| KV token 池 | 262,144 tokens |
| 并发请求数 | 1 |
| 输入 | 约 4K tokens，记录实际输入 token 数 |
| 输出 | 固定生成 1,024 tokens |
| 专家缓存 | 使用 PR 支持的自动配置，利用两卡可用显存，记录最终配置 |
| 其余服务设置 | 以当前启动脚本为基础，包括 offload、磁盘 PLE 和 host vision encoder |

262K 表示服务上下文容量及 KV 池；正式请求输入约 4K tokens。

## 原始约定：执行与测量

1. 在上述 C 配置下完成必要的编译与运行预热。
2. 用未命中前缀缓存的约 4K prompt，完成一次正式的 1,024-token 生成。预热使用独立 prompt，避免正式请求复用其前缀。
3. 记录平均 prefill、平均 decode、TTFT 和请求总耗时。平均 decode 取正常生成区间，排除 prefill 和预热；报告平均值。
4. 保存正式请求的实际输入/输出 token 数、计时口径、FreeToken 版本及测试 commit、实际启动参数和结果日志，确认请求正常完成。
5. 将实测速度直接与用户提供的现有基准计算收益。

若生成提前结束或启动失败，应如实记录实际情况，并处理完成 C 所必需的问题；该次结果不能冒充完成了固定输出长度的正式测量。

## 早期失败记录（2026-09-30，已由后续实测更新）

- 独立克隆位于 `runtime/freetoken-pr447`，测试分支 `pr/tp-ep-clean`，commit `81d034d4a2c0f9f4024b8ba820a1cbe65db6d1d5`；该 commit 的 CLI 显示 FreeToken 0.1.2。
- TP2+EP2 服务在加载模型权重前退出。PR 分支 `_sync_get_memory` 对 TP rank 的空闲显存差设置了 2 GiB 上限；本机启动日志报告最小值 22.27 GiB、最大值 29.94 GiB，差 7.67 GiB，因此触发 `Memory across TP ranks are imbalanced`。
- 首次启动还发现隔离 venv 缺少 `ninja`；已仅在该 venv 安装。之后启动越过 NCCL 扩展编译，最终阻塞仍是 TP 显存差检查。
- `--moe-cpu-layers auto` 在该分支的 owner EP 路径不受支持；`--mm-encoder-weights host` 也不是该分支可用参数，故启动探针省略这两个选项。模型没有加载，因此 vision 服务功能及 host encoder 放置均未验证。
- 曾在 5090 临时预留约 6.5 GiB 显存以缩小两卡空闲显存差；FreeToken 使用的 `torch.cuda.mem_get_info` 仍报告原先 22.27/29.94 GiB，启动检查未通过。临时预留进程现已停止，测试服务也已退出。
- 本次没有加载权重、预热或提交约 4K prompt 的 1,024-token 请求。decode、prefill、TTFT、总耗时均未测得；不能据此计算 PR 的速度收益。PR 克隆源码未修改。

## 早期失败结果（历史快照）

| 指标 | 当前部署基准 | C 实测 | 收益计算 |
|---|---:|---:|---|
| 平均 decode（token/s） | 40 | 未测：服务启动失败 | 不可计算 |
| 平均 prefill（token/s） | 约 1,450 | 未测：服务启动失败 | 不可计算 |
| TTFT（秒） | 未提供 | 未测：没有提交请求 | 不可计算 |
| 请求总耗时（秒） | 未提供 | 未测：没有提交请求 | 不可计算 |

结论直接说明 C 是否加速、decode 和 prefill 各提升多少，以及本次实际请求的首字延迟和完成时间。

## 原始约定：范围边界

- 只执行 C；不测 A、B 或其他 TP1 配置。
- 不扩展双路并发、524,288-token KV 池、262K 长输入或其他工作负载。
- 不增加固定缓存对照、GPU rank 顺序对照、重复测试矩阵、PCIe/DRAM 独立微基准或参数搜索。
- PCIe、主机内存带宽和异构 GPU 的影响由 C 的实际速度体现。
- 历史 4K 记录的 1,765 MiB 空闲显存是旧测量数据，不作为本次新增验收门槛。
- 此前讨论中的其他硬件分析和对照方案不构成本次额外执行任务。
