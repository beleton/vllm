# TinyMoe：AMDuProfPcm 分批实验命令

## 0. 模型规模与实验前提

- 模型路径：`/models/TinyMoe`
- 架构：`MixtralForCausalLM`
- 配置关键信息：
  - `hidden_size=768`
  - `num_hidden_layers=12`
  - `num_attention_heads=12`
  - `num_key_value_heads=12`
  - `num_local_experts=2`
  - `num_experts_per_tok=2`
  - `intermediate_size=2048`
  - `torch_dtype=bfloat16`
  - `max_position_embeddings=2048`
- 实际权重文件：`model-00001-of-00001.safetensors`
  - 张量总数：`159`
  - 总参数量：`147,886,848`，约 `147.89M`
  - 实际 BF16 权重体积：`295,773,696 Bytes`，约 `282.07 MiB`
- 参数粗分解：
  - Expert 权重：`113,246,208` 参数，约 `216.00 MiB`，占 `76.58%`
  - Attention 权重：`28,311,552` 参数，约 `54.00 MiB`，占 `19.14%`
  - Embedding + LM Head：`6,292,224` 参数，约 `12.00 MiB`，占 `4.25%`
- 这个模型的 `num_local_experts=2` 且 `num_experts_per_tok=2`，意味着每个 token 基本会走满 2 个 expert；它更适合做“小权重 footprint”对照，不适合拿来观察“大规模稀疏 MoE 只激活少量专家”的局部性行为。

## 1. 当前机器拓扑与 TP 约束

- 当前机器可见 `8` 个 NUMA 节点。
- `lscpu` 显示总 L3 为 `512 MiB (16 instances)`，因此每个 CCX 的 L3 约为 `32 MiB`。
- 当前系统是 `8` 个 NUMA 节点，对应每个 NUMA 节点大约聚合 `2` 个 CCX，本地聚合 L3 约为 `64 MiB`。
- vLLM CPU 后端在 `VLLM_CPU_OMP_THREADS_BIND=auto` 时，默认是“1 个 rank 绑定 1 个 NUMA 节点”。
- 因而在当前机器上：
  - `TP1` 默认只会用到 `1` 个 NUMA 节点
  - `TP2` 默认只会用到前 `2` 个 NUMA 节点
  - `TP4` 默认只会用到前 `4` 个 NUMA 节点
  - `TP8` 才会默认铺满 `8` 个 NUMA 节点
- 这意味着 `TP2/TP4` 的结果更适合用来观察“每 rank 权重分片变小后 locality 如何变化”，不适合直接当作“整机完全利用下的最优吞吐”来和 `TP8` 横向比较。

## 2. 各 TP 下的 BF16 权重分片大小

| TP | 每 rank 权重大小 | 与当前每 NUMA 节点约 64 MiB 本地 L3 的关系 |
| --- | --- | --- |
| 1 | `282.07 MiB` | 明显大于本地 L3 |
| 2 | `141.03 MiB` | 明显大于本地 L3 |
| 4 | `70.52 MiB` | 略大于本地 L3 |
| 8 | `35.26 MiB` | 小于本地 L3 |

- 上表只比较了“静态权重体积”和“单 rank 可利用的本地聚合 L3”。
- 即使 `TP8` 的静态分片大小低于 `64 MiB`，也不代表后续迭代一定能稳定命中 L3，因为还要考虑：
  - 激活、KV cache、输出缓冲
  - victim L3 语义
  - 跨 CCX 访问仍可能被记为 L3 miss
  - 层间访问距离导致的 cache eviction

## 3. 固定实验负载

- 本文按你现有的固定格式组织：
  - `AMDuProfPcm profile -m xxx -a --start-delay 10000 -d 60 -O ... -- evalscope perf ...`
- 固定负载：
  - `--parallel 1 2 4 8 16 32 64`
  - `--number 1 2 4 8 16 32 64`
- 固定测量窗口：
  - `parallel=16`
  - `number=16`
- 由于 TinyMoe 的 `max_position_embeddings=2048`，这里改为：
  - `--min-prompt-length 896 --max-prompt-length 896`
  - `--min-tokens 1024 --max-tokens 1024`
- 这样总长度为 `1920`，给 chat template 和特殊 token 预留约 `128` token 的余量。
- 建议每一批跑 `3` 次，取中位数。

## 4. 目录规划

- 测试结果建议放在：
  - `/home/zjj/vllm/test_results/TinyMoe/TP1`
  - `/home/zjj/vllm/test_results/TinyMoe/TP2`
  - `/home/zjj/vllm/test_results/TinyMoe/TP4`
  - `/home/zjj/vllm/test_results/TinyMoe/TP8`

## 5. 一次性前置

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

## 6. 统一说明

- 下文所有命令中的：
  - `${TP}` 取值为 `1 / 2 / 4 / 8`
  - `8122` 建议使用 `8122 / 8123 / 8124 / 8125` 之一，避免多个服务相互冲突
- 推荐先跑 `TP8`，再依次跑 `TP4 / TP2 / TP1`。
- 这里默认使用真实 TinyMoe 权重，不使用 `dummy`。
- 推荐显式设置：
  - `VLLM_CPU_OMP_THREADS_BIND=auto`
  - `VLLM_CPU_KVCACHE_SPACE=8`
- `8 GiB` KV cache 对当前 `max_model_len=2048`、`max-num-seqs=64` 已经比较充裕；若启动时报 cache 不足，可上调到 `12` 或 `16`。

## 7. vLLM 启动命令

```bash
vllm serve /models/TinyMoe \
    --served-model-name TinyMoe \
    --port 8122 \
    --dtype bfloat16 \
    --block-size 128 \
    --max-num-seqs 64 \
    --max_model_len 2048 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8
```

### TP 与输出目录对应关系

| TP | 建议端口 | 输出根目录 |
| --- | --- | --- |
| 1 | `8122` | `/home/zjj/vllm/test_results/TinyMoe/TP1` |
| 2 | `8123` | `/home/zjj/vllm/test_results/TinyMoe/TP2` |
| 4 | `8124` | `/home/zjj/vllm/test_results/TinyMoe/TP4` |
| 8 | `8125` | `/home/zjj/vllm/test_results/TinyMoe/TP8` |

## 8. Evalscope 整体 sweep

```bash
evalscope perf \
    --model TinyMoe \
    --parallel 1 2 4 8 16 32 64 \
    --number 1 2 4 8 16 32 64 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 896 --max-prompt-length 896 \
    --tokenizer-path /models/TinyMoe \
    --outputs-dir /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/evalscope_res
```

## 9. Batch-1：内存互连 + miss/TLB

```bash
AMDuProfPcm profile \
    -m ipc,memory,ccm_bw,l1,cache_miss,tlb \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/PCm_res/batch1_mem_ccm_l1_miss_tlb -- \
evalscope perf \
    --model TinyMoe \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 896 --max-prompt-length 896 \
    --tokenizer-path /models/TinyMoe \
    --outputs-dir /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/evalscope_res/batch1_mem_ccm_l1_miss_tlb
```

## 10. Batch-2：缓存来源（L3 + DC）

```bash
AMDuProfPcm profile \
    -m ipc,l3,dc \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/PCm_res/batch2_l3_dc -- \
evalscope perf \
    --model TinyMoe \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 896 --max-prompt-length 896 \
    --tokenizer-path /models/TinyMoe \
    --outputs-dir /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/evalscope_res/batch2_l3_dc
```

## 11. Batch-3：预取与 L2

```bash
AMDuProfPcm profile \
    -m ipc,l2,swpfdc,hwpfdc \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/PCm_res/batch3_l2_prefetch -- \
evalscope perf \
    --model TinyMoe \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 896 --max-prompt-length 896 \
    --tokenizer-path /models/TinyMoe \
    --outputs-dir /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/evalscope_res/batch3_l2_prefetch
```

## 12. Batch-4：Topdown

```bash
AMDuProfPcm profile \
    -m ipc,pipeline_util \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/PCm_res/batch4_topdown -- \
evalscope perf \
    --model TinyMoe \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 896 --max-prompt-length 896 \
    --tokenizer-path /models/TinyMoe \
    --outputs-dir /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/evalscope_res/batch4_topdown
```

## 13. Batch-5：UMC（MSR 模式）

```bash
AMDuProfPcm profile \
    -m ipc,umc \
    -a --msr --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/PCm_res/batch5_umc_msr -- \
evalscope perf \
    --model TinyMoe \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 896 --max-prompt-length 896 \
    --tokenizer-path /models/TinyMoe \
    --outputs-dir /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/evalscope_res/batch5_umc_msr
```

## 14. Batch-6：Roofline

```bash
AMDuProfPcm profile \
    -m roofline \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/PCm_res/batch6_roofline -- \
evalscope perf \
    --model TinyMoe \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 896 --max-prompt-length 896 \
    --tokenizer-path /models/TinyMoe \
    --outputs-dir /home/zjj/vllm/test_results/TinyMoe/NPS4_TP8/evalscope_res/batch6_roofline
```

## 15. 运行建议

- 第一轮建议先跑：
  - `TP1`
  - `TP2`
  - `TP4`
  - `TP8`
- 第二轮如果目标是判断 “L3 miss 是否主要受每 rank 权重体积支配”，优先观察：
  - `Batch-2` 的 `L3 Miss %`、`L3 Miss (pti)`、`Ave L3 Miss Latency (ns)`
  - `Batch-3` 的 `HwPf DC Fills From DRAM or IO connected in local node (pti)`
  - `Batch-4` 的 `Backend_Bound.Memory`、`Retiring`
- 预期上，若权重 footprint 是主导因素，则从 `TP1 -> TP8` 应更容易看到：
  - `L3 Miss %` 下降
  - `Ave L3 Miss Latency` 下降
  - `Remote/Local Memory DC Fills` 向更近端层级回落
  - `Retiring` 上升或 `Backend_Bound.Memory` 下降

## 16. 关于 TP2/TP4 不能铺满 8 个 NUMA 节点

- 在当前文档默认的 `auto` 绑核策略下：
  - `TP2` 只会启用前 `2` 个 NUMA 节点
  - `TP4` 只会启用前 `4` 个 NUMA 节点
- 这是当前实验设计的既定约束，不是命令写错。
- 如果你强行用手工绑核把一个 rank 铺到多个 NUMA 节点上，确实可以“用满”更多节点，但那会把实验从“1 rank 对应 1 NUMA 域的 locality 测试”变成“跨 NUMA 的 spread/interleave 测试”，结论口径会变化。
- 因此建议：
  - 如果目标是研究 “权重分片大小与 L3/locality 的关系”，保留当前默认 `auto` 行为
  - 如果目标是研究 “如何把整机 8 个 NUMA 节点全部压满获得最大吞吐”，另起一套手工绑核实验，不要和本文件结果混在一起
