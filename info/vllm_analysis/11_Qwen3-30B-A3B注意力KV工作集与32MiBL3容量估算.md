# 11. Qwen3-30B-A3B 注意力 KV 工作集与 32MiB L3 容量估算

> 更新时间：2026-04-07 16:29:07 +0800  
> 目标：回答“在本机一个 `16` 物理核心、共享 `32 MiB` L3 的局部域里，`Qwen3-30B-A3B` 的 CPU attention 到多长上下文会把 KV 工作集顶出 L3”。  
> 证据类型：模型 `config.json`、本机拓扑记录、当前 `acc-local-l3` 实现说明、`P2/NPS1_TP2` attention-only 结果。

## 结论

- 只按单层 attention 的 `K+V` 容量做理论估算，`Qwen3-30B-A3B` 的单个 `kv_head` 约需要 `512 B / token`。
- 对一个 `32 MiB` 的本地 L3：
  - 若要容纳全模型 `4` 个 `kv_heads` 的 KV，理论上限约是 `16,384 tokens`。
  - 若看 `TP=2` 时单个 rank 本地的 `2` 个 `kv_heads`，理论上限约是 `32,768 tokens`。
  - 若看当前 `acc-local-l3` 更接近的“单个 subgroup 对应 1 个 `kv_head`”口径，理论上限约是 `65,536 tokens`。
- 但对当前常用的 `batch=16`，这些阈值都会按 batch 再除一次：
  - 全 `4 kv_heads`：约 `1,024` token / seq
  - `TP=2` 本地 `2 kv_heads`：约 `2,048` token / seq
  - 单 `kv_head`：约 `4,096` token / seq
- 现有 `P2/NPS1_TP2` attention-only 结果显示，真实转折已经在 `q=512 -> 1024` 区间开始出现，而不是等到理论上限才出现。这说明真实工作集不只是 KV，还包含 scratchpad、Q tile、reduction buffer 和 paged KV 访问带来的碎片化。

## 1. 已知前提

### 1.1 模型配置

`/models/Qwen3-30B-A3B/config.json` 中：

- `head_dim=128`
- `num_attention_heads=32`
- `num_key_value_heads=4`
- `torch_dtype=bfloat16`

证据：[/models/Qwen3-30B-A3B/config.json](/models/Qwen3-30B-A3B/config.json#L10) [/models/Qwen3-30B-A3B/config.json](/models/Qwen3-30B-A3B/config.json#L21) [/models/Qwen3-30B-A3B/config.json](/models/Qwen3-30B-A3B/config.json#L25) [/models/Qwen3-30B-A3B/config.json](/models/Qwen3-30B-A3B/config.json#L33)

### 1.2 本机本地 L3 粒度

当前机器 `lscpu` 记录的口径是：

- 共 `16` 个 L3 instance
- 每个本地 L3 slice 约 `32 MiB`
- 每个 slice 对应 `16` 个物理核心

证据：[/home/zjj/vllm/info/进展.md](/home/zjj/vllm/info/进展.md#L4)

## 2. 单个 token 的 KV 大小

对一个 `kv_head`：

```text
K + V = 2 × head_dim × dtype_bytes
      = 2 × 128 × 2
      = 512 B / token
```

因此：

- `1 kv_head`：`512 B / token`
- `2 kv_heads`：`1024 B / token`
- `4 kv_heads`：`2048 B / token`

这里的估算只算单层 attention 的 KV，不包含：

- Q tile
- logits / partial output
- reduction buffer
- block table / 元数据
- 代码和其他共享数据

## 3. 32MiB L3 的理论容量阈值

### 3.1 不区分 batch，只看总 token 数

对 `32 MiB = 33,554,432 B`：

```text
1 kv_head:  33,554,432 / 512  = 65,536 tokens
2 kv_heads: 33,554,432 / 1024 = 32,768 tokens
4 kv_heads: 33,554,432 / 2048 = 16,384 tokens
```

所以：

- 若一个本地 L3 要承载全模型全部 `4` 个 `kv_heads` 的 KV，理论上限约 `16,384 tokens`
- 若看 `TP=2` 时单 rank 本地 `2` 个 `kv_heads`，理论上限约 `32,768 tokens`
- 若看单个 `kv_head`，理论上限约 `65,536 tokens`

### 3.2 换算成每条序列的上下文长度

如果 batch 内每条序列长度相同，则：

```text
每条序列可容纳的 kv_len ≈ 总 token 上限 / batch_size
```

常见 batch 下的理论阈值如下：

| batch_size | 全 4 kv_heads | TP=2 本地 2 kv_heads | 单 kv_head |
| ---: | ---: | ---: | ---: |
| 1 | 16384 | 32768 | 65536 |
| 8 | 2048 | 4096 | 8192 |
| 16 | 1024 | 2048 | 4096 |
| 32 | 512 | 1024 | 2048 |

## 4. 这些阈值和当前实现的关系

当前 `acc-local-l3` 的实现是：

- 先把线程按 `L3/CCX` 分成 subgroup
- 再把 `kv_head` 按 round-robin 映射到 subgroup
- 每个 subgroup 只处理自己负责的 `kv_head`

证据：[/home/zjj/vllm/info/vllm_analysis/10_CPU_attention_acc_locality新kernel实现说明.md](/home/zjj/vllm/info/vllm_analysis/10_CPU_attention_acc_locality新kernel实现说明.md#L63) [/home/zjj/vllm/info/vllm_analysis/10_CPU_attention_acc_locality新kernel实现说明.md](/home/zjj/vllm/info/vllm_analysis/10_CPU_attention_acc_locality新kernel实现说明.md#L206)

因此，对当前 `acc-local-l3` 来说，最贴近它设计目标的容量口径，不是“4 个 `kv_heads` 全塞进一个 L3”，而更接近：

- 一个 subgroup 的主要共享工作集，接近某个 `kv_head` 对应的 KV

所以在 `acc-local-l3` 的语境下，`32 MiB` L3 对应的单 `kv_head` 理论阈值 `65,536 tokens`，或者在 `batch=16` 下约 `4,096 token / seq`，比“全 `4 kv_heads` 共用一个 L3”的 `1,024 token / seq` 宽松很多。

## 5. 为什么真实转折会更早

上面的数值只是“单层 KV 容量上限”，不是“真实 attention 工作集上限”。真实运行时至少还会占用：

- Q tile
- logits buffer
- partial output / reduction buffer
- scratchpad
- paged KV 访问带来的额外缓存碎片
- 多线程下同一 KV 前缀的重复读取

当前实现说明文档也明确写了，`acc-local-l3` 现在还只是：

- `kv_head -> subgroup` 的最小闭环
- 没有 `group_span > 1`
- 没有长度 heuristic
- 没有重写 legacy scheduler

证据：[/home/zjj/vllm/info/vllm_analysis/10_CPU_attention_acc_locality新kernel实现说明.md](/home/zjj/vllm/info/vllm_analysis/10_CPU_attention_acc_locality新kernel实现说明.md#L293)

所以真实“开始顶出 L3”的点，通常会早于纯 KV 容量公式给出的上限。

## 6. 与现有实验结果对照

当前 `P2/NPS1_TP2` 的 `attention-only` 结果里，`prefill-like/global-fixed/batch_16` 已经能看到明显转折：

- `q=512` 时：
  - `same-node another CCX %=61.15`
  - `local memory / I/O %=38.53`
- `q=1024` 时：
  - `same-node another CCX %=30.70`
  - `local memory / I/O %=68.80`
- `q=2048` 时：
  - `same-node another CCX %=21.95`
  - `local memory / I/O %=77.32`

证据：[/home/zjj/vllm/test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md](/home/zjj/vllm/test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md#L24) [/home/zjj/vllm/test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md](/home/zjj/vllm/test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md#L25) [/home/zjj/vllm/test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md](/home/zjj/vllm/test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md#L26)

这说明：

- 对当前 `TP=2 + batch=16` 的 attention-only `prefill`，真实转折已经在 `512 -> 1024` 区间开始出现
- 这个转折明显早于“单 `kv_head` 工作集”的理论阈值 `4096`
- 也早于“本地 `2 kv_heads`”的理论阈值 `2048`
- 但和“全 `4 kv_heads` 聚合工作集”的 `1024` 非常接近

因此，现阶段更稳妥的解释是：

- 对 `batch=16` 的当前实验口径，`q/kv ≈ 1024` 已经是一个很值得警惕的 L3 容量边界
- 不是因为公式证明“1024 一定溢出”
- 而是因为“理论估算 + 当前 PCM 结果”都指向这个区间正在发生从 `another CCX` 向 `local memory` 的来源切换

## 7. 当前可用的工程判断

若只针对当前机器、当前模型、当前 `batch=16` 的 attention-only `prefill` 讨论，可先采用下面这个工程口径：

- `q/kv <= 512`：更可能仍以 `same-node another CCX` 为主，`acc-local-l3` 更有机会打中靶点
- `q/kv ≈ 1024`：处在明显转折区间，应重点观察 locality 指标是否转向 `local memory / I/O`
- `q/kv >= 2048`：更应假设主瓶颈已明显偏向本地内存，而不是继续假设“只靠收紧 `CCX/L3` 就能解决”

这个判断目前只绑定：

- `Qwen3-30B-A3B`
- 本机 `32 MiB` 本地 L3 slice
- `P2/NPS1_TP2/global-fixed/batch_16`

不能直接外推到其他模型、其他 batch 或其他 TP/NPS 配置。
