# 2026-04-05 Qwen3-30B-A3B P3_AttnOnly acc-local-l3 对照观察

> 本文内容已按 `2026-04-09` 最新代码和结果重写，旧代码对应的实验结论不再保留。
> 结果根目录：`test_results/P3_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/prefill-like/global-fixed/batch_16`
> 当前对比口径：`balanced_span4` vs `acc-local-l3_span4`

## 问题

- 在 `NPS1_TP2 + prefill-like + global-fixed + batch=16 + num_query_heads=32 + num_kv_heads=4` 下，修正后的 `acc-local-l3_span4` 相对 `balanced_span4` 是否已经带来稳定收益。

## 方法差异

- `balanced` 仍按全 rank 线程池均衡分工，不按 L3/CCX 拓扑限制 `kv_head`
- `acc-local-l3` 会先按 `(numa_node, socket_id, l3_cache_id)` 把线程划成 subgroup，再把 `kv_head` 绑定到一个或多个 subgroup
- 当前代码已修正 `group_span>1` 的执行口径：同一 `kv_head` 覆盖多个 subgroup 时，attention / reduction work 在这些 subgroup 的联合线程池上做唯一分片，不再重复执行同一批 legacy slot

## 从 runtime log 看执行组织方式

- `balanced` 的 log 更像“一个统一线程池在消费一批 work item”
- 在 `benchmark_balanced.log` 里，可以直接看到：
  - `thread_num=127`
  - `effective_thread_num=59`
  - `actual_kv_head_num=2`
  - `attention_task_num=118`
  - `workitem_group_num=59`
- 这说明在当前 case 下，`balanced` 是把 `2` 个本地 `kv_head` 对应的 attention work 放在一个统一调度口径里做均衡切分

- `acc-local-l3` 的 log 更像“先按 L3 局部域分组，再把每个 kv_head 限定给一组 subgroup”
- 在 `benchmark_acc_local_l3_span4.log` 里，可以直接看到：
  - `thread_num=127`
  - `subgroup_num=8`
  - `group_span=4`
  - `actual_kv_head_num=2`
  - `attention_task_num=118`
  - `kv_head 0: subgroups=[0,1,2,3]`
  - `kv_head 1: subgroups=[4,5,6,7]`
- 这说明在当前 case 下，`acc-local-l3_span4` 不是把所有线程放在一个统一池里抢同一批 work，而是先把线程切成 8 个 locality subgroup，再让 `kv_head 0/1` 分别落到两段不同的 subgroup 区间

- 同一个 log 还显示：
  - `kv_head 0` 的 `covered_thread_num=64`
  - `kv_head 1` 的 `covered_thread_num=63`
  - 各 subgroup 打印出的 `legacy_slots` 是互补分片，例如 `kv_head 0` 在 subgroup `0/1/2/3` 上分别对应 `0-15`、`16-31`、`32-47`、`48-58`
- 这说明当前修复后的 `acc-local-l3_span4` 已经不是旧版本那种“多个 subgroup 重复做同一批工作”，而是“在 locality 约束下，把同一 `kv_head` 的工作分给一组相邻 subgroup 共同完成”

- 所以从顶层看，这两种方法的核心区别不是算子本身变了，而是“谁和谁一起做同一批 attention work”的组织方式变了：
  - `balanced` 追求全线程池的均衡分担
  - `acc-local-l3` 追求先保持共享 K/V 工作集的局部性，再在局部域内部或少量相邻局部域之间分担

## Dry-run latency

| q_len | kv_len | balanced_span4 slowest rank mean (ms) | acc-local-l3_span4 slowest rank mean (ms) | acc 相对 balanced 变化 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 0.074393 | 0.067254 | -9.60% |
| 128 | 128 | 0.189787 | 0.170897 | -9.95% |
| 256 | 256 | 0.447598 | 0.439554 | -1.80% |
| 512 | 512 | 1.487757 | 1.577071 | +6.00% |

## PCM system aggregated

| q_len | kv_len | mode | L3 Access (pti) | L3 Miss (pti) | L3 Miss % | Ave L3 Miss Latency (ns) | Local Memory / I/O % | another CCX in same node % | another CCX in remote node % | Remote Memory / I/O % |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | balanced_span4 | 6.20 | 2.59 | 41.79 | 478.60 | 0.65 | 99.30 | 0.04 | 0.03 |
| 64 | 64 | acc-local-l3_span4 | 0.90 | 0.88 | 97.91 | 154.21 | 0.72 | 98.72 | 0.40 | 0.17 |
| 128 | 128 | balanced_span4 | 4.64 | 1.58 | 34.07 | 428.16 | 0.99 | 98.95 | 0.06 | 0.03 |
| 128 | 128 | acc-local-l3_span4 | - | - | - | - | - | - | - | - |
| 256 | 256 | balanced_span4 | 3.25 | 1.27 | 38.97 | 198.74 | 5.64 | 94.17 | 0.12 | 0.08 |
| 256 | 256 | acc-local-l3_span4 | 1.27 | 0.82 | 64.10 | 151.19 | 4.39 | 95.26 | 0.25 | 0.12 |
| 512 | 512 | balanced_span4 | 1.90 | 0.69 | 36.48 | 155.69 | 36.58 | 63.01 | 0.25 | 0.26 |
| 512 | 512 | acc-local-l3_span4 | 1.27 | 0.43 | 34.17 | 144.42 | 12.82 | 86.52 | 0.46 | 0.24 |

## 分析

- `acc-local-l3_span4` 在 `q64/q128/q256` 上已经快于 `balanced_span4`
- `acc-local-l3_span4` 到 `q512` 时仍慢于 `balanced_span4`
- `q512` 的 PCM 同时显示，`acc-local-l3_span4` 的 `L3 Access (pti)`、`L3 Miss (pti)`、`L3 Miss %`、`Ave L3 Miss Latency` 都优于 `balanced_span4`
- `q512` 的 miss 来源分布也更偏向 `another CCX in same node`，并且 `Local Memory / I/O %` 明显下降：`36.58 -> 12.82`
- 因此当前结果支持这样的判断：修正后的 `acc-local-l3_span4` 已经在部分较短 prefill case 上转化为 latency 收益；但到 `q512` 这一级别时，虽然 locality 指标继续改善，端到端 latency 还没有超过 `balanced_span4`

## 结论

- 不能再沿用旧代码时期“`acc-local-l3` 明显更慢”的结论
- 当前更准确的结论是：
  - 修正后的 `acc-local-l3_span4` 已在 `q64/q128/q256` 上优于 `balanced_span4`
  - 在 `q512` 上，`acc-local-l3_span4` 的 locality 指标更好，但 latency 仍落后 `balanced_span4`
  - 这说明当前实现已经把一部分 locality 改善转成了实际收益，但收益还没有在所有已测长度点上稳定成立

## 边界

- 当前只覆盖 `prefill-like`
- 当前只覆盖 `NPS1_TP2`
- 当前只覆盖 `global-fixed`
- 当前只覆盖 `batch=16`
- 当前只覆盖 `q=kv in {64, 128, 256, 512}`
- `q128_kv128/acc-local-l3_span4` 当前没有 PCM 报告，因此该点只能做 latency 对比，不能做 PCM 对比

## 相关结果

- `test_results/P3_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/2026-04-09_prefill_global_fixed_batch16_balanced_span4_vs_acc_local_l3_span4_summary.md`
