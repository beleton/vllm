# 实验结果解读
- [2026-04-15_Qwen3-30B-A3B_P3_AttnOnly_NPS1_TP2_prefill_batch16_summary分析.md](./2026-04-15_Qwen3-30B-A3B_P3_AttnOnly_NPS1_TP2_prefill_batch16_summary分析.md)：`P3_AttnOnly/NPS1_TP2/prefill-like/global-fixed/batch_16` 的 `summary.csv` 精简结论，聚焦 `balanced_span4` 与 `acc-local-l3_span4` 的性能转折和 locality 边界
- [2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md](./2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md)：`Qwen3-30B-A3B` 在本机 `32 MiB` 本地 L3 下的 KV 理论容量与当前实验转折区间
- [2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md](./2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md)：`PD_Test` 下 `prefill/decode` 分相 PCM 的关键观察和边界
- [2026-04-02_Qwen3-30B-A3B_P2_AttnOnly_NPS1_TP2观察.md](./2026-04-02_Qwen3-30B-A3B_P2_AttnOnly_NPS1_TP2观察.md)：`P2_AttnOnly/NPS1_TP2` 首轮 attention-only 结果的关键观察和边界
- [2026-04-05_Qwen3-30B-A3B_P3_AttnOnly_acc-local-l3对照观察.md](./2026-04-05_Qwen3-30B-A3B_P3_AttnOnly_acc-local-l3对照观察.md)：`P3_AttnOnly` 下 `balanced/acc-local-l3` 首轮对照的性能与 locality 边界
