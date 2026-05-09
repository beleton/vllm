# 实验结果解读
- [2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md](./2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md)：汇总 `P3/P4` attention-only 与 CAT 结果，给出 `acc-local-l3` 不适合作为当前 attention 优化主线的证据链
- [2026-04-21_balanced_vs_acc_batch1_qhead32_kvhead16_qlen256_rank0_mapping.html](./2026-04-21_balanced_vs_acc_batch1_qhead32_kvhead16_qlen256_rank0_mapping.html)：`benchmark_balanced_batch1_qhead32_kvhead16_qlen256.log` 与 `benchmark_acc_local_l3_batch1_qhead32_kvhead16_span1_qlen256.log` 的 `rank0` 任务放置对照页，展示 `CCD × kv_head` 矩阵和跨 `CCD` 扩散摘要
- [2026-04-21_共享读写微基准PCM结果分析.md](./2026-04-21_共享读写微基准PCM结果分析.md)：`test_results/microbench` 下共享读/写微基准的 `PCM` 结果，聚焦写共享在同/跨 `CCD` 下的 `IPC/L3/DC` 差异
- [2026-04-15_Qwen3-30B-A3B_P3_AttnOnly_NPS1_TP2_prefill_batch16_summary分析.md](./2026-04-15_Qwen3-30B-A3B_P3_AttnOnly_NPS1_TP2_prefill_batch16_summary分析.md)：`P3_AttnOnly/NPS1_TP2/prefill-like/global-fixed/batch_16` 的 `summary.csv` 精简结论，聚焦 `balanced_span4` 与 `acc-local-l3_span4` 的性能转折和 locality 边界
- [2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md](./2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md)：`Qwen3-30B-A3B` 在本机 `32 MiB` 本地 L3 下的 KV 理论容量与当前实验转折区间
- [2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md](./2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md)：`PD_Test` 下 `prefill/decode` 分相 PCM 的关键观察和边界
- [2026-04-02_Qwen3-30B-A3B_P2_AttnOnly_NPS1_TP2观察.md](./2026-04-02_Qwen3-30B-A3B_P2_AttnOnly_NPS1_TP2观察.md)：`P2_AttnOnly/NPS1_TP2` 首轮 attention-only 结果的关键观察和边界
- [2026-04-05_Qwen3-30B-A3B_P3_AttnOnly_acc-local-l3对照观察.md](./2026-04-05_Qwen3-30B-A3B_P3_AttnOnly_acc-local-l3对照观察.md)：`P3_AttnOnly` 下 `balanced/acc-local-l3` 首轮对照的性能与 locality 边界
