# vllm_analysis 索引
- `01_vllm_serve启动到server_ready.md` 启动链路的 API server 和 Engine/worker 握手顺序。
- `02_CPU_worker初始化与绑核.md` CPU worker 初始化、自动/手动绑核以及 NUMA 归属策略。
- `03_prefill_decode执行分流.md` prefill/decode 任务如何在 CPU 里分流并送入 attention。
- `04_TP_group与进程间通信.md` TP group 建立、rank 映射，数据面用 gloo、控制面用 MessageQueue，SHM 保留为可选实现。
- `05_CPU关键算子路径.md` attention/linear/matmul/moe 等算子在 CPU 端走向的关键源码路径与线程任务划分。
- `06_CPU环境变量与生效路径.md` CPU backend 环境变量和 runtime 设置在哪些路径生效、影响哪些组件。
- `07_benchmarks与tests筛选.md` vllm bench serve、kernel 微基准和 tests 入口的选择逻辑与先后顺序。
- `08_CPU_attention计算切分、数据放置与Chiplet优化借鉴.md` attention 计算切分、KV 数据布置和可能借鉴的 Chiplet 优化思路。
- `09_CPU_attention并行计算过程_线程任务workitem与tile.md` attention 并行计算的 task/workitem/tile 组织，线程实际参与量。
