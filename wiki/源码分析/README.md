# 源码分析
- [vllm_serve启动到server_ready.md](./vllm_serve启动到server_ready.md)：`vllm serve` 从 CLI 到 API server ready 的启动链路和关键分叉点
- [CPU_worker初始化与绑核.md](./CPU_worker初始化与绑核.md)：CPU worker 的自动/手动绑核规则、OMP 线程数来源和 NUMA 内存策略生效点
- [prefill_decode执行分流.md](./prefill_decode执行分流.md)：CPU attention 路径里 prefill / decode 的共用主算子、metadata 差异和 split-KV 分叉点
- [TP_group与进程间通信.md](./TP_group与进程间通信.md)：CPU backend 下 TP group 的实际通信路径、控制面/数据面分工和常见 collective 落点
- [CPU关键算子路径.md](./CPU关键算子路径.md)：attention、linear、matmul、MoE 的 CPU 主路径与线程级工作划分入口
- [CPU_attention并行计算过程_线程任务workitem与tile.md](./CPU_attention并行计算过程_线程任务workitem与tile.md)：CPU attention 的真实线程、runtime task、workitem 和 tile 是怎样展开的
- [CPU环境变量与生效路径.md](./CPU环境变量与生效路径.md)：CPU backend 关键环境变量、运行时自动写入变量和主要生效位置
- [CPU_attention_acc_locality新kernel实现说明.md](./CPU_attention_acc_locality新kernel实现说明.md)：`acc-local-l3` 新路径的 metadata、subgroup 和 runtime 任务过滤机制
- [CPU_attention代码分析.md](./CPU_attention代码分析.md)：CPU attention 源码分析入口；当前先记录 `dtype/head_dim/isa` 三层 dispatch 与模板实例化路径
