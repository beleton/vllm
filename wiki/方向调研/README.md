# 方向调研

- [Chiplet后续研究方向总览.md](./Chiplet后续研究方向总览.md)：attention 路线排除后的新研究方向筛选、优先级、判停条件与推进顺序
- [HNSW向量检索_Chiplet研究方向.md](./HNSW向量检索_Chiplet研究方向.md)：HNSW 图检索的热层复制、冷图分区、远端扩展与 query grouping 设计
- [HNSW_CCD-aware_Dispatcher技术分析报告.md](./HNSW_CCD-aware_Dispatcher技术分析报告.md)：面向多 CCD HNSW 的低成本 query signature、query grouping、路由与负载均衡分析
- [IVF批量检索_Chiplet研究方向.md](./IVF批量检索_Chiplet研究方向.md)：IVF batch 检索的 cluster-aware query grouping 与 per-CCD inverted-list ownership
- [LSM_HTAP_Chiplet研究方向.md](./LSM_HTAP_Chiplet研究方向.md)：LSM compaction 与 HTAP delta merge 的 phase-aware Local/Mixed 切换
- [网络IO_RPC_KV_Chiplet研究方向.md](./网络IO_RPC_KV_Chiplet研究方向.md)：网络 I/O、RPC 与 KV serving 的 queue/worker/mempool 分区和 tail spill
- [AgentToolSandbox_Chiplet研究方向.md](./AgentToolSandbox_Chiplet研究方向.md)：Agent tool sandbox 的共享 OS 路径分片、session/page-cache 亲和与混部隔离
- [Zen平台_TiNA适用性复核与新增场景.md](./Zen平台_TiNA适用性复核与新增场景.md)：复核 TiNA Remote-tier 策略在 AMD Zen 上的适用性，并补充 embedding、GNN、HPC/CFD、EDA/PDES、k-mer 等新候选场景及 CPU 使用形态边界
- [HNSW之外_Chiplet_Locality应用场景研究方向分析.md](./HNSW之外_Chiplet_Locality应用场景研究方向分析.md)：跳出 HNSW similar-query scheduler 后，对 embedding、GNN、Agent、LSM/HTAP、HPC/EDA、RPC/KV 等候选方向的系统性筛选、排序与后续实验路线
- [RAG_Chiplet优化空间调研.md](./RAG_Chiplet优化空间调研.md)：旧版 RAG 管线调研，保留为背景资料
- [Agent_Chiplet优化空间调研.md](./Agent_Chiplet优化空间调研.md)：旧版 Agent 场景调研，保留为背景资料
- [网络IO_DPDK_RPC_KV_Chiplet优化方向研究备忘.md](./网络IO_DPDK_RPC_KV_Chiplet优化方向研究备忘.md)：网络 I/O 子任务研究备忘，保留来源 URL 与候选点
