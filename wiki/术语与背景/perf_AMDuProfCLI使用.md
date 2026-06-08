# perf stat 测量数据来源

```bash
perf stat -x, \
  --delay 10000 \
  -o same_ccd.stat.csv \
  -e cpu/event=0x43,umask=0x01,name=local_l2/ \
  -e cpu/event=0x43,umask=0x02,name=local_ccx_cache/ \
  -e cpu/event=0x43,umask=0x04,name=other_ccx_same_node_cache/ \
  -e cpu/event=0x43,umask=0x08,name=local_dram/ \
  -e cpu/event=0x43,umask=0x10,name=other_node_cache/ \
  -e cpu/event=0x43,umask=0x40,name=remote_dram/ \
  -e cpu/event=0x43,umask=0xdf,name=all_demand_dc_refills/ \
  -- \
  numactl --physcpubind=0-7 --membind=0 \
  python /home/zjj/HNSW/scripts/hnsw_query_bench.py \
    --num-elements 1000000 \
    --query-count 262144 \
    --threads 8 \
    --m 16 \
    --ef-construction 100 \
    --ef 64 \
    --k 10 \
    --index-path /home/zjj/HNSW/results/indexes/hnsw_l2_d128_M16_efc100_n1000000.bin \
    --warmup-iters 0 \
    --min-runtime-s 30 \
    --output-json /home/zjj/abc/perf/same_ccd.json
```

# AMDuProfCLI 测量延迟占比

```bash
# 非 root 采样时降低 perf_event_paranoid
sudo sysctl kernel.perf_event_paranoid=1 

AMDuProfCLI collect \
    -e event=ibs-op,interval=50000 \
    --start-delay 10 \
    --output-dir /home/zjj/abc/AMDuProfCLI \
    numactl --physcpubind=0-7 --membind=0 \
    python /home/zjj/HNSW/scripts/hnsw_query_bench.py \
      --num-elements 1000000 \
      --query-count 262144 \
      --threads 8 \
      --m 16 \
      --ef-construction 100 \
      --ef 64 \
      --k 10 \
      --index-path /home/zjj/HNSW/results/indexes/hnsw_l2_d128_M16_efc100_n1000000.bin \
      --warmup-iters 0 \
      --min-runtime-s 30 \
      --output-json /home/zjj/abc/AMDuProfCLI/same_ccd.amduprof.json

AMDuProfCLI report -i /home/zjj/abc/AMDuProfCLI/AMDuProf-numactl-Custom_May-20-2026_15-25-10 \
    --view ibs_op_ld,ibs_op_ld_lat \
    --show-event-count --cutoff 0
```

# 关键概念

## AMD chiplet 拓扑与 IBS 标签对应关系

IBS 访存层级标签直接对应 AMD 物理拓扑：

| IBS 标签 | 物理含义 | 备注 |
|---|---|---|
| `LOCAL_CACHE` | 同一 **CCX**（即同一 CCD）内的共享 L3 或其他 core 的 L1/L2 | CCX = Core Complex，在 Zen4+ 中等同于 CCD |
| `PEER_CACHE` | 同一 **NUMA node** 内不同 CCX 的 L2/L3 | 跨 CCD 但仍在同一 socket |
| `RMT_CACHE` | 不同 **NUMA node** 上不同 CCX 的 L2/L3 | 跨 socket 访问远端 cache |
| `LOCAL_DRAM` | 本地 NUMA node 的 DRAM | 本地内存控制器 |
| `RMT_DRAM` | 远程 NUMA node 的 DRAM | 跨 socket 访问远端内存 |

**关于 CCX/CCD 的关系**：
- Zen3 及之前：一个 CCD = 2 个 CCX，每个 CCX 有自己的 L3
- Zen4 及之后（含你的 Turin）：一个 CCD = 1 个 CCX，L3 由 CCD 内所有 core 共享
- `diferent CCX` 在这种架构下等同于 `diferent CCD`

## dc_miss_lat 延迟含义

- **不是纯访存延迟**，而是 **use-latency**（使用延迟）
- 从 L1 DC miss 被检测到数据返回 core 的总周期数，包含流水线排队等待时间
- 不能直接等价于"L2 延迟 15 cycle, L3 延迟 50 cycle"等硬件参数

## tag_to_retire 含义

- `IBS_TAG_TO_RETIRE_CYCLES`：从 op 被 IBS 选中采样（tagged）到 op 退休（retire）的总周期数
- 反映了 op 从发出到完成的完整生命周期
- `%IBS_LD_L1_DC_MISS_LAT_CYCLES` = L1 DC miss 延迟占全部 tag-to-retire 周期的百分比，用于评估内存延迟对整体性能的影响

---

# IBS 事件指标

以下指标来自 AMDuProfCLI report 输出的 `MONITORED EVENTS` → `IBS Events` 部分。

## 综合指标

| 指标名 | 含义 |
|---|---|
| `IBS_ALL_OPS` | 所有 IBS OP 采样数。IBS 仅采样退休的 op，不包括因流水线冲刷等而被取消的 op |
| `IBS_TAG_TO_RETIRE_CYCLES` | 所有 IBS op 的 tag-to-retire 周期总和。Tag-to-retire = 从被选中采样到退休的周期数 |
| `IBS_COMP_TO_RETIRE_CYCLES` | 所有 IBS op 的 completion-to-retire 周期总和。从 op 完成到退休的周期数 |

## 访存操作指标

| 指标名 | 含义 |
|---|---|
| `IBS_LOAD_STORE` | 执行 load 或 store 的 IBS op 采样数 |
| `IBS_LOAD` | 执行 load 操作的 IBS op 采样数 |
| `IBS_STORE` | 执行 store 操作的 IBS op 采样数 |
| `IBS_L1_DC_HIT` | load 或 store 命中 L1 data cache 的采样数 |
| `IBS_L1_DC_MISS` | load 或 store 未命中 L1 data cache 的采样数 |
| `IBS_LD_L1_DC_HIT` | load 操作命中 L1 data cache 的采样数 |
| `IBS_LD_L1_DC_MISS` | load 操作未命中 L1 data cache 的采样数 |
| `IBS_ST_L1_DC_HIT` | store 操作命中 L1 data cache 的采样数 |
| `IBS_ST_L1_DC_MISS` | store 操作未命中 L1 data cache 的采样数 |
| `IBS_LD_L2_HIT` | load 操作命中 L2 cache 的采样数 |
| `IBS_LD_L2_MISS` | load 操作未命中 L2 cache 的采样数 |
| `IBS_LOCKED_OP` | 执行 locked 操作（原子操作）的采样数 |
| `IBS_UC_MEM_ACCESS` | 访问 uncacheable (UC) 内存的采样数 |
| `IBS_WC_MEM_ACCESS` | 访问 write-combining (WC) 内存的采样数 |
| `IBS_MISALIGN_ACCESS` | 跨 64 字节边界（非对齐访问）的采样数 |
| `IBS_MAB_HIT` | 命中 MAB（Miss Address Buffer）已有条目的采样数。MAB 跟踪未完成的 DC miss |

## 访存层级指标（核心：数据来源分类）

| 指标名 | 物理含义 |
|---|---|
| `IBS_LD_LOCAL_CACHE_HIT` | load 命中同一 CCX 内共享 L3 或其他 core 的 L1/L2 |
| `IBS_LD_PEER_CACHE_HIT` | load 命中同一 NUMA node 内不同 CCX 的 L2/L3 |
| `IBS_LD_RMT_CACHE_HIT` | load 命中不同 NUMA node 上不同 CCX 的 L2/L3 |
| `IBS_LD_LOCAL_DRAM_HIT` | load 命中本地 NUMA node 的 DRAM |
| `IBS_LD_RMT_DRAM_HIT` | load 命中远程 NUMA node 的 DRAM |
| `IBS_LD_DRAM_HIT` | load 命中 DRAM（不区分本地/远程） |
| `IBS_LD_CACHE_HIT` | load 命中 local 或 remote cache，状态为 Owned (O) |
| `IBS_LD_CACHE_HITM` | load 命中 local 或 remote cache，状态为 Modified (M) |
| `IBS_LD_LOCAL_CACHE_HITM` | load 命中同一 CCX 内 L3/L2，状态为 Modified (M) |
| `IBS_LD_PEER_CACHE_HITM` | load 命中同一 NUMA node 内其他 L3，状态为 Modified (M) |
| `IBS_LD_RMT_CACHE_HITM` | load 命中不同 NUMA node 的其他 L3，状态为 Modified (M) |
| `IBS_LD_NVDIMM_HIT` | load 命中 NVDIMM（非易失性内存） |
| `IBS_LD_LOCAL_NVDIMM_HIT` | load 命中本地 NVDIMM |
| `IBS_LD_RMT_NVDIMM_HIT` | load 命中远程 NVDIMM |
| `IBS_LD_EXT_MEM_HIT` | load 命中扩展内存（Extension Memory） |
| `IBS_LD_PEER_AGENT_MEM` | load 命中 Peer Agent Memory |
| `IBS_LD_NON_MAIN_MEM_HIT` | load 命中 MMIO / PCI 配置空间 / Local APIC（非主内存） |

### IBS 没有显式的 "L3 miss" 指标

IBS 的 data source 机制只告诉你**数据最终从哪里拿到**，不告诉你在哪一层 miss。因此没有 `IBS_LD_L3_MISS` 这样的指标。

**反推 L3 miss**：
- `IBS_LD_LOCAL_DRAM_HIT` + `IBS_LD_RMT_DRAM_HIT` → 必然 L3 miss（最终走 DRAM）
- `IBS_LD_PEER_CACHE_HIT` + `IBS_LD_RMT_CACHE_HIT` → 本地 L3 miss（数据在其他 CCD/socket 的 cache 里）
- `IBS_LD_L2_MISS` - (`LOCAL_CACHE + PEER_CACHE + RMT_CACHE + DRAM`) → L3 miss 的大致上限

**直接过滤 L3 miss 样本**：Zen4+ 支持 `ibsop-l3miss=1`，采集时只保留 L3 miss 的 IBS OP 样本：

```bash
AMDuProfCLI collect \
    -e event=ibs-op,ibsop-l3miss=1,interval=50000 \
    ...
```

**显式计数 L3 miss 次数**：用 PMU 计数器（`perf stat` 方式），事件 `L1_DEMAND_DC_REFILLS_LOCAL_DRAM` (umask=0x08) 和 `L1_DEMAND_DC_REFILLS_REMOTE_DRAM` (umask=0x40) 统计的 refill 源是 DRAM，也等价于 L3 miss 后的 DRAM 访问。

## 访存延迟指标（dc_miss_lat 拆分）

| 指标名 | 含义 |
|---|---|
| `IBS_LD_L1_DC_MISS_LAT` | 所有 load 操作的 L1 DC miss 延迟总和（CPU 周期）。从检测到 L1 miss 到数据返回 core |
| `IBS_LD_L2_HIT_LAT` | load 命中 L2 的延迟总和 |
| `IBS_LD_LOCAL_CACHE_HIT_LAT` | load 命中同一 CCX 内共享 L3 或 L1/L2 的延迟总和 |
| `IBS_LD_PEER_CACHE_HIT_LAT` | load 命中同一 NUMA node 内不同 CCX 的 L2/L3 的延迟总和 |
| `IBS_LD_RMT_CACHE_HIT_LAT` | load 命中不同 NUMA node 上不同 CCX 的 L2/L3 的延迟总和 |
| `IBS_LD_LOCAL_DRAM_HIT_LAT` | load 命中本地 NUMA node DRAM 的延迟总和 |
| `IBS_LD_RMT_DRAM_HIT_LAT` | load 命中远程 NUMA node DRAM 的延迟总和 |
| `IBS_LD_DRAM_HIT_LAT` | load 命中 DRAM 的延迟总和（不区分本地/远程） |
| `IBS_LD_NVDIMM_HIT_LAT` | load 命中 NVDIMM 的延迟总和 |
| `IBS_LD_LOCAL_NVDIMM_HIT_LAT` | load 命中本地 NVDIMM 的延迟总和 |
| `IBS_LD_RMT_NVDIMM_HIT_LAT` | load 命中远程 NVDIMM 的延迟总和 |
| `IBS_LD_EXTN_MEM_HIT_LAT` | load 命中扩展内存的延迟总和 |
| `IBS_LD_LOCAL_EXTN_MEM_HIT_LAT` | load 命中本地扩展内存的延迟总和 |
| `IBS_LD_RMT_EXTN_MEM_HIT_LAT` | load 命中远程扩展内存的延迟总和 |
| `IBS_LD_PEER_AGENT_MEM_HIT_LAT` | load 命中 Peer Agent Memory 的延迟总和 |
| `IBS_LD_LOCAL_PEER_AGENT_MEM_HIT_LAT` | load 命中本地 Peer Agent Memory 的延迟总和 |
| `IBS_LD_RMT_PEER_AGENT_MEM_HIT_LAT` | load 命中远程 Peer Agent Memory 的延迟总和 |
| `IBS_LD_NON_MAIN_MEM_HIT_LAT` | load 命中 MMIO/Config/PCI/APIC 的延迟总和 |

## TLB 指标

| 指标名 | 含义 |
|---|---|
| `IBS_L1_DTLB_HIT` | load/store 初始命中 L1 DTLB 的采样数 |
| `IBS_DTLB_L1M_L2H` | load/store 初始未命中 L1 DTLB 但命中 L2 DTLB 的采样数 |
| `IBS_DTLB_L1M_L2M` | load/store 初始未命中 L1 DTLB 且未命中 L2 DTLB 的采样数 |
| `IBS_L1_DTLB_REFILL_LAT` | L1 DTLB refill 从触发到完成的周期数 |

## 分支指标

| 指标名 | 含义 |
|---|---|
| `IBS_BR` | 退休的分支 op 采样数 |
| `IBS_BR_MISP` | 退休的分支预测错误 op 采样数 |
| `IBS_TAKEN_BR` | 退休的被采用分支 op 采样数 |
| `IBS_TAKEN_BR_MISP` | 退休的被采用但预测错误的分支 op 采样数 |
| `IBS_RET` | 退休的子程序返回 op 采样数（`IBS_BR` 的子集） |
| `IBS_RET_MISP` | 退休的预测错误的子程序返回 op 采样数 |
| `IBS_FUSED_INST_OP` | 被标记的 op 属于 fused 指令对 |
| `IBS_MICROCODE_OP` | 被标记的 op 来自微码 |

## 分支延迟指标

| 指标名 | 含义 |
|---|---|
| `IBS_BR_TAG_TO_RETIRE_CYCLES` | 所有分支 op 的 tag-to-retire 周期总和 |
| `IBS_BR_MISP_TAG_TO_RETIRE_CYCLES` | 所有分支预测错误 op 的 tag-to-retire 周期总和 |
| `IBS_TAKEN_BR_TAG_TO_RETIRE_CYCLES` | 所有被采用分支 op 的 tag-to-retire 周期总和 |
| `IBS_RET_TAG_TO_RETIRE_CYCLES` | 所有返回 op 的 tag-to-retire 周期总和 |
| `IBS_BR_COMP_TO_RETIRE_CYCLES` | 所有分支 op 的 completion-to-retire 周期总和 |
| `IBS_BR_MISP_COMP_TO_RETIRE_CYCLES` | 所有分支预测错误 op 的 completion-to-retire 周期总和 |
| `IBS_TAKEN_BR_COMP_TO_RETIRE_CYCLES` | 所有被采用分支 op 的 completion-to-retire 周期总和 |
| `IBS_RET_COMP_TO_RETIRE_CYCLES` | 所有返回 op 的 completion-to-retire 周期总和 |

## Tag-to-Retire 细分指标

| 指标名 | 含义 |
|---|---|
| `IBS_LD_TAG_TO_RETIRE_CYCLES` | 所有 load op 的 tag-to-retire 周期总和 |
| `IBS_ST_TAG_TO_RETIRE_CYCLES` | 所有 store op 的 tag-to-retire 周期总和 |
| `IBS_LD_ST_TAG_TO_RETIRE_CYCLES` | 所有 load/store op 的 tag-to-retire 周期总和 |
| `IBS_UC_MEM_ACCESS_TAG_TO_RETIRE_CYCLES` | 所有 UC 内存访问 op 的 tag-to-retire 周期总和 |
| `IBS_WC_MEM_ACCESS_TAG_TO_RETIRE_CYCLES` | 所有 WC 内存访问 op 的 tag-to-retire 周期总和 |
| `IBS_MISALIGN_ACCESS_TAG_TO_RETIRE_CYCLES` | 所有非对齐访问 op 的 tag-to-retire 周期总和 |

## 其他指标

| 指标名 | 含义 |
|---|---|
| `IBS_DC_MISS_OPEN_MEM_REQS` | DC fill 被发起时观察到的未完成（in-flight）内存请求数 |
| `IBS_DC_PAGE_SIZE_4K` | DC 访问的微架构 4K 页大小（可能与架构页大小不同） |
| `IBS_DC_PAGE_SIZE_2M` | DC 访问的微架构 2M 页大小 |
| `IBS_DC_PAGE_SIZE_1G` | DC 访问的微架构 1G 页大小 |

---

# 派生指标（Derived Metrics）

以下来自 AMDuProfCLI report 的 `DERIVED METRICS` 部分，由上述 IBS 事件计算得到。

## 综合占比

| 指标名 | 公式 | 含义 |
|---|---|---|
| `%IBS_BR` | `IBS_BR * 100 / IBS_ALL_OPS` | 分支 op 占全部 op 的百分比 |
| `%IBS_LOAD_STORE` | `IBS_LOAD_STORE * 100 / IBS_ALL_OPS` | 访存 op 占全部 op 的百分比 |
| `%IBS_LOAD` | `IBS_LOAD * 100 / IBS_ALL_OPS` | Load op 占全部 op 的百分比 |
| `%IBS_STORE` | `IBS_STORE * 100 / IBS_ALL_OPS` | Store op 占全部 op 的百分比 |

## 分支预测派生指标

| 指标名 | 公式 | 含义 |
|---|---|---|
| `%IBS_BR_MISP` | `IBS_BR_MISP * 100 / IBS_BR` | 分支预测错误率（%） |
| `IBS_BR_MISP_RATE_%` | `IBS_BR_MISP * 100 / IBS_BR` | 分支预测错误率（%），同上 |
| `%IBS_BR_MISP_CYCLES` | `IBS_BR_MISP_TAG_TO_RETIRE_CYCLES * 100 / IBS_TAG_TO_RETIRE_CYCLES` | 分支预测错误浪费的周期占总 tag-to-retire 周期的百分比 |
| `IBS_BR_MISP_PTI` | `IBS_BR_MISP * 1000 / IBS_ALL_OPS` | 每千条 op 中的分支预测错误数 |
| `%IBS_TAKEN_BR` | `IBS_TAKEN_BR * 100 / IBS_ALL_OPS` | 被采用分支占全部 op 的百分比 |
| `%IBS_RET` | `IBS_RET * 100 / IBS_ALL_OPS` | 子程序返回占全部 op 的百分比 |

## 访存命中率派生指标

| 指标名 | 公式 | 含义 |
|---|---|---|
| `IBS_LD_L1_DC_HIT_RATE_%` | `IBS_LD_L1_DC_HIT * 100 / IBS_LOAD` | Load 命中 L1 DC 的百分比 |
| `IBS_LD_L1_DC_MISS_RATE_%` | `IBS_LD_L1_DC_MISS * 100 / IBS_LOAD` | Load 未命中 L1 DC 的百分比 |
| `IBS_LD_L2_HIT_RATE_%` | `IBS_LD_L2_HIT * 100 / IBS_LOAD` | Load 命中 L2 的百分比 |
| `IBS_LD_LOCAL_CACHE_HIT_RATE_%` | `IBS_LD_LOCAL_CACHE_HIT * 100 / IBS_LOAD` | Load 命中同一 CCX 内 L3/L2 的百分比 |
| `IBS_LD_PEER_CACHE_HIT_RATE_%` | `IBS_LD_PEER_CACHE_HIT * 100 / IBS_LOAD` | Load 命中同一 NUMA node 内不同 CCX 的 L2/L3 的百分比 |
| `IBS_LD_RMT_CACHE_HIT_RATE_%` | `IBS_LD_RMT_CACHE_HIT * 100 / IBS_LOAD` | Load 命中不同 NUMA node 上 CCX 的 L2/L3 的百分比 |
| `IBS_LD_LOCAL_DRAM_HIT_RATE_%` | `IBS_LD_LOCAL_DRAM_HIT * 100 / IBS_LOAD` | Load 命中本地 DRAM 的百分比 |
| `IBS_LD_RMT_DRAM_HIT_RATE_%` | `IBS_LD_RMT_DRAM_HIT * 100 / IBS_LOAD` | Load 命中远程 DRAM 的百分比 |
| `IBS_LD_DRAM_HIT_RATE_%` | `IBS_LD_DRAM_HIT * 100 / IBS_LOAD` | Load 命中 DRAM 的百分比（不区分本地/远程） |
| `IBS_LD_NVDIMM_HIT_RATE_%` | `IBS_LD_NVDIMM_HIT * 100 / IBS_LOAD` | Load 命中 NVDIMM 的百分比 |
| `IBS_LD_EXT_MEM_HIT_RATE_%` | `IBS_LD_EXT_MEM_HIT * 100 / IBS_LOAD` | Load 命中扩展内存的百分比 |
| `IBS_LD_PEER_AGENT_MEM_RATE_%` | `IBS_LD_PEER_AGENT_MEM * 100 / IBS_LOAD` | Load 命中 Peer Agent Memory 的百分比 |
| `IBS_LD_NON_MAIN_MEM_HIT_RATE_%` | `IBS_LD_NON_MAIN_MEM_HIT * 100 / IBS_LOAD` | Load 命中非主内存（MMIO/PCI/APIC）的百分比 |
| `IBS_ST_L1_DC_MISS_RATE_%` | `IBS_ST_L1_DC_MISS * 100 / IBS_STORE` | Store 未命中 L1 DC 的百分比 |

## 访存延迟占比派生指标

| 指标名 | 公式 | 含义 |
|---|---|---|
| `IBS_LD_L1_DC_MISS_LAT_AVE` | `IBS_LD_L1_DC_MISS_LAT / IBS_LD_L1_DC_MISS` | Load L1 DC miss 平均延迟（cycle） |
| `%IBS_LD_L1_DC_MISS_LAT_CYCLES` | `IBS_LD_L1_DC_MISS_LAT * 100 / IBS_TAG_TO_RETIRE_CYCLES` | L1 DC miss 延迟占全部 tag-to-retire 周期的百分比 |
| `%IBS_LD_L2_HIT_LAT` | `IBS_LD_L2_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | L2 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_LOCAL_CACHE_HIT_LAT` | `IBS_LD_LOCAL_CACHE_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | 本地 CCX cache 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_PEER_CACHE_HIT_LAT` | `IBS_LD_PEER_CACHE_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | Peer CCX cache 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_RMT_CACHE_HIT_LAT` | `IBS_LD_RMT_CACHE_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | 远程 CCX cache 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_LOCAL_DRAM_HIT_LAT` | `IBS_LD_LOCAL_DRAM_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | 本地 DRAM 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_RMT_DRAM_HIT_LAT` | `IBS_LD_RMT_DRAM_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | 远程 DRAM 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_DRAM_HIT_LAT` | `IBS_LD_DRAM_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | 全部 DRAM 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_EXTN_MEM_HIT_LAT` | `IBS_LD_EXTN_MEM_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | 扩展内存命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_NVDIMM_HIT_LAT` | `IBS_LD_NVDIMM_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | NVDIMM 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_PEER_AGENT_MEM_HIT_LAT` | `IBS_LD_PEER_AGENT_MEM_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | Peer Agent Memory 命中延迟占 L1 miss 延迟的百分比 |
| `%IBS_LD_NON_MAIN_MEM_HIT_LAT` | `IBS_LD_NON_MAIN_MEM_HIT_LAT * 100 / IBS_LD_L1_DC_MISS_LAT` | 非主内存命中延迟占 L1 miss 延迟的百分比 |

## DTLB 派生指标

| 指标名 | 公式 | 含义 |
|---|---|---|
| `%IBS_L1_DTLB_REFILL_LAT_CYCLES` | `IBS_L1_DTLB_REFILL_LAT * 100 / IBS_TAG_TO_RETIRE_CYCLES` | L1 DTLB miss 造成的等待周期占全部 tag-to-retire 周期的百分比 |

## Tag-to-Retire 周期分布派生指标

| 指标名 | 公式 | 含义 |
|---|---|---|
| `%IBS_BR_TAG_TO_RETIRE_CYCLES` | `IBS_BR_TAG_TO_RETIRE_CYCLES * 100 / IBS_TAG_TO_RETIRE_CYCLES` | 分支 op 的 tag-to-retire 周期占比 |
| `%IBS_BR_MISP_TAG_TO_RETIRE_CYCLES` | `IBS_BR_MISP_TAG_TO_RETIRE_CYCLES * 100 / IBS_TAG_TO_RETIRE_CYCLES` | 分支预测错误 op 的 tag-to-retire 周期占比 |
| `%IBS_BR_COMP_TO_RETIRE_CYCLES` | `IBS_BR_COMP_TO_RETIRE_CYCLES * 100 / IBS_COMP_TO_RETIRE_CYCLES` | 分支 op 的 completion-to-retire 周期占比 |
| `%IBS_BR_MISP_COMP_TO_RETIRE_CYCLES` | `IBS_BR_MISP_COMP_TO_RETIRE_CYCLES * 100 / IBS_COMP_TO_RETIRE_CYCLES` | 分支预测错误 op 的 completion-to-retire 周期占比 |

---

# 常用视图速查

AMDuProfCLI 预定义的 `--view` 选项，配合 `report` 使用：

| 视图名 | 适用场景 |
|---|---|
| `ibs_op_overview` | 全面性能概览（分支、访存命中率、平均延迟、tag-to-retire 分布） |
| `ibs_op_ls_overview` | 访存模式概览（load/store 比例、对齐、锁定等） |
| `ibs_op_ld` | **Load 数据来源命中率**（L1 → L2 → LOCAL → PEER → RMT → DRAM） |
| `ibs_op_ld_lat` | **Load 延迟占比拆分**（各层级的延迟占总 L1 miss 延迟的百分比） |
| `ibs_op_ld_ext` | 扩展内存访存分析（NVDIMM、CXL 等） |
| `ibs_op_backend_bottle` | 后端瓶颈分析（cache miss、延迟分布） |
| `ibs_op_branch` | 分支预测分析 |
| `ibs_op_bad_speculation` | 错误投机分析 |
| `ibs_fetch_overall` | 前端取样概览 |
| `ibs_fetch_ic` | I-cache miss 分析 |
| `ibs_fetch_itlb` | ITLB miss 分析 |
| `ibs_fetch_front_bottle` | 前端瓶颈分析 |
| `memory` | Cache 伪共享（False Sharing）检测 |
