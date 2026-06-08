# AMDuProf 指标与使用

当前项目最常用的 AMDuProfCLI / AMDuProfPcm 指标语义和常见坑。当前机器：AMD EPYC 9745，family `0x1a`。

## 解释规则

- **PTI**：分母是退休指令，不是 dispatch/issue 指令。
- **DC Fills**：回填到 L1D 的 cache line fill 次数，不是 load/store 指令数。`All/Demand/HwPf/SwPf` 是按触发类型的子集。`From XXX` 表示回填的响应源。
- **L3 Miss Latency From XXX (%)**：延迟占比，不是 miss 次数占比。
- `report-cumulative.csv` 主要读取 `System (Aggregated)` 列。

## 通用指标

| 指标 | 含义 |
| --- | --- |
| CPI (Sys + User) | 总体每条指令平均周期数，越低越好。 |
| IPC (Sys + User) | 总体每周期退休指令数，越高越好。 |
| Eff Freq (MHz) | 有效工作频率。 |
| System time (%) | 内核态时间占比，过高可能说明系统调用/调度开销大。 |
| Utilization (%) | CPU 利用率。 |

## 内存互连指标

| 指标 | 含义 |
| --- | --- |
| Total Mem Bw (GB/s) | 总内存带宽（DF 统计）。 |
| Total Mem RdBw / WrBw (GB/s) | 总内存读/写带宽。 |
| Local Inbound Read Data Bytes (GB/s) | 从本地内存或互连读入的带宽。 |
| Remote Inbound Read Data Bytes (GB/s) | 从远端节点读入的带宽，越低越好。 |
| Remote DRAM Reads % | 远端 DRAM 读占比，越低越好。 |

## L3 与数据来源

`DC Fills` 统计回填到 L1D 的 cache line fill。`Demand DC Fills From XXX` 可近似理解为 demand load 在 L1D miss 后最终从 XXX 返回。

| 指标 | 含义 |
| --- | --- |
| All DC Fills (pti) | 所有 data-cache fill（含 demand、HW prefetch、SW prefetch）。 |
| All Demand DC Fills (pti) | 仅 demand load 触发的 fill。 |
| L3 Access (pti) | 每千条指令的 L3 访问数。 |
| L3 Miss % / L3 Miss (pti) | L3 miss 比例和每千条指令 miss 数。 |
| Ave L3 Miss Latency (ns) | L3 miss 平均延迟。 |
| Demand DC Fills From Local L2 (pti) | demand fill 响应源为本地 L2。越高越好。 |
| Demand DC Fills From Local L3 or different L2 in same CCX (pti) | demand fill 响应源为同 CCX 的 L3/其他核 L2。 |
| Demand DC Fills From another CCX in same node (pti) | demand fill 响应源为同节点其他 CCX。越低越好。 |
| Demand DC Fills From Local Memory or I/O (pti) | demand fill 响应源为本地内存/IO。 |
| Demand DC Fills From Remote memory or I/O (pti) | demand fill 响应源为远端内存/IO。越低越好。 |
| DC Fills From XXX (pti) | 同上但包含所有触发类型（All = Demand + HwPf + SwPf）。 |
| L3 Miss Latency From another CCX in same node (%) | 跨 CCX 路径在总 miss 延迟中的代价占比，不表示访问量占比。 |
| L3 Miss Latency From Local Memory or I/O (%) | 本地内存路径在总 miss 延迟中的占比。 |
| L3 Miss Latency From Remote Memory or I/O (%) | 远端内存路径在总 miss 延迟中的占比。 |

## 跨 CCX 分析口径

研究跨 CCX 访问行为时，优先看访问量，不优先看延迟占比：

1. 主指标：**Demand DC Fills From another CCX in same node (pti)**——直接描述 demand 访问有多少次从同节点其他 CCX 回填。
2. 对照指标：`Demand DC Fills From Local L3 or different L2 in same CCX (pti)`、`Demand DC Fills From Remote memory or I/O (pti)`——判断访问是停在本地 CCX、跨 CCX、还是外溢到更远层级。
3. 代价指标：`L3 Miss Latency From another CCX in same node (%)` + `Ave L3 Miss Latency (ns)`——判断跨 CCX 访问是否真的拖慢执行。

判断"跨 CCX 行为是否变多"看 demand fill 量；判断"是否拖慢执行"再结合延迟占比。

## L2 与预取

| 指标 | 含义 |
| --- | --- |
| L2 Access (pti) / L2 Hit (pti) / L2 Miss (pti) | L2 访问/命中/未命中。 |
| HwPf DC Fills From XXX (pti) | 硬件预取触发的 fill，可按来源拆分。用于判断预取行为是否拉来过多远端数据。 |
| SwPf DC Fills From XXX (pti) | 软件预取触发的 fill，同上。 |

## Topdown

| 指标 | 含义 |
| --- | --- |
| Retiring | 完成有用工作的 slot 占比，越高越好。 |
| Frontend_Bound | 前端供给不足造成的停顿占比。 |
| Backend_Bound.Memory | 访存造成的后端停顿占比。 |
| Bad_Speculation | 错误推测浪费的 slot 占比。 |

## 常用命令

```bash
# 系统信息
AMDuProfCLI info --system

# 列出可用事件
AMDuProfCLI info --list pmu-events

# PCM profile（L3 指标）
AMDuProfPcm profile -m l3 -a -d 140 -I 100 -O <output_dir> -- <workload>

# IBS L3-miss 采样
AMDuProfCLI collect -e event=ibs-op,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp --start-delay <s> --duration <s> \
  --output-dir <dir> -- <workload>
```

## AMDuProfCLI collect 使用

`collect` 负责采集原始 profile 数据。常见输出目录名形如 `AMDuProf-<app>-<profile>_<date>`，目录内通常包含 `session.uprof`、`cpu/*.caperf`、`cpu/*.ri` 等文件。该目录可继续交给 `AMDuProfCLI report` 转换成 `report.csv`、`cpu.db`、`callstack.db`，也可在 GUI 中打开。

### 基本形态

```bash
# 启动目标程序并采集
AMDuProfCLI collect [options] -- <workload> [args...]

# attach 到已有进程
AMDuProfCLI collect [options] -p <pid1,pid2> -d <seconds>

# 系统级采集
AMDuProfCLI collect [options] -a -d <seconds>
```

`<workload>` 前建议加 `--`，避免 workload 自己的参数被 AMDuProfCLI 解析。

### 常用范围参数

| 参数 | 含义 | 备注 |
| --- | --- | --- |
| `-p, --pid <PID,..>` | attach 到已有进程 | 多个 PID 用逗号分隔。适合 vLLM worker、HNSW benchmark 等已启动进程。 |
| `--tid <TID,..>` | attach 到已有线程 | 只采指定线程。 |
| `-a, --system-wide` | 系统级采集 | 不指定时默认采 launched application 或 `-p` 指定进程。 |
| `-c, --cpu <core,..>` | 限定采集 CPU | 支持 `0-3` 这种范围。用于固定 CCD/NUMA 对照。 |
| `--affinity <core-id,..>` | 设置被启动应用的 CPU affinity | 只对 launch application 的 per-process profile 有效。 |
| `--no-inherit` | 不采集子进程 | 默认会继承采集 launched application 的子进程。 |

### 常用时间与输出参数

| 参数 | 含义 | 备注 |
| --- | --- | --- |
| `-d, --duration <seconds>` | 采集时长 | attach、system-wide 或长时间 workload 常用。 |
| `--start-delay <seconds>` | 延迟开始采集 | 用于跳过初始化、warmup、索引加载。 |
| `--start-paused` | 启动后先暂停采集 | 需要应用通过 profile control API 恢复采集。 |
| `-b, --terminate` | 采集结束后终止 launched application | 只终止 launch 进程本身，子进程可能仍在。 |
| `-o, --output-dir <dir>` / `--output-dir <dir>` | 输出目录 | 建议每次实验单独目录。 |
| `-w, --working-dir <dir>` | workload 工作目录 | 默认是被启动程序所在目录。 |
| `-m, --mmap-pages <size>` | perf mmap buffer 大小 | 采样丢失时可增大，例如 `256M`。 |

### `-e, --event` 事件参数

`-e` 可指定 timer、PMU、IBS 或 predefined event。多个事件可重复写多个 `-e`。

```bash
AMDuProfCLI collect \
  -e event=ibs-op,interval=50000,call-graph \
  -e event=RETIRED_INST,interval=250000 \
  -o <dir> -- <workload>
```

`-e` 内部常用 key：

| key | 含义 | 备注 |
| --- | --- | --- |
| `event=<timer|ibs-fetch|ibs-op|PMU-event>` | 事件类型或 PMU 事件名 | PMU 事件可用 `AMDuProfCLI info --list pmu-events` 查询。 |
| `umask=<mask>` | PMU unit mask | predefined event 不需要手写。 |
| `user=<0|1>` | 是否采用户态 | 默认 `1`。 |
| `os=<0|1>` | 是否采内核态 | 默认 `1`。 |
| `cmask=<mask>` | PMU count mask | 范围 `0x0` 到 `0x7f`。 |
| `inv=<0|1>` | 反转 count mask 条件 | PMU 事件用。 |
| `interval=<n>` | 采样间隔 | timer 默认 `1.0 ms`；IBS/PMU 默认 `250000`。间隔越小样本越多，开销和文件越大。 |
| `frequency=<Hz>` | 按频率采样 | 仅 core PMC 事件支持。 |
| `call-graph` | 对该事件采 call graph | 常与 `--call-graph-mode fp` 配合。 |
| `ibsop-count-control=<0|1>` | IBS OP 采样计数方式 | `0` 为 cycle count，`1` 为 dispatch count；默认 `1`。 |
| `ibsop-l3miss=<0|1>` | IBS OP 只保留 L3 miss load/store 样本 | Zen4 及以后支持。`sleep` 这类目标可能没有样本。 |
| `ibsfetch-l3miss=<0|1>` | IBS Fetch 只保留 L3 miss fetch 样本 | Zen4 及以后支持。 |
| `ibsop-ldlat=<cycles>` | 按 data-cache miss latency 过滤 IBS OP | Zen5 及以后支持；值需为 `128..2048` 且为 128 的倍数。 |

### call graph 参数

| 参数 | 含义 | 备注 |
| --- | --- | --- |
| `-g` | 等价于 `--call-graph fp` | 使用 frame pointer 回溯。 |
| `--call-graph <F:N>` | 启用 call stack sampling | `F` 可为 `fp`、`fpo`、`dwarf`；`N` 是每个样本的栈大小。 |
| `--call-graph-mode <fp|fpo|dwarf>` | 指定回溯模式 | 默认 `fp`。Python/C++ 混合栈通常仍以 native 栈为主。 |
| `--call-graph-size <size>` | 每个样本保存的栈大小 | `fpo/dwarf` 模式有效，范围 `16..32768` 字节。 |
| `--call-graph-depth <num>` | 展示回溯深度 | 对 Hotspots/Threading 配置有效。 |

`fp` 开销低，但要求目标二进制保留 frame pointer。`fpo/dwarf` 可补偿 frame pointer omission，但原始数据更大，采集开销更高。

### HNSW/访存分析推荐命令

先不加 L3 miss 过滤，确认采样链路和 call graph 正常：

```bash
AMDuProfCLI collect \
  -e event=ibs-op,interval=500,call-graph \
  --call-graph-mode fp \
  --start-delay <warmup_seconds> \
  --duration <profile_seconds> \
  --output-dir <dir> \
  -- <hnsw_workload>
```

确认 `report.csv` 中有 `IBS_ALL_OPS`、`IBS_LOAD` 等样本后，再采 L3 miss：

```bash
AMDuProfCLI collect \
  -e event=ibs-op,interval=50000,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp \
  --start-delay <warmup_seconds> \
  --duration <profile_seconds> \
  --output-dir <dir> \
  -- <hnsw_workload>
```

若目标是已有进程：

```bash
AMDuProfCLI collect \
  -p <pid> \
  -e event=ibs-op,interval=50000,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp \
  --duration <profile_seconds> \
  --output-dir <dir>
```

## AMDuProfCLI report 使用

`report` 读取 `collect` 生成的目录，转换原始数据并输出 `report.csv`。转换后目录中会新增 `cpu.db`、`callstack.db`、`binaries/`、`sources/` 等文件；这些文件也有助于 GUI 打开。

### 基本形态

```bash
AMDuProfCLI report \
  -i <collect_output_dir> \
  --category cpu \
  --detail \
  --cutoff 0
```

### 常用参数

| 参数 | 含义 | 备注 |
| --- | --- | --- |
| `-i, --input-dir <dir>` | collect 输出目录 | 必填。 |
| `--category <PROFILE>` | 只生成指定类别报告 | 常用 `cpu`。也支持 `mpi`、`openmp`、`os`、`gputrace`、`gpuprof`。 |
| `--detail` | 生成详细报告 | 会输出函数、进程、模块、线程等更细 sections。 |
| `-p, --pid <PID,..>` | 只报告指定 PID | 仅 CPU profile 报告适用。 |
| `-g` | 打印 callstack | 需要采集时已有 callstack 样本。 |
| `--host <host|all>` | 指定 host | 多机/集群采集时使用；单机通常不需要。 |
| `--group-by <process|module|thread>` | 按进程、模块或线程汇总 | 默认 `process`。使用 `--detail` 时该选项可能被忽略。 |
| `--view <view-config>` | 只报告指定 view 的事件 | 事件必须已在输入文件中采集。可用 `AMDuProfCLI info --list view-configs` 查询。 |
| `--sort-by <EVENT>` | 指定排序事件或 metric | 例如按 `event=ibs-op` 或 PMU 事件排序。 |
| `--time-filter <T1:T2>` | 只报告采集开始后 `[T1,T2]` 秒区间 | 用于截取 steady-state。 |
| `--agg-interval <low|medium|high|ms>` | 设置样本聚合间隔 | 导入 GUI 时影响 timeline 粒度和数据库大小。 |
| `--cutoff <n>` | 控制报告条目数量 | 默认 `10`；`--cutoff 0` 输出全部数据。 |
| `--show-percentage` | 输出百分比 | 适合看占比。 |
| `--show-sample-count` | 输出样本数 | 默认开启。 |
| `--show-event-count` | 输出估算事件数 | 不是所有事件都适用。 |
| `--ignore-system-module` | 忽略系统模块样本 | 聚焦应用代码时可用。 |
| `--bin-path <path>` | 补充 binary 路径 | 多次传入。跨机器或拷贝结果后修复符号解析。 |
| `--src-path <path>` | 补充源码路径 | 多次传入。用于源码行归因。 |
| `--symbol-path <path>` | 补充 debug symbol 路径 | 多次传入。 |
| `--disasm` | 输出有样本函数的源码/汇编 | 隐含 `--detail`。 |
| `--disasm-only` | 只输出有样本汇编 | 隐含 `--detail`。 |
| `--disasm-full` | 输出函数完整源码/汇编 | 文件可能较大。 |
| `--disasm-style <att|intel>` | 汇编语法 | 默认 `att`。 |
| `--inline` | 展开 inline 函数 | C/C++ native binary 适用。 |

### 常用 view

| view | 用途 |
| --- | --- |
| `ibs_op_overview` | IBS OP 总览，查看 branch/load/store、L1D miss latency 等。 |
| `ibs_op_ld` | load 来源比例，包含 L2、本地 cache、peer cache、remote cache、本地/远端 DRAM。 |
| `ibs_op_ld_lat` | load miss latency 来源占比，适合判断访存代价来源。 |
| `ibs_op_ls_overview` | load/store 访存模式总览。 |
| `memory` | cache sharing / HITM 相关分析。 |
| `dc_focus` | L1 data cache miss/refill 相关分析，需要对应 PMU 事件。 |
| `all` | 输出所有已采集事件和可计算指标。 |

示例：

```bash
AMDuProfCLI report \
  -i <collect_output_dir> \
  --category cpu \
  --detail \
  --view ibs_op_ld_lat \
  --cutoff 0
```

### GUI 打开前检查

先在 Linux 端运行一次 `report`：

```bash
AMDuProfCLI report -i <collect_output_dir> --category cpu --detail --cutoff 0
```

检查 `report.csv` 的 `HOTTEST PROCESSES`、`HOTTEST FUNCTIONS`、`ALL PROCESSES` 是否有数据行。若只有表头，没有任何进程或函数数据，说明采集没有有效样本；Windows GUI 可能报：

```text
The raw file has no data!
```

常见原因：

- workload 太空，例如 `sleep 20` 基本不会产生 IBS L3 miss 样本。
- `ibsop-l3miss=1` 过滤过强，目标运行期间没有被采到 L3 miss load/store。
- `--start-delay` 跳过了主要执行阶段。
- `-p` attach 到错误 PID，或目标进程在采集开始前已退出。
- 只拷贝了 `cpu/*.caperf`，没有拷贝完整采集目录。

在 Windows GUI 打开 Linux 采集结果时，优先拷贝 Linux 端 `report` 后的完整目录，至少包含：

```text
session.uprof
cpu/
cpu.db
callstack.db
report.csv
binaries/
sources/
metadata/
```

## 常见坑

- 从 PDF 复制命令时容易混入 Unicode 横杠，命令里应统一使用 ASCII `-`。
- `L3 Miss Latency From XXX (%)` 是延迟占比，不要直接解释成"该来源的 miss 次数占比"。
- `--pid/--tid` 只适用于 Core metrics，L3 source latency 只能按硬件域采集，不能按单个线程归属。
- 列事件写法是 `AMDuProfCLI info --list pmu-events`。
- `sleep` 这类空 workload 不适合验证 `ibsop-l3miss=1`。先用不带 L3 miss 过滤的 `event=ibs-op,interval=50000` 验证采样链路。
- `--view` 只能报告已采集事件中包含的指标；采集时没有对应事件，report 阶段不能补出来。
- `--cutoff` 默认只保留部分条目；做归档或排查时优先使用 `--cutoff 0`。
