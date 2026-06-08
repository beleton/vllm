# perf mem：按进程的访存特征测量

`perf mem` 基于 AMD IBS OP 采样，记录每次内存访问的 **data source**（数据来自哪个缓存层级/本地内存/远端内存/跨 CCD cache），可以按进程、函数、库等维度分组查看。

## 前提

- 内核 ≥ 6.2（IBS 从 uncore PMU 提升为 core PMU，支持 per-process 采样）。
- AMD Zen3+；Zen4+ 的 data source 字段更完整。
- 需要 root 权限（或 `perf_event_paranoid=-1`）。

确认 IBS 可用：

```bash
ls /sys/bus/event_source/devices/ibs_op/
perf list | grep ibs
```

## 录制

```bash
# 对已有进程采样
sudo perf mem record -p <PID> -- sleep 10

# 直接启动程序
sudo perf mem record -- ./your_program arg1 arg2

# 只关注 load，减少数据量
sudo perf mem record -t load -p <PID> -- sleep 10

# 同时记录物理地址（可看 NUMA node 分布）
sudo perf mem record -p -t load -p <PID> -- sleep 10
```

录制后生成 `perf.data`。底层实际使用 `ibs_op//` 事件采样。

## 查看报告

### 按进程 + 数据来源分组（最常用）

```bash
perf mem report -s comm,mem --stdio
```

输出示例：

```
# Samples: 150K of event 'ibs_op//'
# Overhead       Samples  Command/Data Source
# ........  ............  .....................
    45.20%        67800    your_program
      30.10%       20408    L1 hit
      20.40%       13831    L2 hit
      15.60%       10576    L3 hit
      12.80%        8678    Local RAM hit
       8.30%        5627    Remote RAM hit
       3.20%        2169    N/A
    32.10%        48150    other_process
      ...
```

### 按函数 + 数据来源分组（定位热点代码）

```bash
sudo perf mem report -s sym,mem -H --stdio
```

### 按 DSO + 数据来源分组

```bash
sudo perf mem report -s dso,mem -H --stdio
```

### 只看数据来源总体分布

```bash
sudo perf mem report -s mem --stdio
```

### 只看已解析到符号的样本

```bash
sudo perf mem report -U --stdio
```

### 导出 CSV

```bash
sudo perf mem report -s comm,mem -H -x, --stdio > mem_report.csv
```

## Data Source 字段含义

| 字段 | 含义 |
| --- | --- |
| `L1 hit` | L1 数据缓存命中 |
| `L2 hit` | L2 缓存命中 |
| `L3 hit` | 本地 L3 命中 |
| `Local RAM hit` | 本地 NUMA node 的 DRAM |
| `Remote RAM hit` | 远端 NUMA node 的 DRAM（跨 socket） |
| `Remote node, same socket RAM hit` | 同 socket 内不同 CCD 的 DRAM |
| `Remote socket RAM hit` | 跨 socket 的 DRAM |
| `N/A` | IBS 采到的微指令不是访存指令，无 data source |

**跨 CCD 访问直接看**：
- `Remote RAM hit` / `Remote node, same socket RAM hit`——数据来自其他 CCD 的本地 DRAM。
- 如果是 `another CCX cache` 命中的情况（远端 L3 缓存命中），IBS data source 会报告为 cache hit 且带有 remote 标记。具体字段名取决于内核版本和 perf 版本，较新版本可能区分出 `Remote cache hit`。

## 常用分析维度

### 1. 看跨 CCD 访问占比

```bash
sudo perf mem report -s comm,mem -H --stdio | grep -E "Remote|Command"
```

结合 `L3 hit`、`Local RAM hit`、`Remote RAM hit` 的比例，判断进程的访存是否大量落在本 CCD 之外。

### 2. 定位哪些函数产生跨 CCD 访问

```bash
sudo perf mem report -s sym,mem -H --stdio | grep -B 1 "Remote"
```

### 3. CPU 绑定后测量（减少系统噪声）

```bash
# 把进程绑定到特定核心
taskset -c 0-7 ./your_program &
PID=$!

# 只在这些核心上采样
sudo perf mem record -C 0-7 -p $PID -- sleep 10
sudo perf mem report -s mem --stdio
```

### 4. 与 IBS 原始事件结合

`perf mem` 底层用的就是 `ibs_op`，也可以用原始 IBS 事件获得更细粒度的控制：

```bash
# 只采样 L3 miss 的 load
sudo perf record -e ibs_op/l3missonly=1/ -p <PID> -- sleep 10
sudo perf report --stdio
```

## 补充：libpfm + Core PMU 事件按进程计数（CHARM 方案）

`perf mem` 是**采样**方式。另一种方式是**计数**——直接读取 core PMU 的 DC Fills 事件计数器，这也是 CHARM 论文采用的方法。

### 关键区分：Core PMU vs Uncore PMU

| PMU 类型 | 示例事件 | 按进程？ | 说明 |
| --- | --- | --- | --- |
| Core PMU | `ls_dc_fills_from_sys.*` | ✅ | PMC 计数器，支持 `perf_event_open(pid, -1)` |
| Core PMU | `l3_lookup_state`, `l3_request` | ✅ | 部分 L3 事件也在 core PMU 上，可以按进程 |
| Uncore L3 PMU (`amd_l3`) | `l3_xi_sampled_latency.*` | ❌ | 只能 `-a` system-wide |
| Uncore DF PMU (`amd_df`) | `dram_channel_data_controller_*` | ❌ | 只能 `-a` system-wide |

**AMDuProfPcm 的 `L3 Miss Latency From XXX` 不能按进程，是因为它用的是 `amd_l3` uncore PMU，而不是因为 DC Fills 事件本身不行。**

### CHARM 的方案：DC Fills 计数

CHARM 论文用的是 **Core PMU 的 DC Fills 事件**，不是 uncore L3 PMU：

> "CHARM tracks chiplet cache fill rates using **ANY_DATA_CACHE_FILLS_FROM_SYSTEM** on AMD systems to distinguish fills from on-chip (intra-CCX), on-die (inter-CCX), and remote memory sources (inter-NUMA)."

这和你之前在 AMDuProfPcm 中用的 `Demand DC Fills From another CCX in same node (pti)` 是**同一类事件**（`LsDcFillsFromSys` 系列）。区别在于：

- AMDuProfPcm 不暴露这些 core 事件的 per-thread 模式（工具限制，非硬件限制）
- 通过 `perf_event_open()` / libpfm 直接读取，可以传 `pid` 参数实现按进程

### 具体事件

AMD Zen3+（Family 19h）上，`LsDcFillsFromSys`（PMCx0C0）的子事件：

| 事件 | UnitMask | 含义 |
| --- | --- | --- |
| `ls_dc_fills_from_sys.local_ccx` | 0x01 | 响应来自同 CCX 的 L3/其他核 L2 |
| `ls_dc_fills_from_sys.near_cache` | 0x02 | 响应来自同节点其他 CCX cache |
| `ls_dc_fills_from_sys.dram_io_near` | 0x04 | 响应来自本地 DRAM/IO |
| `ls_dc_fills_from_sys.dram_io_far` | 0x08 | 响应来自远端 DRAM/IO |
| `ls_dc_fills_from_sys.any` | 0x1F | 全部来源 |

`near_cache` 就是跨 CCD 来源。

### 按进程计数 vs 按进程采样

| | `perf mem` (IBS 采样) | libpfm + DC Fills (计数) |
| --- | --- | --- |
| 原理 | IBS 按周期采样访存指令 | PMC 硬件计数器累加 fill 事件 |
| 输出 | 采样比例分布（非精确计数） | 精确事件计数 |
| 开销 | 采样周期内的少量 overhead | 几乎零开销（只读寄存器） |
| Data Source | IBS data source 标签（完整层级） | Fill 事件来源（L1D fill 路径） |
| 能区分跨 CCD | ✅ `Remote RAM hit` | ✅ `near_cache` fill |
| 能区分到函数 | ✅ `-s sym` | ❌ 只有计数 |

两者互补：DC Fills 计数给精确总量和比例；IBS 采样定位到具体代码。

### 对 CHARM 的局限性分析

论文自己也指出 DC Fills 计数器 **"limited in granularity"**：

1. **只覆盖 L1D fill 路径**：数据已经在 L2/L3 命中的访问如果不需要回填 L1D，就不会触发 DC Fill。SW prefetch、write-combining 等路径也可能不触发 fill。
2. **fill 来源 ≠ 请求目的地**：L1D miss 请求发往 L2/L3，但最终 fill 的数据可能来自另一个 CCX 的 cache 或 DRAM。`near_cache` 表示"数据从另一个 CCX 的 cache 拿到了"，但如果脏数据先被写回 DRAM、再由 DRAM 回填，这条路径会落到 `dram_io_near`，丢失了"数据原本在另一个 CCX"的信息。
3. **不区分 demand vs prefetch**：`ls_dc_fills_from_sys` 的 all 版本包含 HW prefetch 触发的 fill。需要用 demand 子事件（如 `ls_demand_data_cache_fills_from_sys.*`）来单独看 demand load 的跨 CCD 行为，对应 AMDuProfPcm 中的 `Demand DC Fills From XXX`。

### 用 perf stat 直接验证

不需要写代码，可以直接用 `perf stat` 按进程验证这些事件：

```bash
# 查看你机器上可用的 ls_dc_fills 事件
perf list | grep -i "ls_dc_fills\|ls_.*fills"

# 按进程计数（-p <PID>）
sudo perf stat -e ls_dc_fills_from_sys.near_cache,\
ls_dc_fills_from_sys.local_ccx,\
ls_dc_fills_from_sys.dram_io_near,\
ls_dc_fills_from_sys.dram_io_far \
  -p <PID> -- sleep 10

# 用 --per-thread 看线程级别
sudo perf stat --per-thread \
  -e ls_dc_fills_from_sys.near_cache,\
ls_dc_fills_from_sys.local_ccx \
  -p <PID> -- sleep 10
```

### libpfm 编程方式（概要）

```c
#include <perfmon/pfmlib.h>
#include <linux/perf_event.h>

// 1. 初始化
pfm_initialize();

// 2. 将 AMD 事件名编码为 perf_event_attr
perf_event_attr_t attr;
memset(&attr, 0, sizeof(attr));
pfm_perf_encode_arg_t arg = { .attr = &attr, .size = sizeof(arg) };
arg.fstr = "ls_dc_fills_from_sys:near_cache";  // 跨 CCD
pfm_get_os_event_encoding(&arg);

attr.disabled = 1;
attr.exclude_kernel = 0;

// 3. 按进程打开（pid=target_pid, cpu=-1）
int fd = syscall(SYS_perf_event_open, &attr, target_pid, -1, -1, 0);

// 4. 使能 → 运行被测代码 → 读取
ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
// ... 被测代码运行 ...
ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

uint64_t count;
read(fd, &count, sizeof(count));
```

CHARM 具体如何封装可参考 [libpfm4 perf_examples/task.c](https://sourceforge.net/p/perfmon2/libpfm4/ci/master/tree/perf_examples/task.c)。

## 边界

- **采样不是全量**：IBS 按固定周期采样（默认约 500k 次/秒），结果反映的是比例分布，不是精确计数。
- **N/A 比例高是正常的**：IBS 会采样所有微指令（不仅是访存），其中 ~70% 可能是非访存指令。可以通过 `-t load` 参数或 `l3missonly=1` 过滤，但不能完全消除。此问题在较新内核（≥ 6.13）通过 `swfilt` 的 `ldop/stop` 过滤有所改善。
- **`Remote RAM hit` 的分辨率**：在一些内核版本上只区分 `Local RAM hit` vs `Remote RAM hit`，不进一步拆分 `same socket` vs `remote socket`。跨 CCD 访问被归入 `Remote RAM hit`，也可能包含同 socket 跨 CCD 和跨 socket 两种情况。
- **与 AMDuProfPcm 互补**：`perf mem` 按进程归属（AMDuProfPcm 的 L3 source latency 只能按硬件域聚合），但 AMDuProfPcm 的 DF 级别跨 CCD 计数器更精细。两者联合使用效果最好。

## 参考命令速查

```bash
# system-wide 快速看总体内存层级分布
sudo perf mem record -a -- sleep 5
sudo perf mem report -s mem --stdio

# 对特定进程，按进程+来源
sudo perf mem record -p <PID> -- sleep 10
sudo perf mem report -s comm,mem -H --stdio

# 定位函数热点
sudo perf mem report -s sym,mem -H --stdio

# 导出 CSV
sudo perf mem report -s comm,mem -H -x, --stdio > report.csv
```