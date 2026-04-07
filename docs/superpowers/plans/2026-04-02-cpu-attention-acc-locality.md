# CPU Attention ACC Locality Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改动原有 `cpu_attention_with_kv_cache` kernel 代码路径的前提下，新增一套独立的 CPU attention locality-aware kernel / op / builder 路径，使短中长度 `prefill` 能把同一 `kv_head` 的工作限制在更小的 `L3/CCX` 局部域内。

**Architecture:** 保留现有 `cpu_attention_with_kv_cache` / `get_scheduler_metadata` 作为旧路径，不在原 kernel 文件里直接改调度逻辑。新增一套平行实现，例如 `cpu_attention_with_kv_cache_acc_locality` 与对应的新 metadata op，把现有 kernel 作为参考模板复制到新文件中演化；线程拓扑建模、builder 选择、benchmark 开关都围绕新 op 展开，并通过 feature gate 保证默认仍走旧路径。

**Tech Stack:** Python CPU platform/binding, C++ CPU custom op, OpenMP, libnuma, pytest, benchmark helper

---

### Task 1: 建立 `L3/CCX` 级拓扑模型

**Files:**
- Modify: `vllm/platforms/cpu.py`
- Modify: `vllm/v1/worker/cpu_binding.py`
- Test: `tests/v1/worker/test_cpu_binding.py`

- [ ] **Step 1: 先写失败测试，固定新的拓扑字段与分组语义**

在 [tests/v1/worker/test_cpu_binding.py](/home/zjj/vllm/tests/v1/worker/test_cpu_binding.py) 增加两类断言：
- `LogicalCPUInfo` 能表达 `socket_id` 和 `l3_cache_id`
- 给定一组 fake CPU 拓扑时，能把同 NUMA node 内的 CPU 再按 `l3_cache_id` 聚成多个 locality group

- [ ] **Step 2: 运行测试确认当前实现不支持这些字段**

Run: `pytest tests/v1/worker/test_cpu_binding.py -q`
Expected: 新增断言失败，提示 `LogicalCPUInfo` / 分组 helper 缺失

- [ ] **Step 3: 扩展 Python 侧拓扑结构**

在 [cpu.py](/home/zjj/vllm/vllm/platforms/cpu.py)：
- 给 `LogicalCPUInfo` 增加 `socket_id`、`l3_cache_id`
- 把 `lscpu` 采集从 `CPU,CORE,NODE` 扩成至少能恢复 `L3` 身份的字段
- 提供一个稳定 helper，把允许 CPU 列表组织成 `numa -> l3_cache_id -> cpus`

在 [cpu_binding.py](/home/zjj/vllm/vllm/v1/worker/cpu_binding.py)：
- 保留现有 `resolve_local_omp_cpuid()` 行为不变
- 仅新增只读 helper，供 benchmark / dry-run 输出 locality group 摘要

- [ ] **Step 4: 重新运行绑定测试**

Run: `pytest tests/v1/worker/test_cpu_binding.py -q`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add vllm/platforms/cpu.py vllm/v1/worker/cpu_binding.py tests/v1/worker/test_cpu_binding.py
git commit -m "feat: add cpu l3 locality topology helpers"
```

### Task 2: 在 C++ 线程绑定层缓存 locality metadata

**Files:**
- Modify: `csrc/cpu/utils.hpp`
- Modify: `csrc/cpu/utils.cpp`
- Test: `tests/v1/worker/test_cpu_binding.py`

- [ ] **Step 1: 先写失败测试，固定“绑定后可恢复 locality group”这一契约**

在 [tests/v1/worker/test_cpu_binding.py](/home/zjj/vllm/tests/v1/worker/test_cpu_binding.py) 增加 helper 级测试，断言给定 OMP CPU 列表后，能得到与 `l3_cache_id` 对齐的 group 切分结果。

- [ ] **Step 2: 运行测试确认当前 C++ 侧没有这份 metadata**

Run: `pytest tests/v1/worker/test_cpu_binding.py -q`
Expected: FAIL，原因是缺少 locality group 结果或 helper

- [ ] **Step 3: 在绑定函数里缓存进程级 locality 信息**

在 [utils.cpp](/home/zjj/vllm/csrc/cpu/utils.cpp) / [utils.hpp](/home/zjj/vllm/csrc/cpu/utils.hpp)：
- 绑定 OMP 线程时，同时按 CPU id 解析 `l3_cache_id`
- 记录 `omp_thread_idx -> cpu_id -> l3_cache_id`
- 提供只读 getter，供 attention scheduler 读取
- 若拿不到 `l3_cache_id`，显式退回到“每个线程都属于同一 group”的兼容路径

不要改变现有 `init_cpu_threads_env()` 的返回字符串格式；只增加内部 metadata。

- [ ] **Step 4: 跑测试确认兼容旧绑定**

Run: `pytest tests/v1/worker/test_cpu_binding.py -q`
Expected: PASS，且原有 `auto/manual binding` 测试不回归

- [ ] **Step 5: 提交这一小步**

```bash
git add csrc/cpu/utils.hpp csrc/cpu/utils.cpp tests/v1/worker/test_cpu_binding.py
git commit -m "feat: cache cpu thread locality metadata"
```

### Task 3: 搭建平行的新 op / wrapper 路径

**Files:**
- Create: `csrc/cpu/cpu_attn_acc_locality.cpp`
- Create: `csrc/cpu/cpu_attn_acc_locality_impl.hpp`
- Modify: `csrc/cpu/torch_bindings.cpp`
- Modify: `vllm/_custom_ops.py`
- Test: `tests/kernels/attention/test_cpu_attn.py`

- [ ] **Step 1: 先写失败测试，固定新 op 名与旧路径并存**

在 [tests/kernels/attention/test_cpu_attn.py](/home/zjj/vllm/tests/kernels/attention/test_cpu_attn.py) 增加断言：
- 旧 `cpu_attention_with_kv_cache` 路径继续可用
- 新 wrapper，例如 `cpu_attention_with_kv_cache_acc_locality` / `cpu_attn_get_scheduler_metadata_acc_locality`，已经暴露但尚未实现
- 选择新路径时不会回退去调用旧 op 名

- [ ] **Step 2: 运行测试确认当前仓库还没有新 op**

Run: `pytest tests/kernels/attention/test_cpu_attn.py -q`
Expected: FAIL，原因是新 wrapper / op 缺失

- [ ] **Step 3: 新建独立 kernel 入口文件**

新增：
- [cpu_attn_acc_locality.cpp](/home/zjj/vllm/csrc/cpu/cpu_attn_acc_locality.cpp)
- [cpu_attn_acc_locality_impl.hpp](/home/zjj/vllm/csrc/cpu/cpu_attn_acc_locality_impl.hpp)

要求：
- 以现有 [cpu_attn.cpp](/home/zjj/vllm/csrc/cpu/cpu_attn.cpp) / [cpu_attn_impl.hpp](/home/zjj/vllm/csrc/cpu/cpu_attn_impl.hpp) 为模板复制
- 不在原文件中直接加入 locality-aware 逻辑
- 新文件的命名、类型、metadata 结构与旧路径明确区分，避免后续混淆

- [ ] **Step 4: 注册新的 torch op 和 Python wrapper**

在 [torch_bindings.cpp](/home/zjj/vllm/csrc/cpu/torch_bindings.cpp) / [vllm/_custom_ops.py](/home/zjj/vllm/vllm/_custom_ops.py)：
- 新增新 op 定义与 wrapper
- 旧 op 定义完全保留
- wrapper 命名显式带 `acc_locality`，避免误用

- [ ] **Step 5: 重新跑测试，确认旧路径不回归、新路径可解析**

Run: `pytest tests/kernels/attention/test_cpu_attn.py -q`
Expected: 至少旧路径仍 PASS；新路径若先接占位实现，应能进入下一步开发而非导入失败

- [ ] **Step 6: 提交这一小步**

```bash
git add csrc/cpu/cpu_attn_acc_locality.cpp csrc/cpu/cpu_attn_acc_locality_impl.hpp csrc/cpu/torch_bindings.cpp vllm/_custom_ops.py tests/kernels/attention/test_cpu_attn.py
git commit -m "feat: add parallel cpu attention locality op path"
```

### Task 4: 在新 kernel 中实现 locality-aware scheduler metadata

**Files:**
- Modify: `csrc/cpu/cpu_attn_acc_locality_impl.hpp`
- Modify: `csrc/cpu/cpu_attn_acc_locality.cpp`
- Modify: `vllm/v1/attention/backends/cpu_attn.py`
- Test: `tests/kernels/attention/test_cpu_attn.py`

- [ ] **Step 1: 先写失败测试，固定新 metadata 的行为边界**

在 [tests/kernels/attention/test_cpu_attn.py](/home/zjj/vllm/tests/kernels/attention/test_cpu_attn.py) 增加轻量断言：
- 新 metadata op 会为 `kv_head` 选择局部线程子池，而不是默认全线程池
- 旧 `get_scheduler_metadata` 行为保持不变
- 新旧路径在 `enable_kv_split=True/False` 下都不改变数值正确性

- [ ] **Step 2: 运行 CPU attention 测试确认新模式尚不存在**

Run: `pytest tests/kernels/attention/test_cpu_attn.py -q`
Expected: FAIL，原因是 metadata 中没有 locality-aware 调度信息

- [ ] **Step 3: 在 scheduler 输入里增加 feature gate**

在 [cpu_attn_acc_locality.cpp](/home/zjj/vllm/csrc/cpu/cpu_attn_acc_locality.cpp) 和 [cpu_attn.py](/home/zjj/vllm/vllm/v1/attention/backends/cpu_attn.py)：
- builder 新增可选调度模式，例如 `balanced` 与 `acc-local-l3`
- 默认仍走旧 kernel
- 只有显式选中 `acc-local-l3` 时，才调用新 metadata op 和新 attention op

- [ ] **Step 4: 把 workitem 生成从“全线程池一次生成”改成“按子池大小生成，可重复复用”**

在 [cpu_attn_acc_locality_impl.hpp](/home/zjj/vllm/csrc/cpu/cpu_attn_acc_locality_impl.hpp)：
- 抽出当前 `workitems/reduce_workitems` 构造逻辑，使其能接受 `subgroup_thread_num`
- 新增 `kv_head -> subgroup_id` 映射
- metadata 里保存每个 subgroup 的 `workitem` 前缀和 / 有效线程数

关键要求：
- 不改变单个 workitem 的数值语义
- 只改变 workitem 被哪一批线程消费
- 若 `ACC` 数量少于 subgroup 数量，允许多个 subgroup 空闲；不要强行铺满

- [ ] **Step 5: 保持首版策略简单**

首版只实现：
- locality level = `l3`
- group span = `1`
- 不做长度自适应 heuristic

也就是先验证“单个 `ACC` 收敛到一个 `L3/CCX` 子池”这条最小闭环；复杂 heuristic 留到后续实验确认后再加。

- [ ] **Step 6: 跑 CPU attention 数值测试**

Run: `pytest tests/kernels/attention/test_cpu_attn.py -q`
Expected: PASS

- [ ] **Step 7: 提交这一小步**

```bash
git add csrc/cpu/cpu_attn_acc_locality_impl.hpp csrc/cpu/cpu_attn_acc_locality.cpp vllm/v1/attention/backends/cpu_attn.py tests/kernels/attention/test_cpu_attn.py
git commit -m "feat: add locality-aware metadata for new cpu attention kernel"
```

### Task 5: 改新 kernel 的 runtime task 枚举，让 `ACC` 真正只跑在选中的子池上

**Files:**
- Modify: `csrc/cpu/cpu_attn_acc_locality_impl.hpp`
- Test: `tests/kernels/attention/test_cpu_attn.py`

- [ ] **Step 1: 先写失败测试，固定 runtime 任务数量和映射**

在 [tests/kernels/attention/test_cpu_attn.py](/home/zjj/vllm/tests/kernels/attention/test_cpu_attn.py) 新增内部一致性断言：
- locality-aware 模式下，总 task 数不再等于 `actual_kv_head_num * global_effective_thread_num`
- 每个 `kv_head` 只消费其所属 subgroup 的 thread offsets

- [ ] **Step 2: 运行测试确认当前 runtime 仍按全线程池展开**

Run: `pytest tests/kernels/attention/test_cpu_attn.py -q`
Expected: FAIL，原因是新 kernel runtime 仍按全线程池展开

- [ ] **Step 3: 重写 task 枚举与索引**

在 [cpu_attn_acc_locality_impl.hpp](/home/zjj/vllm/csrc/cpu/cpu_attn_acc_locality_impl.hpp)：
- 把 `task_idx -> kv_head_idx / thread_offset` 的映射改为
  `task_idx -> kv_head_idx / subgroup_id / subgroup_thread_offset`
- 只让 `kv_head` 访问其选中的 subgroup workitem 列表
- reduction buffer 的索引继续按 `kv_head` 维持隔离，避免跨 head 冲突

重点检查：
- `split_id/local_split_id` 不要因为 subgroup 重排而失效
- `reduction_split_num` 与 scratchpad size 的计算要按新任务空间复核

- [ ] **Step 4: 跑数值测试与 smoke benchmark**

Run:
- `pytest tests/kernels/attention/test_cpu_attn.py -q`
- `pytest tests/benchmarks/test_cpu_attn_mp.py -q`

Expected: 全部 PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add csrc/cpu/cpu_attn_acc_locality_impl.hpp tests/kernels/attention/test_cpu_attn.py tests/benchmarks/test_cpu_attn_mp.py
git commit -m "feat: restrict new cpu attention accs to local thread subgroups"
```

### Task 6: 补 benchmark 开关、dry-run 可观测性与实验入口

**Files:**
- Modify: `benchmarks/kernels/cpu/benchmark_cpu_attn.py`
- Modify: `benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py`
- Modify: `tests/benchmarks/test_cpu_attn_mp.py`
- Optional Modify: `info/experiment_ops/Qwen3-30B-A3B_attention-only_TP实验步骤.md`

- [ ] **Step 1: 先写失败测试，固定 benchmark 对 locality 模式的接口**

在 [tests/benchmarks/test_cpu_attn_mp.py](/home/zjj/vllm/tests/benchmarks/test_cpu_attn_mp.py) 增加断言：
- CLI 支持 `--attn-locality-mode` 和 `--attn-locality-group-span`
- `dry_run_summary.json` 会写出 locality mode、group span、每个 rank 的 locality groups 摘要

- [ ] **Step 2: 运行 benchmark 测试确认参数尚不存在**

Run: `pytest tests/benchmarks/test_cpu_attn_mp.py -q`
Expected: FAIL

- [ ] **Step 3: 给 benchmark 暴露 feature gate，并把干跑信息写全**

在 [benchmark_cpu_attn.py](/home/zjj/vllm/benchmarks/kernels/cpu/benchmark_cpu_attn.py) / [benchmark_cpu_attn_mp.py](/home/zjj/vllm/benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py)：
- 增加 locality mode / group span 参数
- 进入 worker 前把这些值传到 attention metadata builder 或环境变量
- 显式记录本次选择的是旧 kernel 还是新 kernel
- 在 `dry_run_summary.json` 中记录 rank 绑定 CPU、对应 locality groups、启用模式

- [ ] **Step 4: 跑 benchmark 测试**

Run: `pytest tests/benchmarks/test_cpu_attn_mp.py -q`
Expected: PASS

- [ ] **Step 5: 如需保留给后续实验复用，再补实验文档**

仅在接口稳定后修改 [Qwen3-30B-A3B_attention-only_TP实验步骤.md](/home/zjj/vllm/info/experiment_ops/Qwen3-30B-A3B_attention-only_TP实验步骤.md)，加入 locality-aware 对照命令；不要在接口未定前先写文档。

- [ ] **Step 6: 提交这一小步**

```bash
git add benchmarks/kernels/cpu/benchmark_cpu_attn.py benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py tests/benchmarks/test_cpu_attn_mp.py info/experiment_ops/Qwen3-30B-A3B_attention-only_TP实验步骤.md
git commit -m "feat: expose cpu attention locality benchmarking controls"
```

### Task 7: 验证边界，只做当前证据支持的首轮结论

**Files:**
- Modify: `info/进展.md`
- Modify: `process.txt`
- Output: `test_results/P2_AttnOnly/...`

- [ ] **Step 1: 先跑最小验证矩阵**

优先只跑：
- `NPS1_TP2`
- `global-fixed`
- `batch=16`
- `prefill-like q=128/256/512/1024`

不要一上来扩展到 decode、`per-rank-fixed`、`NPS2_TP4/NPS4_TP8`，先判断 kernel 级 locality 模式是否真的改变 `same-node another CCX %` 与时间。

- [ ] **Step 2: 验证必须同时看时间和来源占比**

至少对比：
- slowest-rank mean latency
- `same-node another CCX %`
- `local memory / I/O %`
- `Ave L3 Miss Latency (ns)`

只有当“时间改善”与“跨 CCX 来源占比下降”同时出现时，才允许写成正向证据。

- [ ] **Step 3: 如果首轮只在短中长度有效，明确把 heuristic 留到下一轮**

不要在首版实现里直接内嵌长度阈值。先拿实测决定是否需要：
- 仅在 `prefill q <= 512` 启用
- 或按 `group span=1/2` 分段切换

- [ ] **Step 4: 更新研究文档时保持证据边界**

在 [进展.md](/home/zjj/vllm/info/进展.md) 与 [process.txt](/home/zjj/vllm/process.txt) 中明确写：
- 哪个时间戳批次
- 哪些 `NPS/TP/batch/q_len/kv_len`
- locality mode 的确切取值
- 哪些场景仍未覆盖

- [ ] **Step 5: 提交这一小步**

```bash
git add info/进展.md process.txt test_results/P2_AttnOnly
git commit -m "docs: record cpu attention locality validation results"
```

## Notes

- 首版不要改原有 [cpu_attn.cpp](/home/zjj/vllm/csrc/cpu/cpu_attn.cpp) 与 [cpu_attn_impl.hpp](/home/zjj/vllm/csrc/cpu/cpu_attn_impl.hpp) 的 kernel 逻辑；新逻辑全部进入平行新文件。
- 首版不要引入“复制 KV 到多个 CCX”这类高内存方案。
- 首版不要默认开启 locality-aware 调度；必须 feature gate。
- 如果新 kernel 文件体积过大，允许继续拆出 `scheduler` / `runtime` 辅助头文件，但不要回头往旧 kernel 文件塞 locality-aware 分支。
- 当前证据只支持把它当成 `prefill` 候选优化方向，不能预设 decode 也会收益；依据见 [info/进展.md](/home/zjj/vllm/info/进展.md#L36) 和 [summary.md](/home/zjj/vllm/test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md#L22)。
