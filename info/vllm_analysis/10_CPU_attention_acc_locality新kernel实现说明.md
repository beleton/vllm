# 10. CPU attention `acc-local-l3` 新 kernel 实现说明

## 1. 背景与约束

这次实现的目标，不是直接改原有 `cpu_attention_with_kv_cache`，而是在它旁边新增一条 `acc-local-l3` 路径，把“共享同一组 `K/V` 的 attention 计算尽量限制在同一个 `L3/CCX` 线程子池内”这个思路先落成可运行版本。

这个约束已经体现在源码结构里：
- 旧 kernel 和旧 metadata 入口仍保留在 `csrc/cpu/cpu_attn_impl.hpp`、`csrc/cpu/torch_bindings.cpp`
- 新路径单独放在 `csrc/cpu/cpu_attn_acc_locality.cpp` 和 `csrc/cpu/cpu_attn_acc_locality_impl.hpp`
- Python 侧通过 `locality_mode` 显式切换，默认还是 `balanced`，见 `vllm/v1/attention/backends/cpu_attn.py`

所以这次实现本质上是“在 legacy kernel 外层包一层 locality-aware 调度”，而不是重写 attention 数学本身。

## 2. 整体调用链

当前 `acc-local-l3` 的完整调用链如下：

1. Python metadata builder 在 `vllm/v1/attention/backends/cpu_attn.py` 里解析 `VLLM_CPU_ATTN_LOCALITY_MODE`，并通过 `_get_cpu_attn_ops()` 选择新旧 op。
2. 如果 mode 是 `acc-local-l3`，metadata 构造走 `ops.cpu_attn_get_scheduler_metadata_acc_locality()`，其 Python wrapper 在 `vllm/_custom_ops.py`，C++ 注册在 `csrc/cpu/torch_bindings.cpp`。
3. C++ 入口 `get_scheduler_metadata_acc_locality()` 位于 `csrc/cpu/cpu_attn_acc_locality.cpp`，先根据 ISA/head_dim 推出 `MaxQHeadNumPerIteration`，再调用 `cpu_attention_acc_locality::build_scheduler_metadata()`。
4. forward 阶段同样通过 `_get_cpu_attn_ops()` 选择 `ops.cpu_attention_with_kv_cache_acc_locality()`，最终落到 `csrc/cpu/cpu_attn_acc_locality.cpp` 里的 `cpu_attention_with_kv_cache_acc_locality()`。
5. 真正执行循环由 `csrc/cpu/cpu_attn_acc_locality_impl.hpp` 里的 `AttentionMainLoop` 完成。

这条链路有两个关键点：
- Python 侧只负责“选择哪条路径”，不做 locality 调度决策。
- C++ 新 kernel 仍复用 legacy attention 实现和 tile 计算逻辑，变化集中在 metadata 布局和线程可见任务集合。

## 3. 设计思路

### 3.1 不重写 legacy metadata，而是在外层加一层 header

`csrc/cpu/cpu_attn_acc_locality_impl.hpp` 里的 `AttentionMetadata` 不是从零重新定义全部 scheduler 信息，而是采用：

- 前面放一个新的 locality header
- 后面原样嵌入 legacy `cpu_attention::AttentionMetadata`

对应字段里最关键的是：
- `legacy_metadata_offset`
- `legacy_metadata_size`
- `subgroup_num`
- `subgroup_thread_num[]`
- `thread_to_group_id[]`
- `thread_to_local_offset[]`
- `kv_head_to_subgroup[]`

这样做的原因很直接：
- legacy metadata 里已经包含完整的 workitem/reduction 切分结果，直接复用风险最低
- 新路径只需要决定“哪些线程可以消费哪些 legacy thread slots”，不必复制整套 scheduler 算法
- 新旧结果更容易对齐，测试时也更容易验证数值一致性

### 3.2 locality 粒度选 `L3/CCX`，不是 NUMA node

这次没有把线程池只按 NUMA node 划分，而是更细到 `socket_id + l3_cache_id + numa_node` 这个组合键。

证据在 `csrc/cpu/utils.cpp`：
- `get_socket_id_for_cpu()` 读 `/sys/devices/system/cpu/cpu*/topology/physical_package_id`
- `get_l3_cache_id_for_cpu()` 读 `/sys/devices/system/cpu/cpu*/cache/index3/id`
- `get_numa_node_for_cpu()` 通过 `numa_node_of_cpu()`
- `build_locality_groups()` 用这三个值作为 key 组装 `ThreadLocalityGroupInfo`

也就是说，当前 subgroup 的物理语义是“同一个 NUMA 节点下、同一个 socket、同一个 L3 cache id 的线程集合”。对于 AMD Chiplet 机器，这个分组更接近我们需要的 `CCX/L3` 局部域。

### 3.3 ACC 映射先做最小闭环：`kv_head -> subgroup`

当前版本没有实现复杂 heuristic，也没有做 `group_span > 1`。真正落地的是最小可运行版本：

- 先从 legacy kernel 推出 `actual_kv_head_num`
- 把每个 `kv_head_idx` 按 round-robin 映射到一个 subgroup
- 每个 subgroup 只让自己的线程去消费该 subgroup 名下 `kv_head` 对应的任务

具体代码在 `build_scheduler_metadata()`：

```cpp
for (int32_t kv_head_idx = 0; kv_head_idx < actual_kv_head_num; ++kv_head_idx) {
  const int32_t subgroup_id =
      metadata->subgroup_num > 0 ? (kv_head_idx % metadata->subgroup_num) : 0;
  metadata->kv_head_to_subgroup[kv_head_idx] = subgroup_id;
}
```

这里的 round-robin 不是最终策略，但它先满足了两个条件：
- 同一 `kv_head` 不再扩散到全线程池
- 各 subgroup 间大体均匀分头，避免一开始就出现明显偏斜

## 4. 线程 locality 信息是怎么来的

### 4.1 Python 绑核阶段只负责给出 CPU 列表

attention benchmark 和 worker 侧沿用已有绑核逻辑，例如：
- `vllm/v1/worker/cpu_binding.py`
- `benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py`

这部分决定了每个 rank 的 `OMP` 线程绑定到哪些逻辑核，但真正把这些逻辑核解释成 `L3 subgroup` 的，是 C++ utils 层。

### 4.2 `ThreadLocalityManager` 把 OMP thread id 变成 subgroup id

`csrc/cpu/utils.hpp` 里新增了 `ThreadLocalityManager`，主要职责是缓存：
- `groups_`
- `thread_to_group_id_`
- `thread_to_local_offset_`

在 `init_cpu_threads_env()` 里，代码会在设置 affinity 之前先调用：

```cpp
cpu_utils::ThreadLocalityManager::get_thread_locality_manager()
    ->set_thread_cpu_ids(omp_cpu_ids);
```

随后 `set_thread_cpu_ids()` 会基于 `build_locality_groups()` 的结果，把第 `i` 个 OMP 线程映射到：
- 所属 subgroup
- subgroup 内本地偏移 `local_offset`

这样新 kernel 在运行时就不需要再重新读 sysfs，也不需要再推断线程绑定关系，直接读缓存即可。

### 4.3 额外加了一个可观测性入口

为了让 benchmark 和测试能直接看到 runtime 理解出来的 subgroup，`csrc/cpu/utils.cpp` 还新增了：

- `describe_cpu_locality_groups(cpu_ids)`

它通过 `torch.ops._C_utils.describe_cpu_locality_groups()` 暴露到 Python，当前已经被 `benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py` 用来生成 `rank_results[*].locality_groups`。

这个接口只做“描述当前 CPU 列表会被分成哪些 locality groups”，不参与 attention 调度本身。

## 5. 新 metadata 是怎么构造的

`build_scheduler_metadata()` 是整个新路径最关键的函数，逻辑可以分成 5 步。

### 5.1 先调用 legacy metadata builder

第一步不是自己算 task，而是直接调用原有的：

```cpp
at::Tensor legacy_metadata_tensor = ::get_scheduler_metadata(...);
```

这一步返回的就是原有 kernel 使用的完整 metadata，包括：
- `workitem_groups_ptr`
- `reduction_items_ptr`
- `effective_thread_num`
- `reduction_item_num`

### 5.2 分配新的大块 metadata buffer

接着新代码分配一个 `int8` tensor，把：
- 新 header
- legacy metadata blob

拼在一起。偏移量通过 `round_up<64>()` 做 64B 对齐，避免 header 和 legacy metadata 之间发生非对齐访问。

### 5.3 根据当前线程绑定结果建立 subgroup

`init_thread_locality_mapping()` 会从 `ThreadLocalityManager` 读取分组信息。

如果 runtime 没有缓存到有效分组，或者分组线程数和实际 `omp_get_max_threads()` 不一致，就自动回退到：
- `subgroup_num = 1`
- 所有线程都属于一个默认 subgroup

这部分保证了新 kernel 即使在没有 locality 信息时也能退化成正确路径，而不是直接崩掉。

### 5.4 嵌入 legacy metadata 后修正其中的内部指针

因为 legacy metadata 被拷贝到了新 buffer 的后半段，它里面原本依赖相对布局的指针需要重新指到新地址：

- `workitem_groups_ptr`
- `reduction_items_ptr`

这些修正在 `build_scheduler_metadata()` 里显式完成，否则 runtime 读取 workitem 时会访问错地址。

### 5.5 统计新路径下的任务规模

新 header 里还维护了：
- `legacy_attention_task_num`
- `attention_task_num`
- `reduction_task_num`

其中 `attention_task_num` 和 `reduction_task_num` 不是简单照搬 legacy 值，而是按“每个 `kv_head` 只占自己 subgroup 内可见线程数”的口径重新估算。这个统计主要用于调试和后续可观测性，不直接驱动数学计算。

## 6. runtime 是怎么把任务限制到 subgroup 内的

### 6.1 仍然复用 legacy mainloop 的大部分计算逻辑

`AttentionMainLoop` 继承自 legacy 的 `cpu_attention::AttentionMainLoop<attention_impl_t>`。因此下面这些逻辑没有改：
- Q/K/V tile 的拷贝和对齐
- logits / partial output / max / sum buffer 的组织
- softmax / reduction 的数值流程
- ISA/head_dim dispatch

这意味着新 kernel 主要改的是“线程看见哪些任务”，不是“单个任务怎么计算”。

### 6.2 每个线程先找到自己的 subgroup 身份

在 `operator()` 里，每个 `thread_id` 会先读：

```cpp
const int32_t subgroup_id = metadata.thread_to_group_id[thread_id];
const int32_t local_offset = metadata.thread_to_local_offset[thread_id];
const int32_t subgroup_thread_num = metadata.subgroup_thread_num[subgroup_id];
```

这三个值分别回答：
- 我属于哪个 subgroup
- 我在 subgroup 内是第几个线程
- 这个 subgroup 总共有多少线程

### 6.3 通过 `kv_head_to_subgroup` 做第一层过滤

每个线程不会再像 legacy 路径那样遍历自己在全局线程池中的所有 head 任务，而是先检查：
- 当前 `kv_head_idx` 是否映射到我的 subgroup

只有匹配的 `kv_head`，当前线程才会参与处理。

### 6.4 通过 `local_offset` 映射到 legacy thread slot

新路径没有重算 legacy `workitem_groups` 的排布，而是把 subgroup 内线程的本地编号，映射回 legacy 的前若干个 thread slot：

- subgroup 第 0 个线程使用 legacy slot 0
- subgroup 第 1 个线程使用 legacy slot 1
- ...
- 最多只使用 `min(subgroup_thread_num, legacy_effective_thread_num)` 个 legacy slots

对应代码体现在：
- 读取 `cu_workitem_num_per_thread[legacy_thread_offset]`
- 从 `workitem_groups + cu_workitem_num_per_thread[legacy_thread_offset]` 取该 slot 的任务组

因此可以把这套 runtime 理解成：
- legacy scheduler 仍负责定义“一个 thread slot 应该执行哪些 tile/workitem”
- 新 kernel 只是在每个 subgroup 内复用这套 slot 定义，并把它限制到该 subgroup 负责的 `kv_head`

### 6.5 reduction 初始化也保持 legacy 结构

新 kernel 对 reduction buffer 的初始化仍沿用 legacy 的 scratchpad/reduction 结构，只是线程实际参与的 head 范围现在受 subgroup 过滤。也因此当前版本能较稳地保持和旧 kernel 的数值一致性。

## 7. Python 与 benchmark 侧怎么接入

### 7.1 backend 侧只暴露一个 mode 开关

`vllm/v1/attention/backends/cpu_attn.py` 里新增了：
- `_CPU_ATTN_LOCALITY_MODE_ENV = "VLLM_CPU_ATTN_LOCALITY_MODE"`
- `_CPU_ATTN_LOCALITY_MODES = {"balanced", "acc-local-l3"}`

默认仍返回 `balanced`。只有显式设置 `acc-local-l3`，才会切到新 kernel/op。

### 7.2 `_custom_ops.py` 只做薄封装

`vllm/_custom_ops.py` 新增两组 wrapper：
- `cpu_attn_get_scheduler_metadata_acc_locality()`
- `cpu_attention_with_kv_cache_acc_locality()`

这里没有业务逻辑，作用只是把 Python 参数透传到 `torch.ops._C.*`。

### 7.3 benchmark 加了 mode 与 locality 摘要

`benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py` 当前已经支持：
- `--attn-locality-mode`
- `--attn-locality-group-span`

其中：
- `mode` 已经真正接到 backend/kernel 选择
- `group_span` 目前还是接口预留，`acc-local-l3` kernel 首版实际只支持 `span=1`

同一个脚本还会把每个 rank 的 `locality_groups` 落到结果 JSON 中，便于后续对照 `OMP` 绑核和 runtime 分组是否一致。

## 8. 可观测性与测试

为了避免“代码能跑，但看不出内部调度状态”，额外加了两个调试入口：

1. `torch.ops._C_utils.describe_cpu_locality_groups(cpu_ids)`
2. `torch.ops._C_utils.inspect_cpu_attn_acc_locality_metadata(scheduler_metadata)`

前者看“给定 CPU 列表会分出哪些 L3 groups”，后者看“当前 metadata 里最终写进去了哪些 subgroup 和 `kv_head` 映射”。

与之对应，已有测试覆盖了两类事情：
- Python/backend/benchmark 能否正确切换新旧路径
- 小规模 case 下，新 kernel 输出是否和 legacy 路径一致

通过记录见：
- `tests/v1/worker/test_cpu_binding.py`
- `tests/benchmarks/test_cpu_attn_mp.py`
- `tests/kernels/attention/test_cpu_attn.py`

## 9. 当前边界

当前实现不是完整终版，边界需要明确。

### 9.1 已经落地的部分

- 新 kernel/op 路径已可单独选择
- `L3/CCX` 分组已接入 runtime metadata
- `kv_head -> subgroup` 的限制已经生效
- 小规模正确性测试和 benchmark smoke 已通过

### 9.2 还没有落地的部分

- `group_span > 1` 还没有真正实现
- 没有做按 `q_len/kv_len` 自动启停的 heuristic
- `kv_head -> subgroup` 目前只是 round-robin，不是代价模型驱动
- 没有重写 legacy scheduler，本质仍是“局部复用 legacy thread slots”

所以现在更准确的描述是：
这是一版“基于现有 CPU attention kernel 的 locality-aware 调度外壳”，而不是一个完全独立的新 scheduler。

## 10. 核心代码阅读顺序

如果后面要继续扩展这条路径，建议按下面顺序读代码：

1. `vllm/v1/attention/backends/cpu_attn.py`
   看 Python 侧是怎么按 `locality_mode` 选 op 的。
2. `vllm/_custom_ops.py`
   看 Python wrapper 到 `torch.ops` 的映射。
3. `csrc/cpu/torch_bindings.cpp`
   看新 op 和调试接口是怎么注册出来的。
4. `csrc/cpu/utils.cpp` / `csrc/cpu/utils.hpp`
   看 `L3` 分组和 `ThreadLocalityManager` 的来源。
5. `csrc/cpu/cpu_attn_acc_locality_impl.hpp`
   先看 `AttentionMetadata`，再看 `build_scheduler_metadata()`，最后看 `AttentionMainLoop`。
6. `csrc/cpu/cpu_attn_acc_locality.cpp`
   看最终 runtime 入口、ISA/head_dim dispatch，以及 `AttentionMainLoop` 的实例化。

## 11. 一句话总结

当前 `acc-local-l3` 的实现逻辑，可以概括成一句话：

在不改 legacy CPU attention 数学实现的前提下，先把线程池按 `L3/CCX` 切成子池，再把每个 `kv_head` 绑定到某个子池，并让该子池只复用自己的 legacy thread slots 去执行对应任务。
