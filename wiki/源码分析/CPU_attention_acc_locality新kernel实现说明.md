# CPU attention acc-local-l3 新 kernel 实现说明

## 问题
- `csrc/cpu/cpu_attn_acc_locality.cpp` 这条新路径到底做了什么。
- 它和 legacy `csrc/cpu/cpu_attn.cpp` 相比，哪些部分复用，哪些部分真正改了。

## 先给结论
- `acc-local-l3` 不是重写了一套新的 attention 数学实现，而是在 legacy CPU attention 外面包了一层 locality-aware 的 scheduler metadata 和 runtime 枚举逻辑。
- 新路径复用的部分很多：
  - `dtype / head_dim / isa` 三层 dispatch 宏基本和 legacy 一样。
  - 真正的计算内核仍然来自 `AttentionImpl<ISA, scalar_t, head_dim>`，即 `VEC / VEC16 / AMX / NEON` 的那套实现。
  - scheduler 的基础 workitem / reduction item 仍先调用 legacy `get_scheduler_metadata(...)` 生成。
  - `AttentionMainLoop` 里大部分 tile、softmax、split-KV reduction 数值逻辑都直接复用 legacy 基类实现。
- 新路径真正新增的部分主要有三类：
  1. 新的 metadata 头：在 legacy metadata 前面再包一层 `cpu_attention_acc_locality::AttentionMetadata`。
  2. 线程局部性分组：把 OpenMP 线程按 `numa_node + socket_id + l3_cache_id` 分成 subgroup。
  3. 新的任务枚举方式：不再像 legacy 那样靠一个全局 atomic counter 抢所有任务，而是让每个线程只处理“本 subgroup 负责的 `kv_head` + 本 subgroup 可见的 legacy thread slot / reduction item”。

## 入口与调用链

### Python 侧怎么切到新路径
- `vllm/v1/attention/backends/cpu_attn.py` 里有 `locality_mode`。
- `_resolve_cpu_attn_locality_mode()` 只接受两种模式：
  - `balanced`
  - `acc-local-l3`
- `_get_cpu_attn_ops()` 会按 mode 选择两组 op：
  - `balanced`:
    - `ops.cpu_attn_get_scheduler_metadata`
    - `ops.cpu_attention_with_kv_cache`
  - `acc-local-l3`:
    - `ops.cpu_attn_get_scheduler_metadata_acc_locality`
    - `ops.cpu_attention_with_kv_cache_acc_locality`

所以新路径的切换点非常明确：不是在同一个 op 内加 if/else，而是 Python 侧直接换了一整组 scheduler op + attention op。

### C++ 侧有哪些入口
- `csrc/cpu/torch_bindings.cpp` 暴露了这几个新符号：
  - `get_scheduler_metadata_acc_locality(...)`
  - `cpu_attention_with_kv_cache_acc_locality(...)`
  - `inspect_cpu_attn_acc_locality_metadata(...)`
- 其中：
  - `get_scheduler_metadata_acc_locality(...)` 负责生成新 metadata
  - `cpu_attention_with_kv_cache_acc_locality(...)` 负责真正执行 attention
  - `inspect_cpu_attn_acc_locality_metadata(...)` 只是调试/观测辅助接口

### `cpu_attn_acc_locality.cpp` 这层本身做了什么
- 这层本质上还是“入口层 + dispatch 层”。
- 文件里的三组 dispatch 宏和 legacy 几乎一致：
  - `VLLM_DISPATCH_FLOATING_TYPES`
  - `CPU_ATTN_DISPATCH_CASE_HEADDIM`
  - `CPU_ATTN_DISPATCH_IMPL`
- 和 legacy 相比，这个文件真正不同的地方只有两处：
  1. scheduler metadata 不再直接调用 legacy 返回，而是调用 `cpu_attention_acc_locality::build_scheduler_metadata(...)`
  2. runtime 不再实例化 legacy `cpu_attention::AttentionMainLoop<attn_impl>`，而是实例化 `cpu_attention_acc_locality::AttentionMainLoop<attn_impl>`

## 文件分工

### `csrc/cpu/cpu_attn_acc_locality.cpp`
- 负责：
  - 解析 `isa_hint`
  - 根据 `dtype / head_dim / isa` 选出具体 `attn_impl`
  - 计算 `max_num_q_per_iter`
  - 组装 `AttentionInput`
  - 调用 `cpu_attention_acc_locality::build_scheduler_metadata(...)`
  - 调用 `cpu_attention_acc_locality::AttentionMainLoop<attn_impl>`

### `csrc/cpu/cpu_attn_acc_locality_impl.hpp`
- 负责：
  - 定义 acc-locality 自己的 metadata 头 `AttentionMetadata`
  - 生成 subgroup 映射
  - 把 legacy metadata 嵌到新 metadata 后面
  - 提供调试摘要 `build_runtime_summary(...)`
  - 定义新的 `AttentionMainLoop<attention_impl_t>`

所以可以把两者理解成：
- `.cpp`：入口和模板分发
- `.hpp`：真正的 acc-locality 数据结构和 runtime 实现

## metadata 怎么构造

### 1. 先和 legacy 一样决定 `attn_impl`
`get_scheduler_metadata_acc_locality(...)` 的流程是：
1. 把字符串 `isa_hint` 解析成 `cpu_attention::ISA`
2. 用 `dtype + head_dim + isa` 三层 dispatch 选出 `attn_impl`
3. 从 `attn_impl` 上读出 `MaxQHeadNumPerIteration`
4. 把这个值传给 `cpu_attention_acc_locality::build_scheduler_metadata(...)`

这一段和 legacy 的核心一致点是：
- 先根据运行时参数选出具体模板实例
- 再从模板实例上拿静态常量

### 2. 新 metadata 头长什么样
新路径的 metadata 头是 `cpu_attention_acc_locality::AttentionMetadata`，关键字段有：
- `magic` / `version`
  - 用来识别这是不是 acc-locality metadata
- `thread_num`
  - 当前 OpenMP 线程数
- `subgroup_num`
  - locality 分组数量
- `group_span`
  - 单个 `kv_head` 会覆盖多少个连续 subgroup
- `legacy_thread_num`
  - legacy metadata 里的总线程数
- `legacy_effective_thread_num`
  - legacy scheduler 实际分到了 workitem 的线程槽数量
- `actual_kv_head_num`
  - 当前 runtime 真正按多少个 kv-head 维度去调度
- `reduction_item_num`
  - 直接继承自 legacy metadata
- `attention_task_num`
  - acc-locality 路径下实际 attention 任务数
- `legacy_attention_task_num`
  - legacy 路径下对应的 attention 任务数
- `reduction_task_num`
  - acc-locality 路径下实际 reduction 任务数
- `subgroup_thread_num[]`
  - 每个 subgroup 里有多少线程
- `thread_to_group_id[]`
  - 每个 OpenMP thread 属于哪个 subgroup
- `thread_to_local_offset[]`
  - 线程在本 subgroup 内的局部编号
- `kv_head_to_subgroup[]`
  - 每个 `kv_head_idx` 的起始 subgroup
- `legacy_metadata_offset` / `legacy_metadata_size`
  - 指向后面嵌入的 legacy metadata blob

和 legacy 的关键区别是：
- legacy 的 `cpu_attention::AttentionMetadata` 只描述 workitem、reduction item、线程前缀和、scratchpad 大小和一个全局 atomic counter
- 新路径多了一层 “subgroup / kv_head 归属 / legacy metadata 嵌入位置” 的 header

### 3. `actual_kv_head_num` 的含义
新路径没有自己发明这个概念，而是沿用 legacy/GQA 口径：

```cpp
actual_kv_head_num = use_gqa ? num_heads_kv : num_heads_q
```

其中 `use_gqa` 的判定依赖：
- `q_heads_per_kv = num_heads_q / num_heads_kv`
- `max_num_q_per_iter % q_heads_per_kv == 0`

这和 legacy mainloop 里的逻辑一致。也就是说，acc-locality 并没有改变 “以 kv-head 还是 q-head 为外层调度维度” 这件事。

### 4. 先调用 legacy scheduler，再包一层
`build_scheduler_metadata(...)` 最关键的步骤是：
1. 先直接调用全局的 legacy `::get_scheduler_metadata(...)`
2. 得到一个完整的 legacy metadata tensor
3. 新申请一块更大的 `int8` tensor
4. 开头放 acc-locality header
5. 后面按 64B 对齐把 legacy metadata 整块 memcpy 进去

所以它不是“参考 legacy scheduler 逻辑后重新生成一份类似结构”，而是真的把 legacy metadata blob 整块嵌进来。

### 5. 为什么 memcpy 之后还要手动修正指针
这是这个实现里最容易漏看的点。

legacy `cpu_attention::AttentionMetadata` 内部包含两个指针：
- `workitem_groups_ptr`
- `reduction_items_ptr`

它们在 legacy metadata 构造时，是按“当前 `this` 地址后面紧跟数组”的方式计算出来的。现在 legacy metadata 被 memcpy 到了新 tensor 的另一段地址上，原来的两个指针值就不再可靠了。

所以 acc-locality 代码在 memcpy 之后立刻重新绑定：
- `legacy_metadata->workitem_groups_ptr = ...`
- `legacy_metadata->reduction_items_ptr = ...`

然后再做：
- `legacy_metadata->reset_counter()`

这里的含义很明确：
- 新路径复用了 legacy metadata 的内容
- 但因为换了物理地址，必须修正内部裸指针

### 6. subgroup 映射怎么生成
`init_thread_locality_mapping(...)` 会去读：
- `cpu_utils::ThreadLocalityManager::get_thread_locality_manager()->get_groups()`

每个 group 是 `ThreadLocalityGroupInfo`，里面有：
- `numa_node`
- `socket_id`
- `l3_cache_id`
- `thread_ids`
- `cpu_ids`

如果这组信息有效，代码会：
- 给每个 group 分一个 `subgroup_id`
- 记录每个 subgroup 的线程数
- 给每个线程写：
  - `thread_to_group_id[thread_id]`
  - `thread_to_local_offset[thread_id]`

如果 group 信息无效、为空，或者不能覆盖全部线程，就回退到默认模式：
- 只有 1 个 subgroup
- 所有线程都属于 subgroup 0
- `local_offset = thread_id`

所以这条路径不是“必须依赖 locality 信息才能运行”，而是：
- 有 locality 数据时按 L3/CCX 分组
- 没有时退回成退化版单组执行

### 7. `kv_head -> subgroup` 现在支持 `group_span`
当前实现仍然没有复杂 heuristic，但已经不再固定是 “1 个 `kv_head` 对应 1 个 subgroup”。

它现在做的是：

```cpp
start_subgroup = (kv_head_idx * group_span) % subgroup_num
```

然后让当前 `kv_head` 覆盖：
- `start_subgroup`
- `start_subgroup + 1`
- ...
- `start_subgroup + group_span - 1`

因此：
- `group_span=1` 时，语义和旧实现一致
- `group_span>1` 时，一个 `kv_head` 会覆盖多个连续 subgroup

这次实现带来的直接结果是：
- `local_kv_heads << subgroup_num` 时，`attention_task_num` 不会再被硬性压到“只有少数 subgroup 活跃”的水平
- 但当前 runtime 还没有把同一个 `kv_head` 的 legacy slot 在多个 subgroup 之间重新切分；每个被覆盖的 subgroup 仍会各自遍历一套从 `local_offset` 开始条带化出来的 legacy slot

所以 `group_span>1` 当前更接近：
- “让更多 subgroup 参与同一 `kv_head`”
- 而不是“把同一 `kv_head` 的 legacy 工作精确拆分给多个 subgroup”

### 8. `attention_task_num / reduction_task_num` 为什么变小
legacy 路径里：
- attention 任务总数是 `actual_kv_head_num * effective_thread_num`
- reduction 任务总数是 `actual_kv_head_num * reduction_item_num`

acc-locality 路径里，不是每个 kv head 都对所有 `effective_thread_num` 或所有 `reduction_item_num` 都开放，而是只对“本 subgroup 能看到的那部分 legacy slot / reduction item”开放。

因此它统计的是：
- 对每个 kv head，只加上 `min(subgroup_thread_num[subgroup_id], legacy_effective_thread_num)`
- reduction 也只加上 `min(subgroup_thread_num[subgroup_id], legacy_metadata->reduction_item_num)`

这正是新路径的核心思想：
- legacy 先给出“全局可执行槽位”
- acc-locality 再把这些槽位限制到 `kv_head` 当前覆盖的 subgroup 内

但 `2026-04-09` 的 `group_span=4` 复测也把一个新边界暴露得很清楚：
- `attention_task_num` 虽然已经从 `32` 恢复到接近 legacy 的 `127/128`
- 但这是通过“同一个 `kv_head` 在多个 subgroup 上各自复用一套 legacy slot”实现的
- runtime debug 里可以直接看到：同一 `kv_head` 覆盖的多个 subgroup 会打印出基本相同的 `legacy_slots=[...]`

因此当前 `group_span>1` 不能直接解释成“并行度恢复且不重复工作”，它实际会引入额外重复执行

## runtime 主循环怎么执行

## 1. 总体结构仍然继承 legacy mainloop
新路径的 `AttentionMainLoop<attention_impl_t>` 直接继承自：
- `cpu_attention::AttentionMainLoop<attention_impl_t>`

所以它直接复用了基类里的很多实现，包括：
- `final_output(...)`
- `partial_output(...)`
- `reduce_splits(...)`
- 底层 `Attention<tile_gemm_t>` 模板和 `execute_attention(...)`

这也是为什么可以说：
- 新 kernel 改的是调度和任务可见性
- 不是重写计算 kernel

### 2. operator() 开头先做的事情
`operator()(const AttentionInput* input)` 开头会：
1. 取 `omp_get_max_threads()`
2. 校验 `metadata.thread_num == thread_num`
3. 取出嵌入的 legacy metadata
4. 若环境变量 `VLLM_CPU_ATTN_ACC_LOCALITY_DEBUG` 非空且非 `0`，打印 runtime summary
5. 建一个本地 `guard_counter`

这里和 legacy 类似的一点是：
- 仍然按 OpenMP 并行区启动所有线程

### 3. OpenMP 并行区仍是 “每个 thread_id 跑一份主体”
外层并行仍然是：

```cpp
#pragma omp parallel for schedule(static, 1)
for (thread_id = 0; thread_id < thread_num; ++thread_id)
```

所以 acc-locality 没有改变“OpenMP 线程级并行”这一层。

### 4. scratchpad、tile 大小、GQA 判定都基本沿用 legacy
进入每个线程后，代码仍然会做和 legacy 基本相同的事情：
- 算 `q_heads_per_kv`
- 判定 `use_gqa`
- 算 `actual_kv_head_num`
- 算 `actual_q_heads_per_kv`
- 算 `max_q_token_num_per_iter`
- 初始化 `AttentionScratchPad`
- 在 split-KV 时清空 reduction flag
- 依据 `available_cache_size` 计算默认 tile 大小

这部分说明：
- acc-locality 没改 attention tile、buffer、softmax/reduction 的数值执行路径
- 它改的是“哪些线程去做哪些 kv head / legacy slot”

### 5. 先把当前线程映射到 subgroup
这是新路径和 legacy 开始分叉的地方。

每个线程会先读：
- `subgroup_id = metadata.thread_to_group_id[thread_id]`
- `local_offset = metadata.thread_to_local_offset[thread_id]`
- `subgroup_thread_num = metadata.subgroup_thread_num[subgroup_id]`

如果这些值非法或 subgroup 为空，线程直接跳过。

这一步的意义是：
- legacy 只知道 “你是全局 thread_id 几号”
- 新路径进一步知道 “你是哪个 L3 subgroup 的第几个线程”

### 6. attention 任务不再通过全局 atomic counter 分发
这是和 legacy 最大的 runtime 差异。

legacy 做法是：
- `metadata.acquire_counter()` 取一个全局 task_idx
- 把 `task_idx` 映射成：
  - `kv_head_idx`
  - `thread_offset`
  - 或 reduction 的 `item_offset`

也就是说，legacy 的任务分发是：
- 所有线程共享一个全局任务空间
- 谁先抢到 counter，谁就去执行对应任务

acc-locality 不这样做。

新路径是显式双层循环：

1. 遍历所有 `kv_head_idx`
2. 只处理 `kv_head_to_subgroup[kv_head_idx] == subgroup_id` 的 kv head
3. 对于每个 kv head：
   - attention 阶段：
     - `legacy_thread_offset = local_offset; legacy_thread_offset < effective_thread_num; legacy_thread_offset += subgroup_thread_num`
   - reduction 阶段：
     - `item_offset = local_offset; item_offset < reduction_item_num; item_offset += subgroup_thread_num`

这意味着：
- 线程不再抢全局任务
- 任务可见性先被 `subgroup_id` 限制
- subgroup 内部再按 `local_offset` 做条带式切分

这里还有一个非常重要、但容易在日志里看漏的语义：
- 只有“当前 subgroup 覆盖到了某个 `kv_head`”时，这个 subgroup 的线程才会真正进入该 `kv_head` 的 attention 主循环
- 如果当前 subgroup 没有覆盖任何 `kv_head`，那么这些线程虽然仍会进入 OpenMP 并行区、做公共初始化和同步，但在后面的 attention / reduction 双层枚举里都会直接跳过

所以在 `group_span=1` 且 `local_kv_heads << subgroup_num` 时，可以把 runtime 近似理解成：
- 只有少数分到 `kv_head` 的 subgroup 真正在做 attention 计算
- 这些活跃 subgroup 内部的线程也不是去“抢任务”，而是按自己固定的 `legacy_thread_offset = local_offset + n * subgroup_thread_num` 序列循环执行
- 其余 subgroup 打印出来的 `legacy_slots` 最多只能说明“如果它有 `kv_head` 会怎么切 slot”，不能说明它已经实际执行了 attention 主循环

### 7. “legacy slot 复用” 是新路径的关键
新路径没有重新生成一份 subgroup 专属 scheduler。

它做的是：
- 仍然复用 legacy scheduler 生成的 `workitem_groups_ptr`
- 仍然复用 `cu_workitem_num_per_thread`
- 但不再把所有 legacy thread slot 都暴露给所有线程

具体来说，假设：
- 某个 subgroup 里有 4 个线程
- `legacy_effective_thread_num = 10`
- 当前线程在 subgroup 内的 `local_offset = 1`

那这个线程看到的 legacy slot 序列就是：
- `1, 5, 9`

代码里就是：

```cpp
for (legacy_thread_offset = local_offset;
     legacy_thread_offset < effective_thread_num;
     legacy_thread_offset += subgroup_thread_num)
```

所以更准确地说：
- 新路径不是废弃 legacy slot
- 而是把 legacy slot 在 subgroup 内做条带式重映射

### 8. attention 计算主体本身几乎没改
`run_attention_for_legacy_thread(...)` 里面的主流程和 legacy 基本一致：
- 从 `curr_workitem_groups` 取当前 legacy slot 对应的一批 workitem
- 取 `req_id / kv_split_pos_start / kv_split_pos_end / q_token_id_start / q_token_num`
- 计算 tile 边界
- 搬运 Q tile 到 buffer
- 调 `attn_impl.template execute_attention<Attention>(...)`
- 做 `final_output(...)` 或 `partial_output(...)`

也就是说：
- 新路径没有动 `execute_attention(...)` 的计算内核
- 没有改 tile 内 softmax 和 partial output 的数学逻辑
- 没有改 split-KV 最终归并公式

### 9. reduction 也被限制在 subgroup 内
新路径对 reduction 的处理和 attention 一样，也是按 subgroup 过滤。

它会：
1. 先判断当前 `kv_head_idx` 是否归当前 subgroup
2. 再让 subgroup 内线程按 `local_offset` 条带式遍历 reduction item
3. 调基类的：
   - `reduce_splits(...)`
   - `final_output(...)`

因此新路径的 locality 约束不只在 attention workitem 上，也作用在 split-KV reduction 阶段。

## 和 legacy 的逐项对比

### 1. 调用入口
- legacy：
  - `get_scheduler_metadata`
  - `cpu_attention_with_kv_cache`
- acc-locality：
  - `get_scheduler_metadata_acc_locality`
  - `cpu_attention_with_kv_cache_acc_locality`

差异：
- 不是同一个 op 内部分支
- 是 Python 侧直接切换到另一对 op

### 2. metadata 结构
- legacy：
  - 单层 `cpu_attention::AttentionMetadata`
  - 核心是 workitem/reduction item + 全局 atomic counter
- acc-locality：
  - 外层 `cpu_attention_acc_locality::AttentionMetadata`
  - 内层嵌一个完整 legacy metadata blob
  - 额外保存 subgroup / thread / kv_head 映射

差异：
- 新路径保留 legacy metadata 作为 payload
- 没有完全抛弃 legacy scheduler 产物

### 3. 线程到任务的映射方式
- legacy：
  - 全局 atomic counter 抢任务
  - 任务空间是 `kv_head × effective_thread_num` 和 `kv_head × reduction_item_num`
- acc-locality：
  - 先按 `kv_head -> subgroup` 限制负责范围
  - subgroup 内再用 `local_offset` 条带式遍历 legacy thread slot / reduction item

差异：
- legacy 是全局抢占
- 新路径是分组后静态枚举

### 4. locality 语义
- legacy：
  - scheduler 只按工作量和 tile 切分，不看 L3/CCX subgroup
- acc-locality：
  - 先按 `ThreadLocalityManager` 的 group 建 subgroup
  - 当前粒度由 `numa_node + socket_id + l3_cache_id` 决定
  - 每个 kv head 只归一个 subgroup

差异：
- 新路径显式引入了 “同一组 K/V 尽量由同一 L3 子池消费” 这一约束

### 5. attention 数学内核
- legacy：
  - `AttentionImpl<ISA, scalar_t, head_dim>`
- acc-locality：
  - 还是 `AttentionImpl<ISA, scalar_t, head_dim>`

差异：
- 没有差异
- 两者底层数学 kernel 相同

### 6. scheduler 是否完全重写
- legacy：
  - 唯一 scheduler
- acc-locality：
  - 没有单独重写 workitem 生成算法
  - 先调用 legacy scheduler，后面只包 header 和限制可见性

差异：
- 新路径当前更像“legacy scheduler 的 locality-aware 外壳”
- 还不是完全独立的新 scheduler

## 为什么这条路径值得单独存在
- 如果直接在 legacy `cpu_attention_with_kv_cache` 里塞 locality 逻辑，会把：
  - 原有 balanced 行为
  - 新的 subgroup 行为
  - 调试切换逻辑
  混在一起。
- 现在这条路径的组织方式更清楚：
  - `balanced` 保持原实现
  - `acc-local-l3` 单独走新的 op 和 metadata
  - 两者底层 `AttentionImpl` 共享

这使得对比实验和调试都更直接：
- 相同 `AttentionImpl`
- 不同任务调度与 locality 约束

## 当前实现边界
- 已经实现的：
  - 单独的新 op 路径
  - subgroup metadata header
  - `ThreadLocalityManager` 分组接入
  - `kv_head -> subgroup` 映射
  - `group_span > 1`
  - attention 和 reduction 都按 subgroup 受限
  - 调试摘要输出
- 还没有实现的：
  - 更智能的 `kv_head -> subgroup` 映射策略
  - 把同一 `kv_head` 的 legacy slot 在多个 covered subgroup 之间真正重分片
  - 按 workload 自动开关 locality mode
  - 完全独立于 legacy scheduler 的 workitem 生成

## 调试时最值得看什么
- metadata 是否正确：
  - `metadata.subgroup_num`
  - `metadata.subgroup_thread_num[...]`
  - `metadata.thread_to_group_id[...]`
  - `metadata.thread_to_local_offset[...]`
  - `metadata.kv_head_to_subgroup[...]`
- legacy payload 是否仍然完整：
  - `metadata.legacy_effective_thread_num`
  - `metadata.reduction_item_num`
  - `metadata.legacy_metadata()->cu_workitem_num_per_thread[...]`
- 运行时当前线程拿到的身份：
  - `thread_id`
  - `subgroup_id`
  - `local_offset`
- 当前线程实际消费的 legacy slot：
  - `legacy_thread_offset = local_offset + n * subgroup_thread_num`

如果这些值都对，那么说明：
- 新路径的 locality 外壳在工作
- 后续性能差异才主要归因到“限制任务可见性”本身，而不是 metadata 构造错误

## 一句话概括
- legacy CPU attention 的核心思路是：先做全局 workload 切分，再让所有线程通过全局 counter 抢任务。
- `acc-local-l3` 新 kernel 的核心思路是：保留 legacy 的 workitem 和计算 kernel，但在外面加一层 `L3/CCX subgroup` 约束，让每个 `kv_head` 只在自己覆盖到的线程子池里执行，并让 subgroup 内线程条带式复用 legacy thread slot 与 reduction item。
