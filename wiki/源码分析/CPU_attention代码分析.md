# CPU attention 代码分析

## 说明
- 这页作为 CPU attention 源码分析的持续入口。
- 当前记录 dispatch 宏、模板实例化、调试切入点，以及 `cpu_attn_impl.hpp` 中基于 L2 cache 的 tile 与 scratchpad 规划。

## 问题
- `VLLM_DISPATCH_FLOATING_TYPES`、`CPU_ATTN_DISPATCH_CASE_HEADDIM`、`CPU_ATTN_DISPATCH_IMPL` 在 CPU attention 路径里分别做什么。
- 调试 `cpu_attention_with_kv_cache_acc_locality` 时，为什么会看到多层 lambda / 宏分发，而不是直接进入某个固定实现。

## 入口
- Python 侧 `vllm/v1/attention/backends/cpu_attn.py::_get_cpu_attn_ops()` 会按 `locality_mode` 选择：
  - `balanced -> ops.cpu_attn_get_scheduler_metadata / ops.cpu_attention_with_kv_cache`
  - `acc-local-l3 -> ops.cpu_attn_get_scheduler_metadata_acc_locality / ops.cpu_attention_with_kv_cache_acc_locality`
- Python 侧 `_get_attn_isa(dtype, block_size, head_size)` 会先给出字符串形式的 `isa`。
- C++ 侧 `csrc/cpu/torch_bindings.cpp` 通过 `TORCH_LIBRARY_EXPAND` 把这些 op 暴露给 Python。
- 真正的 dispatch 落在：
  - `csrc/cpu/cpu_attn.cpp`
  - `csrc/cpu/cpu_attn_acc_locality.cpp`

## 结论
- 这几组宏都不是 attention 计算本身，它们只负责把运行时参数转成一个具体的 C++ 模板实例。
- 在 `cpu_attention_with_kv_cache_acc_locality()` 里，最终会把三类运行时信息依次固化：
  - `dtype -> scalar_t`
  - `head_dim -> constexpr size_t head_dim`
  - `isa -> using attn_impl = AttentionImpl<ISA, scalar_t, head_dim>`
- 之后才会实例化：
  - `cpu_attention_acc_locality::AttentionMainLoop<attn_impl> mainloop;`
  - `mainloop(&input);`
- 所以调试时真正值得确认的不是“宏怎么展开成文本”，而是这 3 个运行时值最后各自选中了哪一个分支。

## 三层 dispatch 的真实作用

### 1. `VLLM_DISPATCH_FLOATING_TYPES`
- 定义位置：
  - `csrc/cpu/cpu_types_x86.hpp`
  - 其他平台有对应版本
- 它最终调用的是 `AT_DISPATCH_SWITCH(...)`。
- 在 CPU x86 路径下，它只允许三种浮点类型：
  - `at::ScalarType::Float`
  - `at::ScalarType::BFloat16`
  - `at::ScalarType::Half`
- 进入某个分支后，会定义出当前分支的 `scalar_t`。

在 `cpu_attn_acc_locality.cpp` 里的效果等价于：

```cpp
switch (query.scalar_type()) {
  case Float: {
    using scalar_t = float;
    ...
  }
  case BFloat16: {
    using scalar_t = c10::BFloat16;
    ...
  }
  case Half: {
    using scalar_t = c10::Half;
    ...
  }
}
```

### 2. `CPU_ATTN_DISPATCH_CASE_HEADDIM`
- 定义位置：
  - `csrc/cpu/cpu_attn.cpp`
  - `csrc/cpu/cpu_attn_acc_locality.cpp`
- 它按 `head_dim` 做第二层 switch。
- 当前 CPU attention 只接受这些 head size：
  - `32`
  - `64`
  - `80`
  - `96`
  - `112`
  - `128`
  - `160`
  - `192`
  - `224`
  - `256`
- 进入某个分支后，会定义：

```cpp
constexpr size_t head_dim = HEAD_DIM;
```

- 这里的目的不是保存一个普通整数，而是把 `head_dim` 变成编译期常量，供模板参数使用。

### 3. `CPU_ATTN_DISPATCH_IMPL`
- 定义位置：
  - `csrc/cpu/cpu_attn.cpp`
  - `csrc/cpu/cpu_attn_acc_locality.cpp`
- 它按 `ISA` 做第三层分发。
- `ISA` 枚举定义在 `csrc/cpu/cpu_attn_impl.hpp`：
  - `AMX`
  - `VEC`
  - `VEC16`
  - `NEON`
- 每个分支里都会定义：

```cpp
using attn_impl = cpu_attention::AttentionImpl<ISA类型, scalar_t, head_dim>;
```

- 例如：
  - `AttentionImpl<ISA::VEC, c10::BFloat16, 128>`
  - `AttentionImpl<ISA::VEC16, c10::Half, 80>`
  - `AttentionImpl<ISA::AMX, c10::BFloat16, 128>`

## 三层连起来后实际发生了什么
- `cpu_attention_with_kv_cache_acc_locality()` 的核心代码是：

```cpp
VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), ..., [&]() {
  CPU_ATTN_DISPATCH_CASE_HEADDIM(query.size(2), [&] {
    CPU_ATTN_DISPATCH_IMPL(input.metadata->legacy_metadata()->isa, [&]() {
      cpu_attention_acc_locality::AttentionMainLoop<attn_impl> mainloop;
      mainloop(&input);
    });
  });
});
```

- 这段代码的实际含义是：
  1. 先看 `query.scalar_type()`，决定 `scalar_t`
  2. 再看 `query.size(2)`，决定编译期常量 `head_dim`
  3. 再看 `metadata->legacy_metadata()->isa`，决定具体 `AttentionImpl`
  4. 最后实例化对应的 `AttentionMainLoop<attn_impl>`

- 所以宏只是把：
  - `dtype`
  - `head_dim`
  - `isa`

  这三个运行时值，映射到一个具体模板实例。

## 两个常见调用点分别在做什么

### `get_scheduler_metadata_acc_locality()`
- 这里先做一轮同样的 dispatch，但目的不是跑 attention。
- 它只取：

```cpp
max_num_q_per_iter = attn_impl::MaxQHeadNumPerIteration;
```

- 然后把这个值传给 `build_scheduler_metadata(...)`。
- 也就是说，这里的 dispatch 只是为了知道“当前选中的 attention 实现一次最多处理多少个 q heads”，以便构造 scheduler metadata。
- 对当前机器，如果最终 dispatch 到 `AttentionImpl<ISA::VEC, scalar_t, head_dim>`，那么 `MaxQHeadNumPerIteration` 就是 `8`；这个常量定义在 `csrc/cpu/cpu_attn_vec.hpp`。

### `cpu_attention_with_kv_cache_acc_locality()`
- 这里的 dispatch 才会真正进入：
  - `AttentionMainLoop<attn_impl>::operator()`
- 后续 tile 大小、`BlockSizeAlignment`、`MaxQHeadNumPerIteration`、`HeadDim` 等静态常量，都会从当前选中的 `attn_impl` 上读取。

## `isa` 是怎么来的
- `acc-local-l3` 路径里的 `isa` 不是在 `cpu_attn_acc_locality.cpp` 里现算的。
- Python 侧 `_get_attn_isa(dtype, block_size, head_size)` 会先选出字符串：
  - 某些 `head_size % 32 != 0 && head_size % 16 == 0` 的情况直接选 `vec16`
  - 支持 AMX 且 `dtype == bfloat16` 且 `block_size % 32 == 0` 时可选 `amx`
  - 其他情况按平台和 `block_size` 选 `vec` / `neon` / `vec16`
- `get_scheduler_metadata_acc_locality()` 里会把字符串 `isa_hint` 解析成 `cpu_attention::ISA`。
- `cpu_attention_with_kv_cache_acc_locality()` 再从 `scheduler_metadata` 里的 legacy metadata 读出这个 `isa`，做最终分发。

## `cpu_attn_impl.hpp` 中的 L2 cache 优化

### 入口
- 这里描述的是 `csrc/cpu/cpu_attn_impl.hpp` 里的 balanced main loop。
- 该文件没有显式 `prefetch` 指令或 cache hint；L2 cache 的使用方式是用 L2 容量约束 `tile`（分块）大小和 `scratchpad`（临时缓冲区）布局。
- L2 可用容量来自 `cpu_utils::get_available_l2_size()`，该函数读取 `at::cpu::L2_cache_size()` 后右移一位，实际只使用 50% L2 容量作为预算。

### Scheduler 阶段
- `AttentionScheduler::schedule()` 会读取 L2 预算，并调用 `calcu_default_tile_size(...)` 计算默认 tile 大小。
- `calcu_default_tile_size(...)` 的 cache 模型包含：
  - `Q`: `q_tile_size * head_dim * q_buffer_elem_size`
  - `K/V`: `2 * k_tile_size * head_dim * elem_size`
  - `Q@K^T logits`: `max_num_q_per_iter * k_tile_size * logits_buffer_elem_size`
  - 中间输出: `q_tile_size * head_dim * output_buffer_elem_size`
- 默认情况下代码令 `q_tile_size == k_tile_size == tile_size`，并把结果按 `round_size` 向下取整，同时限制 `tile_size <= 128 * max_num_q_per_iter`。
- `default_tile_token_num = default_tile_size / q_head_per_kv` 后，scheduler 用它控制每个 workitem 接收的 Q token 数量；当需要开启新的 Q tile 迭代、当前 workitem 的 `q_token_num` 已达到该粒度且当前线程已有工作时，后续 workitem 会切到下一个线程。
- `kv_len_per_thread` 先按所有请求的对齐后 KV 长度除以线程数，再乘以 `(use_gqa ? input.num_heads_kv : input.num_heads_q)` 得到；scheduler 按该值把 workitem 分给线程。开启 KV split 时，只对尾部且 `q_tile_token_num <= split_kv_q_token_num_threshold` 的小 Q tile 切 KV，并生成 reduction item。

### Main Loop 阶段
- `AttentionMainLoop::operator()` 在每个 OpenMP 线程内重新读取 L2 预算，并用同一套 `calcu_default_tile_size(...)` 计算 `default_q_tile_token_num`。
- 每个 Q tile 开始时，代码计算：
  - `actual_q_token_num`
  - `q_head_tile_size = actual_q_token_num * actual_q_heads_per_kv`
  - `rounded_q_head_tile_size`，按 `max_q_head_num_per_iter` 向上取整
- 随后调用 `calcu_tile_size_with_constant_q(...)`，在当前 Q tile 大小固定的前提下回算 `kv_tile_size`。
- 当 `rounded_q_head_tile_size <= max_q_head_num_per_iter` 时，`one_round=true`，公式不把 K/V 计入 cache 分母；否则公式把 K/V 与 logits 一起计入 cache 分母。
- `buffer_manager.update(...)` 使用当前 `q_head_tile_size` 和 `kv_tile_size` 布局本线程 scratchpad，缓冲区包括 Q、logits、partial output、max、sum。
- Q tile 先通过 `copy_q_heads_tile(...)` 拷到 `q_buffer`。后续计算循环按 `kv_tile_size` 遍历 KV 区间，再按 `max_q_token_num_per_iter` 遍历 Q 子块，调用 `execute_attention(...)` 复用同一组 Q、logits、partial output、max、sum 缓冲区。
- 计算结束后，如果当前 workitem 未做 KV split，直接 `final_output(...)` 写回输出；如果做了 KV split，则先写入 reduction scratchpad，再由 reduction task 合并 split 结果。

### Scratchpad 与 cache line 处理
- `AttentionScratchPad` 按 `thread_id * attention_scratchpad_size_per_thread` 为每个 OpenMP 线程分配独立 attention scratchpad 区间。
- Q、logits、partial output、max、sum，以及 reduction 的 flag、output、max、sum 缓冲区大小都通过 `round_to_64(...)` 对齐到 64 字节。
- scheduler 会根据 L2 预算预先计算 `attention_scratchpad_size_per_thread` 和 `reduction_scratchpad_size_per_kv_head`，再通过 `ScratchPadManager::realloc(...)` 一次性申请总 scratchpad。
- `reduce_splits(...)` 里 `local_max[16]` 和 `local_sum[16]` 使用 `alignas(64)`；源码注释说明 split 的 max/sum 元素没有 cache alignment，因此使用本地缓冲减少 false sharing（伪共享）。

## 相关源码
- `vllm/v1/attention/backends/cpu_attn.py`
- `csrc/cpu/torch_bindings.cpp`
- `csrc/cpu/cpu_attn_impl.hpp`
- `csrc/cpu/cpu_types_x86.hpp`
- `csrc/cpu/cpu_attn.cpp`
- `csrc/cpu/cpu_attn_acc_locality.cpp`
- `csrc/cpu/cpu_attn_vec.hpp`
- `csrc/cpu/cpu_attn_vec16.hpp`
- `csrc/cpu/cpu_attn_amx.hpp`
- `csrc/cpu/cpu_attn_neon.hpp`
