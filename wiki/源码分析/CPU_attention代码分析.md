# CPU attention 代码分析

## 说明
- 这页作为 CPU attention 源码分析的持续入口。
- 当前先记录 dispatch 宏、模板实例化和调试切入点，后续可继续补 attention 调度、tile、workitem、KV cache 访问等分析内容。

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
