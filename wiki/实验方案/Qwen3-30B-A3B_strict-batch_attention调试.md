# Qwen3-30B-A3B strict-batch attention 调试

## 目的
- 在 VS Code 里从 `vllm bench strict-batch` 断到 CPU attention 的 Python 与 C++ 实现。
- 重点入口：
  - Python 调度入口：`vllm/benchmarks/strict_batch.py`
  - Python attention 入口：`vllm/v1/attention/backends/cpu_attn.py`
  - C++ 入口：`csrc/cpu/cpu_attn.cpp::cpu_attention_with_kv_cache`
  - OpenMP 主循环：`csrc/cpu/cpu_attn_impl.hpp`

## 核心结论
- `strict-batch` 要求 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，但这不等于模型执行一定在主进程。
- 对 CPU 平台，`TP=2` 时 attention 仍实际跑在 `Worker_TP0 / Worker_TP1` 子进程里。
- 所以：
  - 想直接断内核，最稳的是 `TP=1`
  - 必须保留 `TP=2` 时，应先启动 Python workload，再 attach worker 子进程

## 根因
- `strict-batch` 要求 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；否则脚本会直接报错。
- 这只保证 `EngineCoreClient` 走同进程路径；真正的模型执行器仍由 `Executor.get_class(vllm_config)` 决定。
- `world_size > 1` 时，默认 `distributed_executor_backend` 会变成 `mp`；`world_size == 1` 时才默认走 `uni`。
- CPU 平台上 `world_size > 1` 时还会强制把 backend 收敛成 `mp`。

所以：
- `TP=2` 时，attention 实际跑在 `Worker_TP0 / Worker_TP1` 子进程里
- 直接启动父进程上的 `cppdbg`，不会在主进程命中 `cpu_attention_with_kv_cache`

## 两种调试方式

### 1. 直接断内核：`TP=1`
- 适合先确认能稳定命中：
  - `cpu_attention_with_kv_cache`
- VS Code 里直接选择：
  - `C++: gdb vllm bench strict-batch Qwen3-30B-A3B (attention, TP=1)`

### 2. 保留 `TP=2`
1. 先启动：
   - `Python: vllm bench strict-batch Qwen3-30B-A3B (strict batch prefill)`
2. 等日志出现：
   - `Worker_TP0 pid=...`
   - `Worker_TP1 pid=...`
3. 再启动：
   - `C++: attach gdb to vllm CPU worker (TP child process)`
4. 在进程列表里选对应的 worker Python 进程

这里 attach 的是实际执行 attention 的子进程，不是调度它们的父进程。

## 推荐断点
- 函数断点：
  - `cpu_attention_with_kv_cache`
- 命中后按下面顺序看：
  - `csrc/cpu/cpu_attn.cpp:224`
  - `csrc/cpu/cpu_attn.cpp:263`
  - `csrc/cpu/cpu_attn_impl.hpp:1339`
  - `csrc/cpu/cpu_attn_impl.hpp:1444`
  - `csrc/cpu/cpu_attn_impl.hpp:1454-1457`
- 说明：
  - `csrc/cpu/cpu_attn.cpp:203` 是函数签名行，不是稳定的源码断点位置；优先用函数名断点

## 调试链路

```text
vllm/benchmarks/strict_batch.py
  -> engine.step()
  -> vllm/v1/attention/backends/cpu_attn.py
  -> vllm/_custom_ops.py
  -> csrc/cpu/cpu_attn.cpp
  -> csrc/cpu/cpu_attn_impl.hpp
```

如果要先确认“这一轮请求是否先完整入队再第一次 step”，优先看：
- `vllm/benchmarks/strict_batch.py` 里 request 入队和 `engine.step()` 相邻的那段逻辑
- `vllm/v1/attention/backends/cpu_attn.py` 调到 `_custom_ops` 的那段路径

## 线程建议
- `OMP_NUM_THREADS=1`
  - 适合先保证断点稳定命中
- `OMP_NUM_THREADS=4`
  - 适合再看 OpenMP 抢 task

## gdb 设置
- `set breakpoint pending on`
- `set print thread-events off`
