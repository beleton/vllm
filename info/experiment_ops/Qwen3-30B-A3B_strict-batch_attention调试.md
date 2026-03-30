# Qwen3-30B-A3B：`vllm bench strict-batch` 下的 attention 调试

## 1. 目的

- 在 VS Code 里从 `vllm bench strict-batch` 断到 CPU attention 的 C++ 实现。
- 重点入口：
  - Python 调度入口：`vllm/benchmarks/strict_batch.py`
  - Python attention 入口：`vllm/v1/attention/backends/cpu_attn.py:348-364`
  - C++ 入口：`csrc/cpu/cpu_attn.cpp::cpu_attention_with_kv_cache`
  - OpenMP 主循环：`csrc/cpu/cpu_attn_impl.hpp:1339-1458`

## 2. 根因

- `事实`：`strict-batch` 要求 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；否则脚本会直接报错。证据：`vllm/benchmarks/strict_batch.py:230-245`
- `事实`：这只保证 `EngineCoreClient` 走同进程路径；真正的模型执行器仍由 `Executor.get_class(vllm_config)` 决定。证据：`vllm/v1/engine/llm_engine.py:164-178`
- `事实`：`world_size > 1` 时，默认 `distributed_executor_backend` 会变成 `mp`；`world_size == 1` 时才默认走 `uni`。证据：`vllm/config/parallel.py:602-647`
- `事实`：CPU 平台上 `world_size > 1` 时还会强制把 backend 收敛成 `mp`。证据：`vllm/platforms/cpu.py:216-229`
- `事实`：`mp` 会映射到 `MultiprocExecutor`。证据：`vllm/v1/executor/abstract.py:46-69`

所以：

- `TP=2` 时，attention 实际跑在 `Worker_TP0/Worker_TP1` 子进程里
- 你直接启动父进程上的 `cppdbg`，不会在主进程命中 `cpu_attention_with_kv_cache`

## 3. 两种调试方式

### 3.1 想直接断内核：用 `TP=1`

在 VS Code 里选择：

- `C++: gdb vllm bench strict-batch Qwen3-30B-A3B (attention, TP=1)`

这条配置会直接启动：

```bash
python -m vllm.entrypoints.cli.main bench strict-batch ...
```

`事实`：这里用 `TP=1`，执行器会走 `uni`，最容易直接断进 `cpu_attention_with_kv_cache`。

### 3.2 必须保留 `TP=2`：先跑 Python，再 attach worker

1. 先启动：
   - `Python: vllm bench strict-batch Qwen3-30B-A3B (strict batch prefill)`
2. 等日志里出现：
   - `Worker_TP0 pid=...`
   - `Worker_TP1 pid=...`
3. 再启动：
   - `C++: attach gdb to vllm CPU worker (TP child process)`
4. 在进程列表里选对应的 worker Python 进程

`事实`：这时 attach 的是实际执行 attention 的子进程，不是调度它们的父进程。

## 4. 推荐断点

先直接用函数断点：

- `cpu_attention_with_kv_cache`

命中后再按下面顺序看：

1. `csrc/cpu/cpu_attn.cpp:224`
   - 看 `AttentionInput` 是否已经填好
2. `csrc/cpu/cpu_attn.cpp:263`
   - 看是否真正进入 `AttentionMainLoop`
3. `csrc/cpu/cpu_attn_impl.hpp:1339`
   - 进入 OpenMP 主循环
4. `csrc/cpu/cpu_attn_impl.hpp:1444`
   - 看 `metadata.acquire_counter()` 如何给线程分配 task
5. `csrc/cpu/cpu_attn_impl.hpp:1454-1457`
   - 看当前 task 映射到哪个 `kv_head_idx / thread_offset`

注意：

- `csrc/cpu/cpu_attn.cpp:203` 是函数签名行，不是稳定的源码断点位置；优先用函数名断点

## 5. 调试链路

当前 `strict-batch` 的关键链路是：

```text
vllm/benchmarks/strict_batch.py:157-197
  -> engine.step()
  -> vllm/v1/attention/backends/cpu_attn.py:348-364
  -> vllm/_custom_ops.py:3006-3038
  -> csrc/cpu/cpu_attn.cpp:203-268
  -> csrc/cpu/cpu_attn_impl.hpp:1339-1458
```

如果你要先确认“这一轮请求是否先完整入队再第一次 step”，优先看：

- `vllm/benchmarks/strict_batch.py:165-173`
- `vllm/benchmarks/strict_batch.py:189-197`

## 6. 线程数建议

- `OMP_NUM_THREADS=1`
  - 适合先保证断点稳定命中
- `OMP_NUM_THREADS=4`
  - 适合再看 OpenMP 抢 task

`事实`：attention 内核实际使用的线程数来自 `omp_get_max_threads()`。证据：`csrc/cpu/cpu_attn_impl.hpp:1340`

## 7. 两个 gdb 设置

- `set breakpoint pending on`
  - 允许在 `vllm._C` 还没被 Python import 前，先挂上 `cpu_attention_with_kv_cache` 断点
- `set print thread-events off`
  - 关闭 OpenMP 线程创建/退出提示，减少调试输出噪音
