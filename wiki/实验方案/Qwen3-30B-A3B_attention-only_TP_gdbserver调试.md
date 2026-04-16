# Qwen3-30B-A3B attention-only TP `gdbserver` 调试

## 适用范围
- 目标：调试 `csrc/cpu/cpu_attn_acc_locality_impl.hpp` 里的 `AttentionMainLoop`
- 当前示例 case：`dry_run_attn_only NPS1_TP2 2 prefill-like global-fixed 16 512 512 32 4 acc-local-l3`

## 关键证据
- `benchmark_cpu_attn_mp.py` 内部用 `mp.get_context("spawn")` 起 `tp_size` 个子进程。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py`
- `acc-local-l3` 会分发到 `ops.cpu_attention_with_kv_cache_acc_locality`。证据：`vllm/v1/attention/backends/cpu_attn.py`
- `cpu_attention_with_kv_cache_acc_locality()` 里会调用 `cpu_attention_acc_locality::AttentionMainLoop<attn_impl> mainloop; mainloop(&input);`。证据：`csrc/cpu/cpu_attn_acc_locality.cpp`

## 推荐方案：`gdbserver --multi`

这个方案适合处理 `spawn` 出来的两个 TP rank。

### 终端 1：启动 `gdbserver`

```bash
cd /home/zjj/vllm
conda activate vllm-cpu

mkdir -p /home/zjj/vllm/test_results/debug

gdbserver --multi :1234
```

说明：
- 建议先执行 `script -f ...`，再启动 `gdbserver`，这样可以完整保存这个终端里的输入输出
- 这个 `script` 只记录“终端 1”的内容，也就是 `gdbserver` 的输出；不会记录“终端 2”里手工输入的 `bt`、`info inferiors` 等 `gdb` 命令
- 退出时输入 `exit` 结束 `script`

### 终端 2：用 `gdb` 连接并直接启动 benchmark

```bash
cd /home/zjj/vllm
conda activate vllm-cpu

script -f /home/zjj/vllm/test_results/debug/gdb_terminal2_2026-04-08.log

gdb -q /home/zjj/.conda/envs/vllm-cpu/bin/python \
  -ex 'set breakpoint pending on' \
  -ex 'set pagination off' \
  -ex 'set print thread-events off' \
  -ex 'set follow-fork-mode child' \
  -ex 'set detach-on-fork off' \
  -ex 'set follow-exec-mode same' \
  -ex 'target localhost :1234' \
  -ex 'set remote exec-file /home/zjj/.conda/envs/vllm-cpu/bin/python' \
  -ex 'set environment VLLM_CPU_ATTN_ACC_LOCALITY_DEBUG 1' \
  -ex 'set environment LD_LIBRARY_PATH /home/zjj/.conda/envs/vllm-cpu/lib' \
  -ex 'set logging file /home/zjj/vllm/test_results/debug/gdb_attn_tp2.log' \
  -ex 'set logging overwrite on' \
  -ex 'set logging redirect off' \
  -ex 'set logging enabled on' \
  -ex 'catch fork' \
  -ex 'break /home/zjj/vllm/csrc/cpu/cpu_attn_acc_locality.cpp:178' \
  -ex 'break /home/zjj/vllm/csrc/cpu/cpu_attn_acc_locality_impl.hpp:387' \
  -ex 'set args /home/zjj/vllm/benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py --tp-size 2 --partition-mode global-fixed --workload prefill-like --batch-size 16 --q-len 512 --kv-len 512 --num-query-heads 32 --num-kv-heads 4 --head-size 128 --dtype bfloat16 --block-size 128 --enable-kv-split --omp-threads-bind auto --warmup-iters 0 --iters 10 --attn-locality-mode acc-local-l3 --output-json /home/zjj/vllm/test_results/debug/attn_tp2_prefill_q512_acc_locality_gdb.json' \
  -ex run
```

说明：
- 如果源码断点在共享库加载前设置，`set breakpoint pending on` 会避免交互式询问
- `csrc/cpu/cpu_attn_acc_locality.cpp:178` 是当前源码里真正执行 `mainloop(&input);` 的调用点
- `AttentionMainLoop::operator()` 入口断点是 `csrc/cpu/cpu_attn_acc_locality_impl.hpp:387`
- 并行区入口在 `csrc/cpu/cpu_attn_acc_locality_impl.hpp:400`
- 如果只想尽快停到 main loop，只保留 `csrc/cpu/cpu_attn_acc_locality_impl.hpp:387` 这一处断点也可以
- 本地已验证：这里通过 `-ex 'set logging ...'` 开启的 logging 能记录启动阶段输出，但不能保证后续手工输入的 `bt`、`info inferiors` 等交互输出继续写入同一日志文件
- 若你需要完整保存 `gdb` 交互命令和输出，应在“终端 2”里先执行 `script -f ...`，再启动 `gdb`

## 常用 `gdb` 命令

连上后常用：

```gdb
info inferiors
info threads
bt
frame 0
list
continue
next
step
finish
print thread_id
print subgroup_id
print local_offset
```

如果停在 `catch fork`，先看当前 inferior：

```gdb
info inferiors
inferior <N>
continue
```

通常需要切到新出现的子进程 inferior，再继续跑到 `AttentionMainLoop`。

## 如果先停在 `_GLOBAL__sub_I_*`

可能看到类似：

```gdb
Thread 1 "python" hit Breakpoint 2, _GLOBAL__sub_I_cpu_attn_acc_locality.cpp(void) ()
```

这表示当前还在共享库加载后的静态初始化/算子注册阶段，不表示已经进入 `AttentionMainLoop`。

处理顺序：

```gdb
bt
info breakpoints
continue
```

若随后停在 `catch fork`，先切到新出现的 TP 子进程 inferior，再继续：

```gdb
info inferiors
inferior <N>
continue
```

若你仍在 `csrc/cpu/cpu_attn_acc_locality.cpp` 这一处反复先命中初始化，可直接禁用该断点，只保留 `AttentionMainLoop::operator()` 入口断点：

```gdb
disable <breakpoint-number>
continue
```

原因：
- `TORCH_LIBRARY_EXPAND(...)` 注册算子时，编译器会为对应翻译单元生成 `_GLOBAL__sub_I_*` 初始化函数
- 如果断点在共享库装载阶段就已关联到该翻译单元，`gdb` 可能先在这个初始化函数附近停住

## 进入 main loop 前先看什么

比起先钻宏展开，更直接的是先确认 dispatch 用到的 3 个运行时值：

```gdb
print query.scalar_type()
print query.size(2)
print (int)input.metadata->legacy_metadata()->isa
```

如果已经在 `cpu_attn_acc_locality.cpp` 里的 `mainloop` 变量作用域内，也可以直接看模板实例：

```gdb
ptype mainloop
```

## 常用观察点

这些变量都在 `AttentionMainLoop::operator()` 附近：

```gdb
print thread_num
print input->num_heads
print input->num_kv_heads
print metadata.subgroup_num
print metadata.actual_kv_head_num
print metadata.thread_to_group_id[thread_id]
print metadata.thread_to_local_offset[thread_id]
print metadata.subgroup_thread_num[subgroup_id]
```

若已进入 lambda `run_attention_for_legacy_thread`，可继续看：

```gdb
print kv_head_idx
print legacy_thread_offset
print curr_workitem_groups_num
print q_head_start_idx
print current_group_idx
print kv_start_pos
print kv_end_pos
print q_token_num
```

## 输出保存

如果终端里不方便复制，优先用下面两种方法。

### 方法 1：GDB 自带 logging

若需要可靠保存后续手工交互输出，建议在程序停住后手动重新执行：

```gdb
set logging file /home/zjj/vllm/test_results/debug/gdb_manual_2026-04-08.log
set logging enabled off
set logging overwrite on
set logging redirect off
set logging enabled on
show logging
```

然后再执行：

```gdb
bt
info inferiors
info threads
```

结束时：

```gdb
set logging enabled off
```

说明：
- 已观察到：启动命令里的 `-ex 'set logging ...'` 能写入程序启动和首次停住信息
- 已验证可行：在停住后手动重新设置并开启 logging，后续 `gdb` 输出可以进入新日志文件

### 方法 2：保存整个终端会话

```bash
script -f /home/zjj/vllm/test_results/debug/gdb_full_2026-04-08.typescript
```

这个方法只会保存“当前终端”的输入输出。

- 若在“终端 1”执行，它保存的是 `gdbserver` 会话
- 若在“终端 2”执行，它保存的是 `gdb` 会话，能看到你手工输入的 `bt`、`info inferiors`、`continue` 等命令
- `script` 保存的是原始终端字节流；若会话里有彩色输出、光标移动或 `layout src` 这样的 TUI 重绘，文件里会出现 `\x1b[32m` 这类控制序列，看起来会比较乱
- 因此 `script` 更适合做“完整会话留档”，不适合当作干净文本日志；若需要干净的 `gdb` 文本输出，优先使用上面的 `set logging ...` 方法
