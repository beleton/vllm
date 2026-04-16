# Qwen3-30B-A3B `strict-batch` `gdb` 调试附录

## 适用范围
- 目标：在 `strict-batch` 运行时 attach 到 `Worker_TP`，断到 CPU attention 相关符号
- 原始来源：`info/experiment_ops/gdb_debug.md`

## 启动命令

```bash
export VLLM_CPU_KVCACHE_SPACE=32
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ENGINE_READY_TIMEOUT_S=86400
export VLLM_ENGINE_ITERATION_TIMEOUT_S=86400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400
export OMP_NUM_THREADS=4

/home/zjj/.conda/envs/vllm-cpu/bin/python \
    -m vllm.entrypoints.cli.main \
    bench strict-batch \
    --model /models/Qwen3-30B-A3B \
    --dtype bfloat16 \
    --block-size 128 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 20000 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --load-format dummy \
    --batch-size 16 \
    --input-len 512 \
    --output-len 1 \
    --prompt-seed 0 \
    --num-iters-warmup 0 \
    --num-rounds 4 \
    --disable-detokenize
```

## attach 步骤

```bash
ps -ef | grep Worker_TP
gdb -p <pid>
```

## 常用断点

```gdb
set print thread-events off
break cpu_attention_with_kv_cache
continue

break /home/zjj/vllm/csrc/cpu/cpu_attn.cpp:263
break /home/zjj/vllm/csrc/cpu/cpu_attn_impl.hpp:1339
continue
```
