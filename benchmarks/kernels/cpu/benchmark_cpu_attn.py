# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from vllm._custom_ops import (
    cpu_attn_reshape_and_cache,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import supports_amx_tiles
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    _get_attn_isa,
    _get_cpu_attn_ops,
)


def get_attn_isa(
    block_size: int | None = None,
    dtype: torch.dtype | None = None,
):
    if block_size and dtype:
        return _get_attn_isa(dtype, block_size)
    else:
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            return "neon"
        elif supports_amx_tiles():
            return "amx"
        else:
            return "vec"


# rand number generation takes too much time, cache rand tensors
@functools.lru_cache(maxsize=128, typed=False)
def tensor_cache(
    elem_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.randn(elem_num, dtype=dtype)
    return tensor


@dataclass
class AttentionBenchmarkResult:
    times_ms: list[float]
    time_min_ms: float
    time_max_ms: float
    time_mean_ms: float
    time_std_ms: float
    time_median_ms: float


@dataclass
class PreparedAttentionRun:
    attention_op: Callable[..., Any]
    query: torch.Tensor
    packed_key_cache: torch.Tensor
    packed_value_cache: torch.Tensor
    output: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    scale: float
    window_size: tuple[int, int]
    block_table: torch.Tensor
    scheduler_metadata: torch.Tensor
    s_aux: torch.Tensor | None


def prepare_attention_run(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int = None,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
    num_blocks: int = 4096,
    use_sink: bool = False,
    enable_kv_split: bool = False,
    isa: str | None = None,
    seed: int = 0,
    locality_mode: str = "balanced",
    locality_group_span: int = 1,
) -> PreparedAttentionRun:
    current_platform.seed_everything(seed)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5
    token_num = sum(query_lens)

    if isa is None:
        isa = get_attn_isa(block_size, dtype)

    s_aux = (
        15 * torch.rand((num_query_heads,), dtype=torch.bfloat16) if use_sink else None
    )

    query = tensor_cache(
        elem_num=token_num * num_query_heads * head_size,
        dtype=dtype,
    )
    query = query.view(
        token_num,
        num_query_heads,
        head_size,
    )

    key_value = tensor_cache(
        elem_num=2 * num_blocks * num_kv_heads * block_size * head_size,
        dtype=dtype,
    )
    key_value = key_value.view(
        2,
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
    )
    key_cache, value_cache = key_value.unbind(0)

    packed_key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
    )
    packed_value_cache = torch.empty_like(packed_key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
    )

    scheduler_op, attention_op = _get_cpu_attn_ops(locality_mode)
    scheduler_kwargs = dict(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_tensor,
        dtype=dtype,
        query_start_loc=cu_query_lens,
        causal=True,
        sliding_window_size=sliding_window if sliding_window is not None else -1,
        isa=isa,
        enable_kv_split=enable_kv_split,
    )
    if locality_mode == "acc-local-l3":
        scheduler_kwargs["group_span"] = locality_group_span
    metadata = scheduler_op(**scheduler_kwargs)

    return PreparedAttentionRun(
        attention_op=attention_op,
        query=query,
        packed_key_cache=packed_key_cache,
        packed_value_cache=packed_value_cache,
        output=torch.empty_like(query),
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        window_size=window_size,
        block_table=block_tables,
        scheduler_metadata=metadata,
        s_aux=s_aux,
    )


def run_attention_iters(
    prepared: PreparedAttentionRun,
    iters: int,
) -> list[float]:
    times = []
    for _ in range(iters):
        start_time = time.perf_counter_ns()
        prepared.attention_op(
            query=prepared.query,
            key_cache=prepared.packed_key_cache,
            value_cache=prepared.packed_value_cache,
            output=prepared.output,
            query_start_loc=prepared.query_start_loc,
            seq_lens=prepared.seq_lens,
            scale=prepared.scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=prepared.window_size,
            block_table=prepared.block_table,
            softcap=0,
            scheduler_metadata=prepared.scheduler_metadata,
            s_aux=prepared.s_aux,
        )
        end_time = time.perf_counter_ns()
        times.append((end_time - start_time) / 1e6)
    return times


def benchmark_attention(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int = None,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
    num_blocks: int = 4096,
    use_sink: bool = False,
    enable_kv_split: bool = False,
    isa: str | None = None,
    seed: int = 0,
    iters: int = 20,
    warmup_iters: int = 5,
    locality_mode: str = "balanced",
    locality_group_span: int = 1,
) -> AttentionBenchmarkResult:
    prepared = prepare_attention_run(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        num_blocks=num_blocks,
        use_sink=use_sink,
        enable_kv_split=enable_kv_split,
        isa=isa,
        seed=seed,
        locality_mode=locality_mode,
        locality_group_span=locality_group_span,
    )

    run_attention_iters(prepared, warmup_iters)
    times = run_attention_iters(prepared, iters)

    return AttentionBenchmarkResult(
        times_ms=times,
        time_min_ms=min(times),
        time_max_ms=max(times),
        time_mean_ms=float(np.mean(times)),
        time_std_ms=float(np.std(times)),
        time_median_ms=float(np.median(times)),
    )


@torch.inference_mode()
def main(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int = None,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
    num_blocks: int = 4096,
    use_sink: bool = False,
    enable_kv_split: bool = False,
    isa: str | None = None,
    seed: int = 0,
    iters: int = 20,
    locality_mode: str = "balanced",
) -> None:
    result = benchmark_attention(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        num_blocks=num_blocks,
        use_sink=use_sink,
        enable_kv_split=enable_kv_split,
        isa=isa,
        seed=seed,
        iters=iters,
        locality_mode=locality_mode,
    )

    print("\tmin (ms) = ", result.time_min_ms)
    print("\tmax (ms) = ", result.time_max_ms)
    print("\tmean (ms) = ", result.time_mean_ms)
    print("\tstd = ", result.time_std_ms)
    print("\tmedian (ms) = ", result.time_median_ms)


def generate_seq_lens(
    batch_size: int,
    q_len_min: int,
    q_len_max: int,
    kv_len_min: int,
    kv_len_max: int,
    seed: int = 0,
) -> list[tuple[int, int]]:
    assert 1 <= q_len_min <= q_len_max
    assert 1 <= kv_len_min <= kv_len_max
    assert kv_len_max >= q_len_min

    g = torch.Generator(device="cpu").manual_seed(seed)

    def rint(lo: int, hi: int) -> int:
        return torch.randint(lo, hi + 1, (1,), generator=g).item()

    seq_lens: list[tuple[int, int]] = []
    for _ in range(batch_size):
        # ensure q <= kv
        kv = rint(max(kv_len_min, q_len_min), kv_len_max)
        q = rint(q_len_min, min(q_len_max, kv))
        seq_lens.append((q, kv))

    return seq_lens


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the paged attention kernel.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--q-len-min", type=int, default=512)
    parser.add_argument("--q-len-max", type=int, default=512)
    parser.add_argument("--kv-len-min", type=int, default=512)
    parser.add_argument("--kv-len-max", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4096)

    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=CPUAttentionBackend.get_supported_head_sizes(),
        default=128,
    )
    parser.add_argument("--enable-kv-split", action="store_true")
    parser.add_argument("--block-size", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--use-sink", action="store_true")
    parser.add_argument(
        "--isa", type=str, choices=["vec", "neon", "amx", "vec16"], default=None
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--attn-locality-mode",
        choices=["balanced", "acc-local-l3"],
        default="balanced",
    )
    parser.add_argument("--attn-locality-group-span", type=int, default=1)

    args = parser.parse_args()
    print(args)

    seq_lens = generate_seq_lens(
        args.batch_size,
        args.q_len_min,
        args.q_len_max,
        args.kv_len_min,
        args.kv_len_max,
        args.seed,
    )

    print("batch (query len, kv len) = ", seq_lens)

    main(
        seq_lens=seq_lens,
        num_heads=(args.num_query_heads, args.num_kv_heads),
        head_size=args.head_size,
        sliding_window=args.sliding_window,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        use_sink=args.use_sink,
        enable_kv_split=args.enable_kv_split,
        isa=args.isa
        if args.isa is not None
        else get_attn_isa(args.block_size, STR_DTYPE_TO_TORCH_DTYPE[args.dtype]),
        seed=args.seed,
        iters=args.iters,
        locality_mode=args.attn_locality_mode,
    )
