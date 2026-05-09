# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from benchmarks.kernels.cpu.benchmark_cpu_attn import (
    generate_seq_lens,
    get_attn_isa,
    prepare_attention_run,
    run_attention_iters,
)
from vllm import envs
from vllm._custom_ops import (
    cpu_attn_get_timing_profile,
    cpu_attn_reset_runtime_timing_profile,
    cpu_attn_reset_timing_profile,
)
from vllm.platforms.cpu import CpuPlatform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend
from vllm.v1.worker.cpu_binding import (
    group_logical_cpus_by_l3,
    resolve_local_omp_cpuid,
)

DEFAULT_RUNTIME_CHUNK_ITERS = 1000


@dataclass(frozen=True)
class HeadShardPlan:
    partition_mode: str
    tp_size: int
    global_num_query_heads: int
    global_num_kv_heads: int
    local_num_query_heads: int
    local_num_kv_heads: int


def _is_truthy_env_value(value: str | None) -> bool:
    return value is not None and value != "" and value != "0"


def configure_attention_debug_env_for_rank(rank: int) -> None:
    if "VLLM_CPU_ATTN_DEBUG" not in os.environ:
        return

    os.environ["VLLM_CPU_ATTN_DEBUG_RANK"] = str(rank)
    if rank != 0 and not _is_truthy_env_value(
        os.environ.get("VLLM_CPU_ATTN_DEBUG_ALL_RANKS")
    ):
        os.environ["VLLM_CPU_ATTN_DEBUG"] = "0"


def configure_trace_env_for_rank(rank: int) -> None:
    configure_attention_debug_env_for_rank(rank)


def attention_profile_enabled() -> bool:
    return _is_truthy_env_value(os.environ.get("VLLM_CPU_ATTN_PROFILE"))


def configure_attention_profile_env() -> bool:
    if not attention_profile_enabled():
        return False
    os.environ.setdefault("VLLM_CPU_ATTN_PROFILE_SILENT", "1")
    return True


def reset_attention_profile() -> None:
    if attention_profile_enabled():
        cpu_attn_reset_timing_profile()


def reset_attention_runtime_profile() -> None:
    if attention_profile_enabled():
        cpu_attn_reset_runtime_timing_profile()


def _finalize_scheduler_profile(
    profile: dict[str, Any],
    locality_mode: str,
) -> dict[str, Any]:
    call_count = profile["call_count"]
    result = {
        "call_count": call_count,
        "scheduler_metadata_ns": profile["scheduler_metadata_ns"],
        "scheduler_metadata_avg_ns": (
            profile["scheduler_metadata_ns"] / call_count if call_count > 0 else None
        ),
    }
    if locality_mode == "acc-local-l3":
        result["legacy_scheduler_metadata_ns"] = profile[
            "legacy_scheduler_metadata_ns"
        ]
        result["locality_metadata_build_ns"] = profile[
            "locality_metadata_build_ns"
        ]
    return result


def _finalize_runtime_profile(
    profile: dict[str, Any],
    locality_mode: str,
) -> dict[str, Any]:
    attention_task_count = profile["attention_task_count"]
    reduction_task_count = profile["reduction_task_count"]
    execute_attention_count = profile["execute_attention_count"]
    result = {
        "call_count": profile["call_count"],
        "attention_task_body_ns": profile["attention_task_body_ns"],
        "attention_task_count": attention_task_count,
        "attention_task_body_avg_ns": (
            profile["attention_task_body_ns"] / attention_task_count
            if attention_task_count > 0
            else None
        ),
        "reduction_task_body_ns": profile["reduction_task_body_ns"],
        "reduction_task_count": reduction_task_count,
        "reduction_task_body_avg_ns": (
            profile["reduction_task_body_ns"] / reduction_task_count
            if reduction_task_count > 0
            else None
        ),
        "execute_attention_ns": profile["execute_attention_ns"],
        "execute_attention_count": execute_attention_count,
        "execute_attention_avg_ns": (
            profile["execute_attention_ns"] / execute_attention_count
            if execute_attention_count > 0
            else None
        ),
        "attention_task_non_execute_ns": max(
            profile["attention_task_body_ns"] - profile["execute_attention_ns"],
            0,
        ),
    }
    if locality_mode == "acc-local-l3":
        result["attention_counter_fetch_ns"] = profile[
            "attention_counter_fetch_ns"
        ]
        result["attention_counter_fetch_count"] = profile[
            "attention_counter_fetch_count"
        ]
        result["reduction_counter_fetch_ns"] = profile[
            "reduction_counter_fetch_ns"
        ]
        result["reduction_counter_fetch_count"] = profile[
            "reduction_counter_fetch_count"
        ]
    elif profile["counter_fetch_ns"] or profile["counter_fetch_count"]:
        result["counter_fetch_ns"] = profile["counter_fetch_ns"]
        result["counter_fetch_count"] = profile["counter_fetch_count"]
    return result


def collect_attention_profile(locality_mode: str) -> dict[str, Any] | None:
    if not attention_profile_enabled():
        return None

    profile_snapshot = json.loads(cpu_attn_get_timing_profile())
    profile_key = "balanced" if locality_mode == "balanced" else "acc_locality"
    selected_profile = profile_snapshot[profile_key]
    return {
        "mode": locality_mode,
        "scheduler": _finalize_scheduler_profile(
            selected_profile["scheduler"], locality_mode
        ),
        "runtime": _finalize_runtime_profile(
            selected_profile["runtime"], locality_mode
        ),
    }


def aggregate_attention_profiles(
    profiles: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    collected_profiles = [profile for profile in profiles if profile is not None]
    if not collected_profiles:
        return None

    mode = collected_profiles[0]["mode"]
    assert all(profile["mode"] == mode for profile in collected_profiles)

    scheduler_call_count = sum(
        profile["scheduler"]["call_count"] for profile in collected_profiles
    )
    scheduler_metadata_ns = sum(
        profile["scheduler"]["scheduler_metadata_ns"] for profile in collected_profiles
    )
    attention_task_body_ns = sum(
        profile["runtime"]["attention_task_body_ns"] for profile in collected_profiles
    )
    attention_task_count = sum(
        profile["runtime"]["attention_task_count"] for profile in collected_profiles
    )
    reduction_task_body_ns = sum(
        profile["runtime"]["reduction_task_body_ns"] for profile in collected_profiles
    )
    reduction_task_count = sum(
        profile["runtime"]["reduction_task_count"] for profile in collected_profiles
    )
    execute_attention_ns = sum(
        profile["runtime"]["execute_attention_ns"] for profile in collected_profiles
    )
    execute_attention_count = sum(
        profile["runtime"]["execute_attention_count"] for profile in collected_profiles
    )
    aggregate = {
        "mode": mode,
        "scheduler": {
            "call_count": scheduler_call_count,
            "scheduler_metadata_ns": scheduler_metadata_ns,
            "scheduler_metadata_avg_ns": (
                scheduler_metadata_ns / scheduler_call_count
                if scheduler_call_count > 0
                else None
            ),
        },
        "runtime": {
            "call_count": sum(
                profile["runtime"]["call_count"] for profile in collected_profiles
            ),
            "attention_task_body_ns": attention_task_body_ns,
            "attention_task_count": attention_task_count,
            "attention_task_body_avg_ns": (
                attention_task_body_ns / attention_task_count
                if attention_task_count > 0
                else None
            ),
            "reduction_task_body_ns": reduction_task_body_ns,
            "reduction_task_count": reduction_task_count,
            "reduction_task_body_avg_ns": (
                reduction_task_body_ns / reduction_task_count
                if reduction_task_count > 0
                else None
            ),
            "execute_attention_ns": execute_attention_ns,
            "execute_attention_count": execute_attention_count,
            "execute_attention_avg_ns": (
                execute_attention_ns / execute_attention_count
                if execute_attention_count > 0
                else None
            ),
            "attention_task_non_execute_ns": max(
                attention_task_body_ns - execute_attention_ns,
                0,
            ),
        },
    }
    if mode == "acc-local-l3":
        aggregate["scheduler"]["legacy_scheduler_metadata_ns"] = sum(
            profile["scheduler"]["legacy_scheduler_metadata_ns"]
            for profile in collected_profiles
        )
        aggregate["scheduler"]["locality_metadata_build_ns"] = sum(
            profile["scheduler"]["locality_metadata_build_ns"]
            for profile in collected_profiles
        )
        aggregate["runtime"]["attention_counter_fetch_ns"] = sum(
            profile["runtime"]["attention_counter_fetch_ns"]
            for profile in collected_profiles
        )
        aggregate["runtime"]["attention_counter_fetch_count"] = sum(
            profile["runtime"]["attention_counter_fetch_count"]
            for profile in collected_profiles
        )
        aggregate["runtime"]["reduction_counter_fetch_ns"] = sum(
            profile["runtime"]["reduction_counter_fetch_ns"]
            for profile in collected_profiles
        )
        aggregate["runtime"]["reduction_counter_fetch_count"] = sum(
            profile["runtime"]["reduction_counter_fetch_count"]
            for profile in collected_profiles
        )
    return aggregate


def resolve_head_shard_plan(
    num_query_heads: int,
    num_kv_heads: int,
    tp_size: int,
    partition_mode: str,
) -> HeadShardPlan:
    assert tp_size >= 1
    assert num_query_heads % num_kv_heads == 0, (
        "num_query_heads must be divisible by num_kv_heads."
    )

    if partition_mode == "global-fixed":
        assert num_query_heads % tp_size == 0, (
            "global-fixed requires num_query_heads divisible by tp_size."
        )
        assert num_kv_heads % tp_size == 0, (
            "global-fixed requires num_kv_heads divisible by tp_size."
        )
        local_num_query_heads = num_query_heads // tp_size
        local_num_kv_heads = num_kv_heads // tp_size
        global_num_query_heads = num_query_heads
        global_num_kv_heads = num_kv_heads
    elif partition_mode == "per-rank-fixed":
        local_num_query_heads = num_query_heads
        local_num_kv_heads = num_kv_heads
        global_num_query_heads = num_query_heads * tp_size
        global_num_kv_heads = num_kv_heads * tp_size
    else:
        raise ValueError(f"Unsupported partition_mode: {partition_mode}")

    assert local_num_query_heads % local_num_kv_heads == 0, (
        "Local query heads must be divisible by local KV heads."
    )

    return HeadShardPlan(
        partition_mode=partition_mode,
        tp_size=tp_size,
        global_num_query_heads=global_num_query_heads,
        global_num_kv_heads=global_num_kv_heads,
        local_num_query_heads=local_num_query_heads,
        local_num_kv_heads=local_num_kv_heads,
    )


def resolve_workload_lengths(
    workload: str,
    q_len: int | None,
    kv_len: int | None,
) -> tuple[int, int]:
    if workload == "decode-like":
        resolved_q_len, resolved_kv_len = q_len or 1, kv_len or 1024
    elif workload == "prefill-like":
        resolved_q_len = q_len or kv_len or 1024
        resolved_kv_len = kv_len or resolved_q_len
    else:
        assert q_len is not None and kv_len is not None, (
            "custom workload requires explicit q_len and kv_len."
        )
        resolved_q_len, resolved_kv_len = q_len, kv_len

    assert resolved_q_len > 0 and resolved_kv_len > 0, (
        "Resolved q_len and kv_len must be positive."
    )
    assert resolved_q_len <= resolved_kv_len, "q_len must be <= kv_len."
    return resolved_q_len, resolved_kv_len


def build_seq_lens(
    batch_size: int,
    workload: str,
    q_len: int | None,
    kv_len: int | None,
    seed: int,
) -> list[tuple[int, int]]:
    resolved_q_len, resolved_kv_len = resolve_workload_lengths(
        workload=workload,
        q_len=q_len,
        kv_len=kv_len,
    )
    return generate_seq_lens(
        batch_size=batch_size,
        q_len_min=resolved_q_len,
        q_len_max=resolved_q_len,
        kv_len_min=resolved_kv_len,
        kv_len_max=resolved_kv_len,
        seed=seed,
    )


def build_result_shape_dirname(
    batch_size: int,
    q_len: int | None,
    kv_len: int | None,
) -> str:
    assert q_len is not None and kv_len is not None, (
        "Result path generation requires resolved q_len and kv_len."
    )
    return f"batch_{batch_size}/q{q_len}_KV_{kv_len}"


def _expand_cpu_ids(omp_cpuids: str) -> list[int]:
    cpu_ids: list[int] = []
    for token in omp_cpuids.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid cpu range: {token}")
            cpu_ids.extend(range(start, end + 1))
        else:
            cpu_ids.append(int(token))
    return cpu_ids


def summarize_rank_locality_groups(
    omp_cpuids: str,
    logical_cpu_list,
    prefer_runtime_summary: bool = False,
) -> list[dict[str, Any]]:
    if omp_cpuids == "nobind":
        return []

    describe_groups = getattr(torch.ops._C_utils, "describe_cpu_locality_groups", None)
    if prefer_runtime_summary and callable(describe_groups):
        return json.loads(describe_groups(omp_cpuids))

    selected_cpu_ids = set(_expand_cpu_ids(omp_cpuids))
    selected_cpus = [cpu for cpu in logical_cpu_list if cpu.id in selected_cpu_ids]
    grouped_cpus = group_logical_cpus_by_l3(selected_cpus)

    return [
        {
            "numa_node": numa_node,
            "socket_id": socket_id,
            "l3_cache_id": l3_cache_id,
            "cpu_ids": [cpu.id for cpu in cpus],
            "num_cpus": len(cpus),
        }
        for (numa_node, socket_id, l3_cache_id), cpus in grouped_cpus.items()
    ]


def _summarize_times(
    times: list[float],
    measurement_mode: str,
    measurement_iters: int,
    min_runtime_s: float | None,
) -> dict[str, Any]:
    result = {
        "measurement_mode": measurement_mode,
        "measurement_iters": measurement_iters,
        "time_min_ms": min(times),
        "time_max_ms": max(times),
        "time_mean_ms": float(np.mean(times)),
        "time_std_ms": float(np.std(times)),
        "time_median_ms": float(np.median(times)),
    }
    if min_runtime_s is not None:
        result["target_min_runtime_s"] = min_runtime_s
    result["times_ms"] = times
    return result


def measure_attention_run(
    prepared,
    iters: int | None,
    min_runtime_s: float | None,
    runtime_chunk_iters: int = DEFAULT_RUNTIME_CHUNK_ITERS,
) -> dict[str, Any]:
    assert (iters is None) != (min_runtime_s is None), (
        "Exactly one of iters and min_runtime_s must be set."
    )

    if iters is not None:
        times = run_attention_iters(prepared, iters)
        return _summarize_times(
            times=times,
            measurement_mode="iters",
            measurement_iters=iters,
            min_runtime_s=None,
        )

    assert min_runtime_s is not None and min_runtime_s > 0
    assert runtime_chunk_iters > 0

    start_time = time.perf_counter()
    measurement_iters = 0
    time_min_ms = float("inf")
    time_max_ms = float("-inf")
    time_mean_ms = 0.0
    m2 = 0.0

    while True:
        chunk_times = run_attention_iters(prepared, runtime_chunk_iters)
        for t in chunk_times:
            measurement_iters += 1
            time_min_ms = min(time_min_ms, t)
            time_max_ms = max(time_max_ms, t)
            delta = t - time_mean_ms
            time_mean_ms += delta / measurement_iters
            m2 += delta * (t - time_mean_ms)
        if time.perf_counter() - start_time >= min_runtime_s:
            break

    time_std_ms = float(np.sqrt(m2 / measurement_iters))
    return {
        "measurement_mode": "min-runtime-s",
        "measurement_iters": measurement_iters,
        "target_min_runtime_s": min_runtime_s,
        "runtime_chunk_iters": runtime_chunk_iters,
        "time_min_ms": time_min_ms,
        "time_max_ms": time_max_ms,
        "time_mean_ms": time_mean_ms,
        "time_std_ms": time_std_ms,
        "time_median_ms": None,
    }


def _rank_worker(
    rank: int,
    args_dict: dict[str, Any],
    barrier: mp.Barrier,
    result_queue: mp.Queue,
) -> None:
    try:
        configure_attention_debug_env_for_rank(rank)
        profile_enabled = configure_attention_profile_env()
        tp_size = args_dict["tp_size"]
        omp_cpuids = resolve_local_omp_cpuid(
            omp_cpuids=args_dict["omp_threads_bind"],
            rank=rank,
            local_rank=rank,
            world_size=tp_size,
            cpu_arch=args_dict["cpu_arch"],
            reserve_cpu_num=args_dict["reserve_cpu_num"],
            allowed_numa_nodes=args_dict["allowed_numa_nodes"],
            logical_cpu_list=args_dict["logical_cpu_list"],
        )

        if omp_cpuids != "nobind":
            bind_message = torch.ops._C_utils.init_cpu_threads_env(omp_cpuids)
        else:
            bind_message = "nobind"
        locality_groups = summarize_rank_locality_groups(
            omp_cpuids=omp_cpuids,
            logical_cpu_list=args_dict["logical_cpu_list"],
            prefer_runtime_summary=True,
        )

        shard_plan = resolve_head_shard_plan(
            num_query_heads=args_dict["num_query_heads"],
            num_kv_heads=args_dict["num_kv_heads"],
            tp_size=tp_size,
            partition_mode=args_dict["partition_mode"],
        )
        seq_lens = build_seq_lens(
            batch_size=args_dict["batch_size"],
            workload=args_dict["workload"],
            q_len=args_dict["q_len"],
            kv_len=args_dict["kv_len"],
            seed=args_dict["seed"],
        )
        isa = args_dict["isa"] or get_attn_isa(
            args_dict["block_size"], STR_DTYPE_TO_TORCH_DTYPE[args_dict["dtype"]]
        )

        with torch.inference_mode():
            if profile_enabled:
                reset_attention_profile()
            prepared = prepare_attention_run(
                seq_lens=seq_lens,
                num_heads=(
                    shard_plan.local_num_query_heads,
                    shard_plan.local_num_kv_heads,
                ),
                head_size=args_dict["head_size"],
                sliding_window=args_dict["sliding_window"],
                dtype=STR_DTYPE_TO_TORCH_DTYPE[args_dict["dtype"]],
                block_size=args_dict["block_size"],
                num_blocks=args_dict["num_blocks"],
                use_sink=args_dict["use_sink"],
                enable_kv_split=args_dict["enable_kv_split"],
                isa=isa,
                seed=args_dict["seed"] + rank,
                # Keep only the synthetic request/block layout identical across ranks.
                block_table_seed=args_dict["seed"],
                locality_mode=args_dict["attn_locality_mode"],
                locality_group_span=args_dict["attn_locality_group_span"],
            )
            if profile_enabled:
                reset_attention_runtime_profile()

            if args_dict["warmup_iters"] > 0:
                run_attention_iters(prepared, args_dict["warmup_iters"])
                if profile_enabled:
                    reset_attention_runtime_profile()

            barrier.wait()
            start_time = time.perf_counter_ns()
            result = measure_attention_run(
                prepared=prepared,
                iters=args_dict["iters"],
                min_runtime_s=args_dict["min_runtime_s"],
            )
            end_time = time.perf_counter_ns()
            profile = collect_attention_profile(args_dict["attn_locality_mode"])

        result_queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "omp_cpuids": omp_cpuids,
                "bind_message": bind_message,
                "locality_groups": locality_groups,
                "head_plan": asdict(shard_plan),
                "elapsed_ms": (end_time - start_time) / 1e6,
                "profile": profile,
                "result": result,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "rank": rank,
                "error": repr(exc),
            }
        )


def run_multiprocess_benchmark(args) -> dict[str, Any]:
    shard_plan = resolve_head_shard_plan(
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        tp_size=args.tp_size,
        partition_mode=args.partition_mode,
    )
    seq_lens = build_seq_lens(
        batch_size=args.batch_size,
        workload=args.workload,
        q_len=args.q_len,
        kv_len=args.kv_len,
        seed=args.seed,
    )
    resolved_q_len, resolved_kv_len = resolve_workload_lengths(
        workload=args.workload,
        q_len=args.q_len,
        kv_len=args.kv_len,
    )
    result_shape_dirname = build_result_shape_dirname(
        batch_size=args.batch_size,
        q_len=resolved_q_len,
        kv_len=resolved_kv_len,
    )
    allowed_numa_nodes, logical_cpu_list = CpuPlatform.get_allowed_cpu_core_node_list()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.tp_size)
    result_queue = ctx.Queue()

    args_dict = {
        "tp_size": args.tp_size,
        "partition_mode": args.partition_mode,
        "workload": args.workload,
        "batch_size": args.batch_size,
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "seed": args.seed,
        "num_query_heads": args.num_query_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_size": args.head_size,
        "sliding_window": args.sliding_window,
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "dtype": args.dtype,
        "use_sink": args.use_sink,
        "enable_kv_split": args.enable_kv_split,
        "isa": args.isa,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "min_runtime_s": args.min_runtime_s,
        "omp_threads_bind": args.omp_threads_bind,
        "reserve_cpu_num": args.reserve_cpu_num,
        "cpu_arch": args.cpu_arch,
        "attn_locality_mode": args.attn_locality_mode,
        "attn_locality_group_span": args.attn_locality_group_span,
        "allowed_numa_nodes": allowed_numa_nodes,
        "logical_cpu_list": logical_cpu_list,
    }

    processes = [
        ctx.Process(
            target=_rank_worker,
            args=(rank, args_dict, barrier, result_queue),
        )
        for rank in range(args.tp_size)
    ]
    for process in processes:
        process.start()

    rank_results = [result_queue.get() for _ in range(args.tp_size)]

    for process in processes:
        process.join()

    rank_results.sort(key=lambda item: item["rank"])
    errors = [item for item in rank_results if "error" in item]
    if errors:
        raise RuntimeError(f"Rank worker failed: {errors}")

    slowest_rank_mean_ms = max(item["result"]["time_mean_ms"] for item in rank_results)
    profile_summary = aggregate_attention_profiles(
        [item.get("profile") for item in rank_results]
    )
    return {
        "workload": args.workload,
        "batch_size": args.batch_size,
        "requested_lengths": {
            "q_len": args.q_len,
            "kv_len": args.kv_len,
        },
        "resolved_lengths": {
            "q_len": resolved_q_len,
            "kv_len": resolved_kv_len,
        },
        "result_shape_dirname": result_shape_dirname,
        "seq_lens": seq_lens,
        "head_plan": asdict(shard_plan),
        "attn_locality_mode": args.attn_locality_mode,
        "attn_locality_group_span": args.attn_locality_group_span,
        "allowed_numa_nodes": allowed_numa_nodes,
        "rank_results": rank_results,
        "profile_summary": profile_summary,
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
    }


def add_cli_args(parser: FlexibleArgumentParser) -> None:
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument(
        "--partition-mode",
        choices=["global-fixed", "per-rank-fixed"],
        default="global-fixed",
    )
    parser.add_argument(
        "--workload",
        choices=["decode-like", "prefill-like", "custom"],
        default="decode-like",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--q-len", type=int, default=None)
    parser.add_argument("--kv-len", type=int, default=None)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=CPUAttentionBackend.get_supported_head_sizes(),
        default=128,
    )
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--block-size", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--use-sink", action="store_true")
    parser.add_argument("--enable-kv-split", action="store_true")
    parser.add_argument(
        "--attn-locality-mode",
        choices=["balanced", "acc-local-l3"],
        default="balanced",
    )
    parser.add_argument("--attn-locality-group-span", type=int, default=1)
    parser.add_argument(
        "--isa", type=str, choices=["vec", "neon", "amx", "vec16"], default=None
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    measurement_group = parser.add_mutually_exclusive_group(required=True)
    measurement_group.add_argument("--iters", type=int, default=None)
    measurement_group.add_argument("--min-runtime-s", type=float, default=None)
    parser.add_argument(
        "--omp-threads-bind",
        type=str,
        default=envs.VLLM_CPU_OMP_THREADS_BIND,
    )
    parser.add_argument(
        "--reserve-cpu-num",
        type=int,
        default=envs.VLLM_CPU_NUM_OF_RESERVED_CPU,
    )
    parser.add_argument("--output-json", type=str, default=None)


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Run attention-only CPU benchmark with TP-like multiprocess binding."
    )
    add_cli_args(parser)
    args = parser.parse_args()
    args.cpu_arch = CpuPlatform.get_cpu_architecture()

    summary = run_multiprocess_benchmark(args)
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
