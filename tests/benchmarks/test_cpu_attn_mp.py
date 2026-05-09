# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import benchmarks.kernels.cpu.benchmark_cpu_attn as benchmark_cpu_attn
import benchmarks.kernels.cpu.benchmark_cpu_attn_mp as benchmark_cpu_attn_mp
from benchmarks.kernels.cpu.benchmark_cpu_attn_mp import (
    aggregate_attention_profiles,
    add_cli_args,
    build_result_shape_dirname,
    collect_attention_profile,
    configure_attention_profile_env,
    configure_attention_debug_env_for_rank,
    measure_attention_run,
    reset_attention_profile,
    reset_attention_runtime_profile,
    resolve_head_shard_plan,
    resolve_workload_lengths,
    summarize_rank_locality_groups,
)
from vllm.platforms.cpu import LogicalCPUInfo
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_resolve_head_shard_plan_global_fixed_splits_heads_across_tp():
    plan = resolve_head_shard_plan(
        num_query_heads=32,
        num_kv_heads=8,
        tp_size=4,
        partition_mode="global-fixed",
    )

    assert plan.local_num_query_heads == 8
    assert plan.local_num_kv_heads == 2
    assert plan.global_num_query_heads == 32
    assert plan.global_num_kv_heads == 8


def test_resolve_head_shard_plan_per_rank_fixed_keeps_local_heads_constant():
    plan = resolve_head_shard_plan(
        num_query_heads=32,
        num_kv_heads=8,
        tp_size=4,
        partition_mode="per-rank-fixed",
    )

    assert plan.local_num_query_heads == 32
    assert plan.local_num_kv_heads == 8
    assert plan.global_num_query_heads == 128
    assert plan.global_num_kv_heads == 32


def _make_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    return parser


def test_cli_rejects_missing_iters_and_min_runtime_s():
    parser = _make_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--tp-size", "2"])


def test_cli_rejects_both_iters_and_min_runtime_s():
    parser = _make_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--tp-size",
                "2",
                "--iters",
                "10",
                "--min-runtime-s",
                "1.0",
            ]
        )


def test_cli_accepts_batch_size_and_q_kv_len():
    parser = _make_parser()

    args = parser.parse_args(
        [
            "--tp-size",
            "2",
            "--workload",
            "decode-like",
            "--batch-size",
            "8",
            "--q-len",
            "1",
            "--kv-len",
            "1024",
            "--iters",
            "10",
        ]
    )

    assert args.batch_size == 8
    assert args.q_len == 1
    assert args.kv_len == 1024


def test_cli_accepts_attn_locality_flags():
    parser = _make_parser()

    args = parser.parse_args(
        [
            "--tp-size",
            "2",
            "--iters",
            "10",
            "--attn-locality-mode",
            "acc-local-l3",
            "--attn-locality-group-span",
            "2",
        ]
    )

    assert args.attn_locality_mode == "acc-local-l3"
    assert args.attn_locality_group_span == 2


def test_cli_rejects_deprecated_input_output_len_flags():
    parser = _make_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--tp-size",
                "2",
                "--workload",
                "decode-like",
                "--input-len",
                "1",
                "--output-len",
                "1024",
                "--iters",
                "10",
            ]
        )


def test_measure_attention_run_with_iters_keeps_full_iteration_times(monkeypatch):
    monkeypatch.setattr(
        benchmark_cpu_attn_mp,
        "run_attention_iters",
        lambda prepared, iters: [float(i) for i in range(iters)],
    )

    result = measure_attention_run(
        prepared=object(),
        iters=4,
        min_runtime_s=None,
    )

    assert result["measurement_mode"] == "iters"
    assert result["measurement_iters"] == 4
    assert result["times_ms"] == [0.0, 1.0, 2.0, 3.0]
    assert result["time_mean_ms"] == 1.5
    assert result["time_median_ms"] == 1.5


def test_measure_attention_run_with_min_runtime_s_aggregates_only(monkeypatch):
    clock_values = iter([0.0, 0.04, 0.08, 0.12])
    monkeypatch.setattr(
        benchmark_cpu_attn_mp.time,
        "perf_counter",
        lambda: next(clock_values),
    )
    monkeypatch.setattr(
        benchmark_cpu_attn_mp,
        "run_attention_iters",
        lambda prepared, iters: [1.0] * iters,
    )

    result = measure_attention_run(
        prepared=object(),
        iters=None,
        min_runtime_s=0.1,
        runtime_chunk_iters=2,
    )

    assert result["measurement_mode"] == "min-runtime-s"
    assert result["measurement_iters"] == 6
    assert result["time_mean_ms"] == 1.0
    assert result["time_median_ms"] is None
    assert "times_ms" not in result


def _build_fake_topology() -> list[LogicalCPUInfo]:
    return [
        LogicalCPUInfo(
            id=0,
            physical_core=0,
            numa_node=0,
            socket_id=0,
            l3_cache_id=10,
        ),
        LogicalCPUInfo(
            id=1,
            physical_core=1,
            numa_node=0,
            socket_id=0,
            l3_cache_id=10,
        ),
        LogicalCPUInfo(
            id=2,
            physical_core=2,
            numa_node=0,
            socket_id=0,
            l3_cache_id=11,
        ),
        LogicalCPUInfo(
            id=3,
            physical_core=3,
            numa_node=0,
            socket_id=0,
            l3_cache_id=11,
        ),
    ]


def test_summarize_rank_locality_groups_groups_bound_cpus_by_l3():
    summary = summarize_rank_locality_groups(
        omp_cpuids="0,1,3",
        logical_cpu_list=_build_fake_topology(),
    )

    assert summary == [
        {
            "numa_node": 0,
            "socket_id": 0,
            "l3_cache_id": 10,
            "cpu_ids": [0, 1],
            "num_cpus": 2,
        },
        {
            "numa_node": 0,
            "socket_id": 0,
            "l3_cache_id": 11,
            "cpu_ids": [3],
            "num_cpus": 1,
        },
    ]


def test_prepare_attention_run_uses_selected_locality_mode(monkeypatch):
    calls = {}
    scheduler_metadata = torch.ones(4, dtype=torch.int32)
    attention_op = object()

    def fake_get_cpu_attn_ops(mode):
        calls["mode"] = mode

        def fake_scheduler(**kwargs):
            calls["scheduler_kwargs"] = kwargs
            return scheduler_metadata

        return fake_scheduler, attention_op

    monkeypatch.setattr(benchmark_cpu_attn, "_get_cpu_attn_ops", fake_get_cpu_attn_ops)
    monkeypatch.setattr(
        benchmark_cpu_attn,
        "cpu_attn_reshape_and_cache",
        lambda **kwargs: None,
    )

    prepared = benchmark_cpu_attn.prepare_attention_run(
        seq_lens=[(1, 1)],
        num_heads=(1, 1),
        head_size=32,
        dtype=torch.float32,
        block_size=32,
        num_blocks=2,
        enable_kv_split=True,
        isa="vec",
        seed=0,
        locality_mode="acc-local-l3",
    )

    assert calls["mode"] == "acc-local-l3"
    assert calls["scheduler_kwargs"]["enable_kv_split"] is True
    assert prepared.scheduler_metadata is scheduler_metadata
    assert prepared.attention_op is attention_op


def test_prepare_attention_run_passes_group_span_to_acc_locality_scheduler(
    monkeypatch,
):
    calls = {}
    scheduler_metadata = torch.ones(4, dtype=torch.int32)

    def fake_get_cpu_attn_ops(mode):
        def fake_scheduler(**kwargs):
            calls["scheduler_kwargs"] = kwargs
            return scheduler_metadata

        return fake_scheduler, object()

    monkeypatch.setattr(benchmark_cpu_attn, "_get_cpu_attn_ops", fake_get_cpu_attn_ops)
    monkeypatch.setattr(
        benchmark_cpu_attn,
        "cpu_attn_reshape_and_cache",
        lambda **kwargs: None,
    )

    benchmark_cpu_attn.prepare_attention_run(
        seq_lens=[(1, 1)],
        num_heads=(1, 1),
        head_size=32,
        dtype=torch.float32,
        block_size=32,
        num_blocks=2,
        enable_kv_split=True,
        isa="vec",
        seed=0,
        locality_mode="acc-local-l3",
        locality_group_span=4,
    )

    assert calls["scheduler_kwargs"]["group_span"] == 4


def test_prepare_attention_run_can_share_block_table_seed_only(monkeypatch):
    scheduler_metadata = torch.ones(4, dtype=torch.int32)

    def fake_get_cpu_attn_ops(mode):
        return lambda **kwargs: scheduler_metadata, object()

    monkeypatch.setattr(benchmark_cpu_attn, "_get_cpu_attn_ops", fake_get_cpu_attn_ops)
    monkeypatch.setattr(
        benchmark_cpu_attn,
        "cpu_attn_reshape_and_cache",
        lambda **kwargs: None,
    )

    first = benchmark_cpu_attn.prepare_attention_run(
        seq_lens=[(64, 64)],
        num_heads=(1, 1),
        head_size=32,
        dtype=torch.float32,
        block_size=32,
        num_blocks=128,
        enable_kv_split=True,
        isa="vec",
        seed=1,
        block_table_seed=123,
        locality_mode="balanced",
    )
    second = benchmark_cpu_attn.prepare_attention_run(
        seq_lens=[(64, 64)],
        num_heads=(1, 1),
        head_size=32,
        dtype=torch.float32,
        block_size=32,
        num_blocks=128,
        enable_kv_split=True,
        isa="vec",
        seed=2,
        block_table_seed=123,
        locality_mode="balanced",
    )

    torch.testing.assert_close(first.block_table, second.block_table)


def test_prepare_attention_run_samples_block_table_without_replacement_excluding_zero(
    monkeypatch,
):
    scheduler_metadata = torch.ones(4, dtype=torch.int32)

    def fake_get_cpu_attn_ops(mode):
        return lambda **kwargs: scheduler_metadata, object()

    monkeypatch.setattr(benchmark_cpu_attn, "_get_cpu_attn_ops", fake_get_cpu_attn_ops)
    monkeypatch.setattr(
        benchmark_cpu_attn,
        "cpu_attn_reshape_and_cache",
        lambda **kwargs: None,
    )

    prepared = benchmark_cpu_attn.prepare_attention_run(
        seq_lens=[(128, 128)],
        num_heads=(1, 1),
        head_size=32,
        dtype=torch.float32,
        block_size=32,
        num_blocks=5,
        enable_kv_split=True,
        isa="vec",
        seed=0,
        block_table_seed=0,
        locality_mode="balanced",
    )

    block_ids = prepared.block_table[0].tolist()
    assert len(set(block_ids)) == len(block_ids)
    assert min(block_ids) > 0


def test_rank_worker_uses_shared_block_table_seed_but_rank_data_seed(monkeypatch):
    captured: dict[str, int] = {}
    inference_events: list[str] = []

    class FakeInferenceMode:
        def __enter__(self):
            inference_events.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            inference_events.append("exit")

    class FakeBarrier:
        def wait(self):
            pass

    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    monkeypatch.delenv("VLLM_CPU_ATTN_PROFILE", raising=False)
    monkeypatch.setattr(
        benchmark_cpu_attn_mp,
        "resolve_local_omp_cpuid",
        lambda **kwargs: "nobind",
    )
    monkeypatch.setattr(
        benchmark_cpu_attn_mp.torch,
        "inference_mode",
        lambda: FakeInferenceMode(),
    )

    def fake_prepare_attention_run(**kwargs):
        captured["seed"] = kwargs["seed"]
        captured["block_table_seed"] = kwargs["block_table_seed"]
        return object()

    monkeypatch.setattr(
        benchmark_cpu_attn_mp,
        "prepare_attention_run",
        fake_prepare_attention_run,
    )
    monkeypatch.setattr(
        benchmark_cpu_attn_mp,
        "measure_attention_run",
        lambda **kwargs: {
            "time_mean_ms": 1.0,
            "time_min_ms": 1.0,
            "time_max_ms": 1.0,
            "time_std_ms": 0.0,
            "time_median_ms": 1.0,
            "measurement_mode": "iters",
            "measurement_iters": 1,
            "times_ms": [1.0],
        },
    )

    args_dict = {
        "tp_size": 2,
        "partition_mode": "global-fixed",
        "workload": "prefill-like",
        "batch_size": 1,
        "q_len": 128,
        "kv_len": 128,
        "seed": 123,
        "num_query_heads": 32,
        "num_kv_heads": 16,
        "head_size": 128,
        "sliding_window": None,
        "block_size": 128,
        "num_blocks": 4096,
        "dtype": "bfloat16",
        "use_sink": False,
        "enable_kv_split": True,
        "isa": "vec",
        "warmup_iters": 0,
        "iters": 1,
        "min_runtime_s": None,
        "omp_threads_bind": "nobind",
        "reserve_cpu_num": 0,
        "cpu_arch": None,
        "attn_locality_mode": "balanced",
        "attn_locality_group_span": 1,
        "allowed_numa_nodes": [0, 1],
        "logical_cpu_list": [],
    }

    result_queue = FakeQueue()
    benchmark_cpu_attn_mp._rank_worker(
        rank=1,
        args_dict=args_dict,
        barrier=FakeBarrier(),
        result_queue=result_queue,
    )

    assert captured["seed"] == args_dict["seed"] + 1
    assert captured["block_table_seed"] == args_dict["seed"]
    assert inference_events == ["enter", "exit"]
    assert "error" not in result_queue.items[0]


def test_resolve_workload_lengths_decode_like_supports_q_kv_len():
    q_len, kv_len = resolve_workload_lengths(
        workload="decode-like",
        q_len=1,
        kv_len=1024,
    )

    assert q_len == 1
    assert kv_len == 1024


def test_resolve_workload_lengths_prefill_like_supports_q_kv_len():
    q_len, kv_len = resolve_workload_lengths(
        workload="prefill-like",
        q_len=1024,
        kv_len=1024,
    )

    assert q_len == 1024
    assert kv_len == 1024


def test_resolve_workload_lengths_decode_like_defaults_when_q_kv_len_omitted():
    q_len, kv_len = resolve_workload_lengths(
        workload="decode-like",
        q_len=None,
        kv_len=None,
    )

    assert q_len == 1
    assert kv_len == 1024


def test_build_result_shape_dirname_uses_resolved_q_kv_len():
    dirname = build_result_shape_dirname(
        batch_size=16,
        q_len=1,
        kv_len=1024,
    )

    assert dirname == "batch_16/q1_KV_1024"


def test_build_result_shape_dirname_uses_q_kv_len_for_custom_workload():
    dirname = build_result_shape_dirname(
        batch_size=8,
        q_len=32,
        kv_len=2048,
    )

    assert dirname == "batch_8/q32_KV_2048"


def test_configure_attention_profile_env_enables_silent_collection(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_ATTN_PROFILE", "1")
    monkeypatch.delenv("VLLM_CPU_ATTN_PROFILE_SILENT", raising=False)

    enabled = configure_attention_profile_env()

    assert enabled is True
    assert benchmark_cpu_attn_mp.os.environ["VLLM_CPU_ATTN_PROFILE_SILENT"] == "1"


def test_reset_attention_profile_uses_custom_op_when_profile_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_ATTN_PROFILE", "1")
    calls: list[str] = []

    monkeypatch.setattr(
        benchmark_cpu_attn_mp.torch.ops._C,
        "cpu_attn_reset_timing_profile",
        lambda: calls.append("reset_all"),
        raising=False,
    )

    reset_attention_profile()

    assert calls == ["reset_all"]


def test_reset_attention_runtime_profile_uses_custom_op_when_profile_enabled(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_CPU_ATTN_PROFILE", "1")
    calls: list[str] = []

    monkeypatch.setattr(
        benchmark_cpu_attn_mp.torch.ops._C,
        "cpu_attn_reset_runtime_timing_profile",
        lambda: calls.append("reset_runtime"),
        raising=False,
    )

    reset_attention_runtime_profile()

    assert calls == ["reset_runtime"]


def test_collect_attention_profile_selects_requested_mode_and_derives_metrics(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_CPU_ATTN_PROFILE", "1")

    monkeypatch.setattr(
        benchmark_cpu_attn_mp.torch.ops._C,
        "cpu_attn_get_timing_profile",
        lambda: """
        {
          "balanced": {
            "scheduler": {
              "call_count": 0,
              "scheduler_metadata_ns": 0,
              "legacy_scheduler_metadata_ns": 0,
              "locality_metadata_build_ns": 0
            },
            "runtime": {
              "call_count": 0,
              "counter_fetch_ns": 0,
              "counter_fetch_count": 0,
              "attention_counter_fetch_ns": 0,
              "attention_counter_fetch_count": 0,
              "reduction_counter_fetch_ns": 0,
              "reduction_counter_fetch_count": 0,
              "attention_task_body_ns": 0,
              "attention_task_count": 0,
              "reduction_task_body_ns": 0,
              "reduction_task_count": 0,
              "execute_attention_ns": 0,
              "execute_attention_count": 0
            }
          },
          "acc_locality": {
            "scheduler": {
              "call_count": 1,
              "scheduler_metadata_ns": 30,
              "legacy_scheduler_metadata_ns": 18,
              "locality_metadata_build_ns": 12
            },
            "runtime": {
              "call_count": 10,
              "counter_fetch_ns": 0,
              "counter_fetch_count": 0,
              "attention_counter_fetch_ns": 6,
              "attention_counter_fetch_count": 3,
              "reduction_counter_fetch_ns": 4,
              "reduction_counter_fetch_count": 2,
              "attention_task_body_ns": 80,
              "attention_task_count": 8,
              "reduction_task_body_ns": 20,
              "reduction_task_count": 4,
              "execute_attention_ns": 50,
              "execute_attention_count": 5
            }
          }
        }
        """,
        raising=False,
    )

    profile = collect_attention_profile("acc-local-l3")

    assert profile == {
        "mode": "acc-local-l3",
        "scheduler": {
            "call_count": 1,
            "scheduler_metadata_ns": 30,
            "scheduler_metadata_avg_ns": 30.0,
            "legacy_scheduler_metadata_ns": 18,
            "locality_metadata_build_ns": 12,
        },
        "runtime": {
            "call_count": 10,
            "attention_counter_fetch_ns": 6,
            "attention_counter_fetch_count": 3,
            "reduction_counter_fetch_ns": 4,
            "reduction_counter_fetch_count": 2,
            "attention_task_body_ns": 80,
            "attention_task_count": 8,
            "attention_task_body_avg_ns": 10.0,
            "reduction_task_body_ns": 20,
            "reduction_task_count": 4,
            "reduction_task_body_avg_ns": 5.0,
            "execute_attention_ns": 50,
            "execute_attention_count": 5,
            "execute_attention_avg_ns": 10.0,
            "attention_task_non_execute_ns": 30,
        },
    }


def test_aggregate_attention_profiles_sums_rank_profiles():
    aggregate = aggregate_attention_profiles(
        [
            {
                "mode": "balanced",
                "scheduler": {
                    "call_count": 1,
                    "scheduler_metadata_ns": 10,
                    "scheduler_metadata_avg_ns": 10.0,
                },
                "runtime": {
                    "call_count": 2,
                    "attention_task_body_ns": 20,
                    "attention_task_count": 2,
                    "attention_task_body_avg_ns": 10.0,
                    "reduction_task_body_ns": 8,
                    "reduction_task_count": 2,
                    "reduction_task_body_avg_ns": 4.0,
                    "execute_attention_ns": 12,
                    "execute_attention_count": 2,
                    "execute_attention_avg_ns": 6.0,
                    "attention_task_non_execute_ns": 8,
                },
            },
            {
                "mode": "balanced",
                "scheduler": {
                    "call_count": 1,
                    "scheduler_metadata_ns": 14,
                    "scheduler_metadata_avg_ns": 14.0,
                },
                "runtime": {
                    "call_count": 3,
                    "attention_task_body_ns": 45,
                    "attention_task_count": 3,
                    "attention_task_body_avg_ns": 15.0,
                    "reduction_task_body_ns": 10,
                    "reduction_task_count": 2,
                    "reduction_task_body_avg_ns": 5.0,
                    "execute_attention_ns": 30,
                    "execute_attention_count": 5,
                    "execute_attention_avg_ns": 6.0,
                    "attention_task_non_execute_ns": 15,
                },
            },
        ]
    )

    assert aggregate == {
        "mode": "balanced",
        "scheduler": {
            "call_count": 2,
            "scheduler_metadata_ns": 24,
            "scheduler_metadata_avg_ns": 12.0,
        },
        "runtime": {
            "call_count": 5,
            "attention_task_body_ns": 65,
            "attention_task_count": 5,
            "attention_task_body_avg_ns": 13.0,
            "reduction_task_body_ns": 18,
            "reduction_task_count": 4,
            "reduction_task_body_avg_ns": 4.5,
            "execute_attention_ns": 42,
            "execute_attention_count": 7,
            "execute_attention_avg_ns": 6.0,
            "attention_task_non_execute_ns": 23,
        },
    }


def test_configure_attention_debug_env_for_rank_defaults_to_rank0_only(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_ATTN_DEBUG", "1")
    monkeypatch.delenv("VLLM_CPU_ATTN_DEBUG_ALL_RANKS", raising=False)

    configure_attention_debug_env_for_rank(1)

    assert benchmark_cpu_attn_mp.os.environ["VLLM_CPU_ATTN_DEBUG"] == "0"
    assert benchmark_cpu_attn_mp.os.environ["VLLM_CPU_ATTN_DEBUG_RANK"] == "1"


def test_configure_attention_debug_env_for_rank_keeps_rank0_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_ATTN_DEBUG", "1")
    monkeypatch.delenv("VLLM_CPU_ATTN_DEBUG_ALL_RANKS", raising=False)

    configure_attention_debug_env_for_rank(0)

    assert benchmark_cpu_attn_mp.os.environ["VLLM_CPU_ATTN_DEBUG"] == "1"
    assert benchmark_cpu_attn_mp.os.environ["VLLM_CPU_ATTN_DEBUG_RANK"] == "0"


def test_configure_attention_debug_env_for_rank_can_keep_all_ranks_enabled(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_CPU_ATTN_DEBUG", "1")
    monkeypatch.setenv("VLLM_CPU_ATTN_DEBUG_ALL_RANKS", "1")

    configure_attention_debug_env_for_rank(1)

    assert benchmark_cpu_attn_mp.os.environ["VLLM_CPU_ATTN_DEBUG"] == "1"
    assert benchmark_cpu_attn_mp.os.environ["VLLM_CPU_ATTN_DEBUG_RANK"] == "1"
