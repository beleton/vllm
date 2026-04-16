# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import benchmarks.kernels.cpu.benchmark_cpu_attn as benchmark_cpu_attn
import benchmarks.kernels.cpu.benchmark_cpu_attn_mp as benchmark_cpu_attn_mp
from benchmarks.kernels.cpu.benchmark_cpu_attn_mp import (
    add_cli_args,
    build_result_shape_dirname,
    measure_attention_run,
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
        num_blocks=1,
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
        num_blocks=1,
        enable_kv_split=True,
        isa="vec",
        seed=0,
        locality_mode="acc-local-l3",
        locality_group_span=4,
    )

    assert calls["scheduler_kwargs"]["group_span"] == 4


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
