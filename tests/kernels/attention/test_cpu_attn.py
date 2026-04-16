# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import math

import pytest
import torch

import vllm._custom_ops as ops_module
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import CpuPlatform, supports_amx_tiles
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.cpu_attn import _get_attn_isa, _get_cpu_attn_ops
from vllm.v1.worker.cpu_binding import group_logical_cpus_by_l3

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attention_with_kv_cache_acc_locality,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_get_scheduler_metadata_acc_locality,
    cpu_attn_reshape_and_cache,
)


def test_cpu_attention_acc_locality_wrappers_are_exported_without_replacing_legacy_ops():
    assert callable(cpu_attention_with_kv_cache)
    assert callable(cpu_attn_get_scheduler_metadata)
    assert hasattr(ops_module, "cpu_attention_with_kv_cache_acc_locality")
    assert hasattr(ops_module, "cpu_attn_get_scheduler_metadata_acc_locality")


def test_cpu_attention_backend_defaults_to_legacy_wrappers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_CPU_ATTN_LOCALITY_MODE", raising=False)

    scheduler_op, attn_op = _get_cpu_attn_ops()

    assert scheduler_op is ops_module.cpu_attn_get_scheduler_metadata
    assert attn_op is ops_module.cpu_attention_with_kv_cache


def test_cpu_attention_backend_can_switch_to_acc_locality_wrappers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_CPU_ATTN_LOCALITY_MODE", "acc-local-l3")

    scheduler_op, attn_op = _get_cpu_attn_ops()

    assert scheduler_op is ops_module.cpu_attn_get_scheduler_metadata_acc_locality
    assert attn_op is ops_module.cpu_attention_with_kv_cache_acc_locality


def _select_two_runtime_l3_groups() -> tuple[str, int]:
    _, logical_cpu_list = CpuPlatform.get_allowed_cpu_core_node_list()
    grouped = group_logical_cpus_by_l3(logical_cpu_list)
    if len(grouped) < 2:
        pytest.skip("Current environment does not expose at least two L3 groups.")

    selected_cpu_ids: list[int] = []
    for cpus in list(grouped.values())[:2]:
        selected_cpu_ids.append(cpus[0].id)
    return ",".join(str(cpu_id) for cpu_id in selected_cpu_ids), len(selected_cpu_ids)


def _select_runtime_cpu_ids(max_cpu_num: int = 2) -> str:
    _, logical_cpu_list = CpuPlatform.get_allowed_cpu_core_node_list()
    if not logical_cpu_list:
        pytest.skip("Current environment does not expose CPU topology.")
    selected_cpu_ids = [cpu.id for cpu in logical_cpu_list[:max_cpu_num]]
    return ",".join(str(cpu_id) for cpu_id in selected_cpu_ids)


def _prepare_small_cpu_attention_case(
    isa: str = "vec",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | float | tuple[int, int]]:
    seq_lens = [(32, 128)]
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 32
    block_size = 32
    num_blocks = 16
    scale = head_size**-0.5
    token_num = sum(query_lens)

    query = torch.randn(token_num, num_query_heads, head_size, dtype=dtype)
    key_value = torch.randn(
        2, num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache, value_cache = key_value.unbind(0)
    packed_key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
    )
    packed_value_cache = torch.empty_like(packed_key_cache)
    slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
    )

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = max(kv_lens) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    return {
        "query": query,
        "packed_key_cache": packed_key_cache,
        "packed_value_cache": packed_value_cache,
        "query_start_loc": cu_query_lens,
        "seq_lens": kv_lens_tensor,
        "block_tables": block_tables,
        "scale": scale,
        "window_size": (-1, -1),
    }


def test_cpu_attention_acc_locality_metadata_exposes_l3_subgroups():
    omp_cpuids, thread_num = _select_two_runtime_l3_groups()
    torch.ops._C_utils.init_cpu_threads_env(omp_cpuids)

    case = _prepare_small_cpu_attention_case()
    metadata = cpu_attn_get_scheduler_metadata_acc_locality(
        num_reqs=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        seq_lens=case["seq_lens"],
        dtype=torch.float32,
        query_start_loc=case["query_start_loc"],
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=True,
    )

    summary = json.loads(torch.ops._C_utils.inspect_cpu_attn_acc_locality_metadata(metadata))

    assert summary["subgroup_num"] == 2
    assert summary["thread_num"] == thread_num
    assert summary["actual_kv_head_num"] == 2
    assert summary["kv_head_to_subgroup"] == [0, 1]
    assert summary["attention_task_num"] == 2
    assert summary["legacy_effective_thread_num"] >= 1
    assert summary["legacy_attention_task_num"] == (
        summary["actual_kv_head_num"] * summary["legacy_effective_thread_num"]
    )
    assert summary["attention_task_num"] <= summary["legacy_attention_task_num"]


def test_cpu_attention_acc_locality_metadata_group_span_expands_kv_head_coverage():
    omp_cpuids, thread_num = _select_two_runtime_l3_groups()
    torch.ops._C_utils.init_cpu_threads_env(omp_cpuids)

    case = _prepare_small_cpu_attention_case()
    metadata = cpu_attn_get_scheduler_metadata_acc_locality(
        num_reqs=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        seq_lens=case["seq_lens"],
        dtype=torch.float32,
        query_start_loc=case["query_start_loc"],
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=True,
        group_span=2,
    )

    summary = json.loads(torch.ops._C_utils.inspect_cpu_attn_acc_locality_metadata(metadata))

    assert summary["thread_num"] == thread_num
    assert summary["subgroup_num"] == 2
    assert summary["group_span"] == 2
    assert summary["actual_kv_head_num"] == 2
    assert summary["kv_head_to_subgroup"] == [0, 0]
    assert summary["attention_task_num"] == 4
    assert summary["legacy_attention_task_num"] >= summary["attention_task_num"]


def test_cpu_attention_acc_locality_path_matches_legacy_output_for_small_case():
    omp_cpuids = _select_runtime_cpu_ids()
    torch.ops._C_utils.init_cpu_threads_env(omp_cpuids)

    case = _prepare_small_cpu_attention_case()
    legacy_metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        seq_lens=case["seq_lens"],
        dtype=torch.float32,
        query_start_loc=case["query_start_loc"],
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=True,
    )
    locality_metadata = cpu_attn_get_scheduler_metadata_acc_locality(
        num_reqs=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        seq_lens=case["seq_lens"],
        dtype=torch.float32,
        query_start_loc=case["query_start_loc"],
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=True,
    )

    legacy_output = torch.empty_like(case["query"])
    locality_output = torch.empty_like(case["query"])
    cpu_attention_with_kv_cache(
        query=case["query"],
        key_cache=case["packed_key_cache"],
        value_cache=case["packed_value_cache"],
        output=legacy_output,
        query_start_loc=case["query_start_loc"],
        seq_lens=case["seq_lens"],
        scale=case["scale"],
        causal=True,
        alibi_slopes=None,
        sliding_window=case["window_size"],
        block_table=case["block_tables"],
        softcap=0,
        scheduler_metadata=legacy_metadata,
        s_aux=None,
    )
    cpu_attention_with_kv_cache_acc_locality(
        query=case["query"],
        key_cache=case["packed_key_cache"],
        value_cache=case["packed_value_cache"],
        output=locality_output,
        query_start_loc=case["query_start_loc"],
        seq_lens=case["seq_lens"],
        scale=case["scale"],
        causal=True,
        alibi_slopes=None,
        sliding_window=case["window_size"],
        block_table=case["block_tables"],
        softcap=0,
        scheduler_metadata=locality_metadata,
        s_aux=None,
    )

    torch.testing.assert_close(locality_output, legacy_output, atol=1e-4, rtol=1e-4)


def test_cpu_attention_acc_locality_runtime_log_exposes_task_partition(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    omp_cpuids, _ = _select_two_runtime_l3_groups()
    torch.ops._C_utils.init_cpu_threads_env(omp_cpuids)
    monkeypatch.setenv("VLLM_CPU_ATTN_DEBUG", "1")

    case = _prepare_small_cpu_attention_case()
    metadata = cpu_attn_get_scheduler_metadata_acc_locality(
        num_reqs=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        seq_lens=case["seq_lens"],
        dtype=torch.float32,
        query_start_loc=case["query_start_loc"],
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=True,
    )

    output = torch.empty_like(case["query"])
    capfd.readouterr()
    cpu_attention_with_kv_cache_acc_locality(
        query=case["query"],
        key_cache=case["packed_key_cache"],
        value_cache=case["packed_value_cache"],
        output=output,
        query_start_loc=case["query_start_loc"],
        seq_lens=case["seq_lens"],
        scale=case["scale"],
        causal=True,
        alibi_slopes=None,
        sliding_window=case["window_size"],
        block_table=case["block_tables"],
        softcap=0,
        scheduler_metadata=metadata,
        s_aux=None,
    )
    captured = capfd.readouterr()

    assert "CPU attention acc-locality runtime summary" in captured.out
    assert "subgroup 0" in captured.out
    assert "attention_task_num=" in captured.out
    assert "legacy_attention_task_num=" in captured.out
    assert "kv_heads=" in captured.out
    assert "legacy_slots=" in captured.out


def test_cpu_attention_acc_locality_group_span_runtime_log_shards_legacy_slots(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    omp_cpuids, _ = _select_two_runtime_l3_groups()
    torch.ops._C_utils.init_cpu_threads_env(omp_cpuids)
    monkeypatch.setenv("VLLM_CPU_ATTN_DEBUG", "1")

    case = _prepare_small_cpu_attention_case()
    metadata = cpu_attn_get_scheduler_metadata_acc_locality(
        num_reqs=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        seq_lens=case["seq_lens"],
        dtype=torch.float32,
        query_start_loc=case["query_start_loc"],
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=True,
        group_span=2,
    )
    summary = json.loads(torch.ops._C_utils.inspect_cpu_attn_acc_locality_metadata(metadata))
    thread0_slots = list(range(0, summary["legacy_effective_thread_num"], 2))
    thread1_slots = list(range(1, summary["legacy_effective_thread_num"], 2))

    output = torch.empty_like(case["query"])
    capfd.readouterr()
    cpu_attention_with_kv_cache_acc_locality(
        query=case["query"],
        key_cache=case["packed_key_cache"],
        value_cache=case["packed_value_cache"],
        output=output,
        query_start_loc=case["query_start_loc"],
        seq_lens=case["seq_lens"],
        scale=case["scale"],
        causal=True,
        alibi_slopes=None,
        sliding_window=case["window_size"],
        block_table=case["block_tables"],
        softcap=0,
        scheduler_metadata=metadata,
        s_aux=None,
    )
    captured = capfd.readouterr()

    assert "group_span=2" in captured.out
    assert "covered_thread_num=2" in captured.out
    assert (
        f"thread 0: kv_head 0, covered_thread_offset=0, legacy_slots={thread0_slots}"
        in captured.out
    )
    assert (
        f"thread 1: kv_head 0, covered_thread_offset=1, legacy_slots={thread1_slots}"
        in captured.out
    )


def test_cpu_attention_balanced_runtime_log_exposes_scheduler_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    omp_cpuids = _select_runtime_cpu_ids()
    torch.ops._C_utils.init_cpu_threads_env(omp_cpuids)
    monkeypatch.setenv("VLLM_CPU_ATTN_DEBUG", "1")

    case = _prepare_small_cpu_attention_case()
    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        seq_lens=case["seq_lens"],
        dtype=torch.float32,
        query_start_loc=case["query_start_loc"],
        causal=True,
        sliding_window_size=-1,
        isa="vec",
        enable_kv_split=True,
    )

    output = torch.empty_like(case["query"])
    capfd.readouterr()
    cpu_attention_with_kv_cache(
        query=case["query"],
        key_cache=case["packed_key_cache"],
        value_cache=case["packed_value_cache"],
        output=output,
        query_start_loc=case["query_start_loc"],
        seq_lens=case["seq_lens"],
        scale=case["scale"],
        causal=True,
        alibi_slopes=None,
        sliding_window=case["window_size"],
        block_table=case["block_tables"],
        softcap=0,
        scheduler_metadata=metadata,
        s_aux=None,
    )
    captured = capfd.readouterr()

    assert "CPU attention balanced runtime summary" in captured.out
    assert "effective_thread_num=" in captured.out
    assert "attention_task_num=" in captured.out
    assert "reduction_task_num=" in captured.out
    assert "thread 0: workitem_group_range=" in captured.out

NUM_HEADS = [
    (4, 4),
    (8, 2),
    (9, 3),
]
HEAD_SIZES = [96, 128]
QTYPES = [torch.bfloat16, torch.half, torch.float32]
SLIDING_WINDOWS = [None, 256]
NUM_BLOCKS = [
    1024,
]
SEQ_LENS = [  # (q_len, kv_len)
    [(1, 213), (1, 1), (1, 312), (1, 7), (1, 7812)],  # decode batch
    [(2345, 2345), (5, 5), (3, 16), (134, 5131)],  # prefill batch
    [(992, 2456), (1, 1234), (98, 1145), (1, 4162), (2345, 2345)],  # mixed batch
]


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


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes.float()


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    dtype = query.dtype

    outputs: list[torch.Tensor] = []
    start_idx = 0

    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes[:, None, None]

    if s_aux is not None:
        s_aux = s_aux.float()
        s_aux = s_aux[:, None, None]

    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].float()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len].float()
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len].float()

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()

        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask

        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)

        if alibi_slopes is not None:
            q_start_pos = kv_len - query_len
            q_pos = q_start_pos + torch.arange(0, query_len)[None, :, None]
            kv_pos = torch.arange(0, kv_len)[None, None, :]
            dist = q_pos - kv_pos
            alibi_bias = -alibi_slopes * dist
            attn += alibi_bias

        attn.masked_fill_(mask, float("-inf"))

        if s_aux is not None:
            s_aux_ext = s_aux.repeat(1, query_len, 1)
            attn = torch.cat((s_aux_ext, attn), dim=-1)

        attn = torch.softmax(attn, dim=-1)

        if s_aux is not None:
            attn = attn[:, :, 1:]

        out = torch.einsum("hqk,khd->qhd", attn, v).to(dtype=dtype)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def varlen_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    set_random_seed(0)
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

    # for n heads the set of slopes is the geometric sequence that starts
    # 2^(-8/n)
    alibi_slopes = _get_alibi_slopes(num_query_heads) if use_alibi else None

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

    # KV cache for CPU attention
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

    # use reshape_and_cache to pack key_cache and value_cache
    slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
    )

    metadata = cpu_attn_get_scheduler_metadata(
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
        enable_kv_split=False,
    )

    out_without_split = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_without_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        sliding_window=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=metadata,
        s_aux=s_aux,
    )

    metadata = cpu_attn_get_scheduler_metadata(
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
        enable_kv_split=True,
    )

    out_with_split = torch.empty_like(query)
    cpu_attention_with_kv_cache(
        query=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        output=out_with_split,
        query_start_loc=cu_query_lens,
        seq_lens=kv_lens_tensor,
        scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        sliding_window=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=metadata,
        s_aux=s_aux,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        alibi_slopes=alibi_slopes,
        s_aux=s_aux,
    )

    atol, rtol = 1.5e-2, 1e-2
    (
        torch.testing.assert_close(out_with_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_with_split - ref_output))}",
    )
    (
        torch.testing.assert_close(out_without_split, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(out_without_split - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["vec"])
def test_varlen_with_paged_kv_normal_vec(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["amx"])
@pytest.mark.skipif(not supports_amx_tiles(), reason="no AMX support.")
def test_varlen_with_paged_kv_normal_amx(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [48])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["vec16"])
def test_varlen_with_paged_kv_normal_vec16(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [96, 128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", QTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", ["neon"])
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.ARM,
    reason="Not an Arm CPU.",
)
def test_varlen_with_paged_kv_normal_neon(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [50])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_softcap(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [True])
@pytest.mark.parametrize("use_sink", [False])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_alibi(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", [96])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("use_sink", [True])
@pytest.mark.parametrize("isa", [get_attn_isa()])
def test_varlen_with_paged_kv_sink(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_alibi: bool,
    use_sink: bool,
    isa: str,
) -> None:
    varlen_with_paged_kv(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        use_alibi=use_alibi,
        use_sink=use_sink,
        isa=isa,
    )
