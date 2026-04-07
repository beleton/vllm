# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

import vllm._C  # noqa: F401
from vllm.platforms import CpuArchEnum
from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo
from vllm.v1.worker.cpu_binding import (
    group_logical_cpus_by_l3,
    resolve_local_omp_cpuid,
)


def _build_fake_topology() -> list[LogicalCPUInfo]:
    cpus: list[LogicalCPUInfo] = []
    cpu_id = 0
    for numa_node in range(2):
        for l3_cache_id in range(2):
            for core_offset in range(2):
                core = l3_cache_id * 2 + core_offset
                physical_core = numa_node * 100 + core
                for sibling in range(2):
                    cpus.append(
                        LogicalCPUInfo(
                            id=cpu_id + sibling,
                            physical_core=physical_core,
                            numa_node=numa_node,
                            socket_id=numa_node,
                            l3_cache_id=numa_node * 10 + l3_cache_id,
                        )
                    )
                cpu_id += 2
    return cpus


def test_resolve_local_omp_cpuid_auto_x86_uses_one_thread_per_core():
    local_omp_cpuid = resolve_local_omp_cpuid(
        omp_cpuids="auto",
        rank=0,
        local_rank=0,
        world_size=2,
        cpu_arch=CpuArchEnum.X86,
        allowed_numa_nodes=[0, 1],
        logical_cpu_list=_build_fake_topology(),
        reserve_cpu_num=1,
    )

    assert local_omp_cpuid == "1,3,5"


def test_resolve_local_omp_cpuid_manual_binding_uses_rank_slice():
    local_omp_cpuid = resolve_local_omp_cpuid(
        omp_cpuids="1,3,5|9,11,13",
        rank=1,
        local_rank=1,
        world_size=2,
        cpu_arch=CpuArchEnum.X86,
        allowed_numa_nodes=[0, 1],
        logical_cpu_list=_build_fake_topology(),
    )

    assert local_omp_cpuid == "9,11,13"


def test_logical_cpu_info_keeps_socket_and_l3_ids():
    cpu = _build_fake_topology()[0]

    assert cpu.socket_id == 0
    assert cpu.l3_cache_id == 0


def test_group_logical_cpus_by_l3_splits_each_numa_node_into_l3_groups():
    groups = group_logical_cpus_by_l3(_build_fake_topology())

    assert list(groups) == [
        (0, 0, 0),
        (0, 0, 1),
        (1, 1, 10),
        (1, 1, 11),
    ]
    assert [cpu.id for cpu in groups[(0, 0, 0)]] == [0, 1, 2, 3]
    assert [cpu.id for cpu in groups[(0, 0, 1)]] == [4, 5, 6, 7]


def test_describe_cpu_locality_groups_matches_python_topology():
    allowed_numa_nodes, logical_cpu_list = CpuPlatform.get_allowed_cpu_core_node_list()
    assert allowed_numa_nodes
    grouped = group_logical_cpus_by_l3(logical_cpu_list)
    if not grouped:
        pytest.skip("Current environment does not expose L3 group topology.")

    expected_groups = []
    selected_cpu_ids = []
    for group_key, cpus in list(grouped.items())[: min(2, len(grouped))]:
        numa_node, socket_id, l3_cache_id = group_key
        cpu_ids = [cpu.id for cpu in cpus]
        selected_cpu_ids.extend(cpu_ids)
        expected_groups.append(
            {
                "numa_node": numa_node,
                "socket_id": socket_id,
                "l3_cache_id": l3_cache_id,
                "cpu_ids": cpu_ids,
                "num_cpus": len(cpu_ids),
            }
        )

    actual_groups = json.loads(
        torch.ops._C_utils.describe_cpu_locality_groups(
            ",".join(str(cpu_id) for cpu_id in selected_cpu_ids)
        )
    )

    assert actual_groups == expected_groups
