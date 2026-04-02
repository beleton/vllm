# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import CpuArchEnum
from vllm.platforms.cpu import LogicalCPUInfo
from vllm.v1.worker.cpu_binding import resolve_local_omp_cpuid


def _build_fake_topology() -> list[LogicalCPUInfo]:
    cpus: list[LogicalCPUInfo] = []
    cpu_id = 0
    for numa_node in range(2):
        for core in range(4):
            physical_core = numa_node * 100 + core
            cpus.append(
                LogicalCPUInfo(
                    id=cpu_id,
                    physical_core=physical_core,
                    numa_node=numa_node,
                )
            )
            cpus.append(
                LogicalCPUInfo(
                    id=cpu_id + 1,
                    physical_core=physical_core,
                    numa_node=numa_node,
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
