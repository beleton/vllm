# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

from vllm import envs
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo


def _get_cpu_selector(
    cpu_arch: CpuArchEnum,
) -> Callable[[list[LogicalCPUInfo]], list[LogicalCPUInfo]] | None:
    if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
        return lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4]
    if cpu_arch == CpuArchEnum.X86:
        return lambda cpus: cpus[-1:]
    if cpu_arch == CpuArchEnum.ARM:
        return lambda cpus: cpus
    return None


def group_logical_cpus_by_l3(
    logical_cpu_list: list[LogicalCPUInfo],
) -> dict[tuple[int, int, int], list[LogicalCPUInfo]]:
    groups: dict[tuple[int, int, int], list[LogicalCPUInfo]] = {}
    sorted_cpu_list = sorted(
        logical_cpu_list,
        key=lambda cpu: (
            cpu.numa_node,
            cpu.socket_id,
            cpu.l3_cache_id,
            cpu.id,
        ),
    )
    for cpu_info in sorted_cpu_list:
        group_key = (cpu_info.numa_node, cpu_info.socket_id, cpu_info.l3_cache_id)
        groups.setdefault(group_key, []).append(cpu_info)
    return groups


def get_auto_local_omp_cpuid(
    local_rank: int,
    world_size: int,
    data_parallel_size_local: int = 1,
    reserve_cpu_num: int | None = None,
    cpu_arch: CpuArchEnum | None = None,
    allowed_numa_nodes: list[int] | None = None,
    logical_cpu_list: list[LogicalCPUInfo] | None = None,
) -> str:
    if cpu_arch is None:
        cpu_arch = current_platform.get_cpu_architecture()
    cpu_selector = _get_cpu_selector(cpu_arch)
    if cpu_selector is None:
        return "nobind"

    if allowed_numa_nodes is None or logical_cpu_list is None:
        allowed_numa_nodes, logical_cpu_list = (
            CpuPlatform.get_allowed_cpu_core_node_list()
        )

    assert len(allowed_numa_nodes) >= world_size, (
        f"No enough allowed NUMA nodes to bind threads of "
        f"{world_size} CPUWorkers. "
        f"Allowed NUMA nodes are {allowed_numa_nodes}. "
        "Please try to bind threads manually."
    )

    selected_numa_node = allowed_numa_nodes[local_rank]
    selected_logical_cpus = [
        x for x in logical_cpu_list if x.numa_node == selected_numa_node
    ]

    core_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
    for cpu_info in selected_logical_cpus:
        core_to_cpus.setdefault(cpu_info.physical_core, []).append(cpu_info)

    selected_logical_cpus = []
    for cpu_list in core_to_cpus.values():
        cpu_list = sorted(cpu_list, key=lambda x: x.id)
        selected_logical_cpus.extend(cpu_selector(cpu_list))
    selected_logical_cpus = sorted(selected_logical_cpus, key=lambda x: x.id)

    if reserve_cpu_num is None:
        need_reserve = world_size > 1 or data_parallel_size_local > 1
        reserve_cpu_num = 1 if need_reserve else 0

    assert len(selected_logical_cpus) > reserve_cpu_num, (
        f"VLLM_CPU_NUM_OF_RESERVED_CPU ({reserve_cpu_num}) "
        f"should less than {len(selected_logical_cpus)}."
    )
    if reserve_cpu_num != 0:
        selected_logical_cpus = selected_logical_cpus[:-reserve_cpu_num]

    return ",".join(str(x.id) for x in selected_logical_cpus)


def resolve_local_omp_cpuid(
    omp_cpuids: str,
    rank: int,
    local_rank: int,
    world_size: int,
    data_parallel_rank_local: int | None = None,
    data_parallel_size_local: int = 1,
    reserve_cpu_num: int | None = None,
    cpu_arch: CpuArchEnum | None = None,
    allowed_numa_nodes: list[int] | None = None,
    logical_cpu_list: list[LogicalCPUInfo] | None = None,
) -> str:
    if omp_cpuids == "auto":
        return get_auto_local_omp_cpuid(
            local_rank=local_rank,
            world_size=world_size,
            data_parallel_size_local=data_parallel_size_local,
            reserve_cpu_num=reserve_cpu_num,
            cpu_arch=cpu_arch,
            allowed_numa_nodes=allowed_numa_nodes,
            logical_cpu_list=logical_cpu_list,
        )
    if omp_cpuids == "nobind":
        return "nobind"

    omp_cpuids_list = omp_cpuids.split("|")
    if data_parallel_rank_local is not None:
        omp_cpuids_list = omp_cpuids_list[
            data_parallel_rank_local * world_size : (data_parallel_rank_local + 1)
            * world_size
        ]

    assert rank < len(omp_cpuids_list), (
        f"Rank {rank} exceeds configured CPU binding list. "
        f"Need at least {rank + 1} entries, got {len(omp_cpuids_list)}."
    )
    return omp_cpuids_list[rank]


def get_default_reserve_cpu_num() -> int | None:
    return envs.VLLM_CPU_NUM_OF_RESERVED_CPU
