from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "tools" / "microbench" / "read_shared_cache.c"
BIN = ROOT / "tools" / "microbench" / "read_shared_cache"


def _pick_test_cpus(count: int) -> list[int]:
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= count
    return cpus[:count]


def _compile_binary() -> None:
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-pthread",
            "-std=c11",
            str(SRC),
            "-o",
            str(BIN),
        ],
        check=True,
        cwd=ROOT,
    )


def test_read_shared_cache_runs_and_reports_summary() -> None:
    _compile_binary()
    cpus = _pick_test_cpus(2)

    result = subprocess.run(
        [
            str(BIN),
            "--threads",
            "2",
            "--cpus",
            ",".join(str(cpu) for cpu in cpus),
            "--workset-bytes",
            "4096",
            "--iters",
            "4",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["mode"] == "read-shared"
    assert summary["threads"] == 2
    assert summary["cpus"] == cpus
    assert summary["workset_bytes"] == 4096
    assert summary["iters"] == 4
    assert summary["elapsed_sec"] >= 0.0
    assert summary["total_touches"] > 0


def test_read_shared_cache_rejects_cpu_count_mismatch() -> None:
    _compile_binary()
    cpus = _pick_test_cpus(2)

    result = subprocess.run(
        [
            str(BIN),
            "--threads",
            "2",
            "--cpus",
            str(cpus[0]),
            "--workset-bytes",
            "4096",
            "--iters",
            "4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cpus count must match threads" in result.stderr
