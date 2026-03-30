# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess

import pytest

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.benchmark
def test_bench_strict_batch():
    command = [
        "vllm",
        "bench",
        "strict-batch",
        "--model",
        MODEL_NAME,
        "--input-len",
        "8",
        "--output-len",
        "1",
        "--batch-size",
        "2",
        "--num-rounds",
        "1",
        "--num-iters-warmup",
        "0",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "16",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
