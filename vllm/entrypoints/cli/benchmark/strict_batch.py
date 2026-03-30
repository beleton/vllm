# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.strict_batch import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkStrictBatchSubcommand(BenchmarkSubcommandBase):
    """The `strict-batch` subcommand for `vllm bench`."""

    name = "strict-batch"
    help = "Benchmark strict offline batches with explicit enqueue-before-step."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
