# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a strict offline batch where requests are enqueued before step()."""

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

import vllm.envs as envs
from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType


@dataclass
class RoundResult:
    round_index: int
    latency: float
    waiting_before_first_step: int
    first_step_scheduled_requests: int
    first_step_scheduled_tokens: int
    finished_requests: int
    step_count: int


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={
            k: results[k]
            for k in [
                "avg_latency",
                "percentiles",
                "avg_first_step_scheduled_requests",
                "avg_first_step_scheduled_tokens",
            ]
        },
    )
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt. Currently only n=1.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of warmup rounds to run.",
    )
    parser.add_argument(
        "--num-rounds",
        "--num-iters",
        dest="num_rounds",
        type=int,
        default=30,
        help="Number of benchmark rounds to run.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the generation process of a single strict batch round.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the strict batch results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=0,
        help="Random seed used to generate dummy prompt token IDs.",
    )

    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)


def build_dummy_prompts(
    batch_size: int,
    input_len: int,
    prompt_seed: int,
) -> list[PromptType]:
    rng = np.random.default_rng(prompt_seed)
    dummy_prompt_token_ids = rng.integers(0, 10000, size=(batch_size, input_len))
    return [
        {"prompt_token_ids": batch}
        for batch in dummy_prompt_token_ids.tolist()
    ]


def get_waiting_request_count(engine: Any) -> int:
    return len(engine.engine_core.engine_core.scheduler.waiting)


def step_with_schedule_capture(engine: Any):
    scheduler = engine.engine_core.engine_core.scheduler
    original_schedule = scheduler.schedule
    captured: dict[str, Any] = {}

    def wrapped_schedule(*args, **kwargs):
        scheduler_output = original_schedule(*args, **kwargs)
        captured.setdefault("scheduler_output", scheduler_output)
        return scheduler_output

    scheduler.schedule = wrapped_schedule
    try:
        outputs = engine.step()
    finally:
        scheduler.schedule = original_schedule

    scheduler_output = captured.get("scheduler_output")
    if scheduler_output is None:
        raise RuntimeError(
            "Failed to capture scheduler output for the first strict-batch step."
        )
    return outputs, scheduler_output


def update_finished_request_ids(
    outputs: list[Any], finished_request_ids: set[str]
) -> None:
    for output in outputs:
        if getattr(output, "finished", False):
            finished_request_ids.add(output.request_id)


def run_strict_batch_round(
    engine: Any,
    prompts: list[PromptType],
    sampling_params: Any,
    round_index: int,
) -> RoundResult:
    request_ids = []
    start_time = time.perf_counter()
    for req_index, prompt in enumerate(prompts):
        request_id = f"round{round_index}-req{req_index}"
        engine.add_request(request_id, prompt, sampling_params)
        request_ids.append(request_id)

    waiting_before_first_step = get_waiting_request_count(engine)

    finished_request_ids: set[str] = set()
    outputs, scheduler_output = step_with_schedule_capture(engine)
    update_finished_request_ids(outputs, finished_request_ids)
    step_count = 1

    while engine.has_unfinished_requests():
        outputs = engine.step()
        update_finished_request_ids(outputs, finished_request_ids)
        step_count += 1

    latency = time.perf_counter() - start_time
    if len(finished_request_ids) != len(request_ids):
        raise RuntimeError(
            "Strict batch round finished with mismatched request counts: "
            f"expected {len(request_ids)}, got {len(finished_request_ids)}."
        )

    return RoundResult(
        round_index=round_index,
        latency=latency,
        waiting_before_first_step=waiting_before_first_step,
        first_step_scheduled_requests=len(scheduler_output.num_scheduled_tokens),
        first_step_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        finished_requests=len(finished_request_ids),
        step_count=step_count,
    )


def print_round_result(round_result: RoundResult) -> None:
    print(
        "Round("
        f"{round_result.round_index}"
        "): waiting before first step="
        f"{round_result.waiting_before_first_step}, "
        "first-step scheduled requests="
        f"{round_result.first_step_scheduled_requests}, "
        "first-step scheduled tokens="
        f"{round_result.first_step_scheduled_tokens}, "
        f"steps={round_result.step_count}, "
        f"latency={round_result.latency} seconds"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.input_len <= 0:
        raise ValueError("--input-len must be greater than 0.")
    if args.output_len <= 0:
        raise ValueError("--output-len must be greater than 0.")
    if args.num_iters_warmup < 0:
        raise ValueError("--num-iters-warmup must be non-negative.")
    if args.num_rounds <= 0:
        raise ValueError("--num-rounds must be greater than 0.")
    if args.n != 1:
        raise ValueError("`strict-batch` currently only supports `--n 1`.")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        raise ValueError(
            "`strict-batch` requires `VLLM_ENABLE_V1_MULTIPROCESSING=0`; "
            "otherwise the engine may process requests before a full round "
            "is enqueued."
        )
    engine_args = EngineArgs.from_cli_args(args)

    # Lazy import to avoid importing LLMEngine when the bench command is not
    # selected.
    from vllm import LLMEngine, SamplingParams

    engine = LLMEngine.from_engine_args(engine_args, enable_multiprocessing=False)
    assert engine.model_config.max_model_len >= (
        args.input_len + args.output_len
    ), (
        "Please ensure that max_model_len is greater than "
        "the sum of input_len and output_len."
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    dummy_prompts = build_dummy_prompts(
        batch_size=args.batch_size,
        input_len=args.input_len,
        prompt_seed=args.prompt_seed,
    )

    print("Warming up...")
    warmup_results = []
    for round_index in tqdm(range(args.num_iters_warmup), desc="Warmup rounds"):
        round_result = run_strict_batch_round(
            engine=engine,
            prompts=dummy_prompts,
            sampling_params=sampling_params,
            round_index=round_index,
        )
        warmup_results.append(round_result)
        print_round_result(round_result)

    if args.profile:
        print("Profiling a single strict batch round...")
        engine.start_profile()
        try:
            round_result = run_strict_batch_round(
                engine=engine,
                prompts=dummy_prompts,
                sampling_params=sampling_params,
                round_index=args.num_iters_warmup,
            )
        finally:
            engine.stop_profile()
        print_round_result(round_result)
        return

    round_results = []
    for round_index in tqdm(range(args.num_rounds), desc="Bench rounds"):
        round_result = run_strict_batch_round(
            engine=engine,
            prompts=dummy_prompts,
            sampling_params=sampling_params,
            round_index=args.num_iters_warmup + round_index,
        )
        round_results.append(round_result)
        print_round_result(round_result)

    latencies = np.array([round_result.latency for round_result in round_results])
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    avg_first_step_scheduled_requests = float(
        np.mean(
            [
                round_result.first_step_scheduled_requests
                for round_result in round_results
            ]
        )
    )
    avg_first_step_scheduled_tokens = float(
        np.mean(
            [
                round_result.first_step_scheduled_tokens
                for round_result in round_results
            ]
        )
    )

    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")
    print(
        "Avg first-step scheduled requests: "
        f"{avg_first_step_scheduled_requests}"
    )
    print(
        "Avg first-step scheduled tokens: "
        f"{avg_first_step_scheduled_tokens}"
    )

    if args.output_json:
        results = {
            "avg_latency": float(np.mean(latencies)),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
            "avg_first_step_scheduled_requests":
            avg_first_step_scheduled_requests,
            "avg_first_step_scheduled_tokens":
            avg_first_step_scheduled_tokens,
            "warmup_rounds": [
                dataclasses.asdict(round_result)
                for round_result in warmup_results
            ],
            "rounds": [
                dataclasses.asdict(round_result)
                for round_result in round_results
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)
