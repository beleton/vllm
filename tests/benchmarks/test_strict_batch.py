# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.benchmarks.strict_batch import (
    add_cli_args,
    run_strict_batch_round,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


class FakeRequestOutput:

    def __init__(self, request_id: str, finished: bool = True):
        self.request_id = request_id
        self.finished = finished


class FakeSchedulerOutput:

    def __init__(self, num_scheduled_tokens: dict[str, int]):
        self.num_scheduled_tokens = num_scheduled_tokens
        self.total_num_scheduled_tokens = sum(num_scheduled_tokens.values())


class FakeScheduler:

    def __init__(self, num_scheduled_tokens: dict[str, int]):
        self.waiting: list[str] = []
        self._num_scheduled_tokens = num_scheduled_tokens
        self.schedule_waiting_sizes: list[int] = []

    def schedule(self) -> FakeSchedulerOutput:
        self.schedule_waiting_sizes.append(len(self.waiting))
        return FakeSchedulerOutput(self._num_scheduled_tokens)


class FakeEngine:

    def __init__(self, num_scheduled_tokens: dict[str, int]):
        self.scheduler = FakeScheduler(num_scheduled_tokens)
        self.engine_core = SimpleNamespace(
            engine_core=SimpleNamespace(scheduler=self.scheduler))
        self.added_request_ids: list[str] = []
        self.pending_request_ids: list[str] = []

    def add_request(self, request_id: str, prompt, sampling_params) -> None:
        del prompt, sampling_params
        self.added_request_ids.append(request_id)
        self.pending_request_ids.append(request_id)
        self.scheduler.waiting.append(request_id)

    def step(self) -> list[FakeRequestOutput]:
        self.scheduler.schedule()
        outputs = [
            FakeRequestOutput(request_id, finished=True)
            for request_id in self.pending_request_ids
        ]
        self.pending_request_ids = []
        self.scheduler.waiting.clear()
        return outputs

    def has_unfinished_requests(self) -> bool:
        return bool(self.pending_request_ids)


def test_run_strict_batch_round_enqueues_full_batch_before_first_step():
    engine = FakeEngine({
        "round0-req0": 16,
        "round0-req1": 16,
        "round0-req2": 16,
    })
    prompts = [
        {"prompt_token_ids": [1, 2]},
        {"prompt_token_ids": [3, 4]},
        {"prompt_token_ids": [5, 6]},
    ]

    result = run_strict_batch_round(
        engine=engine,
        prompts=prompts,
        sampling_params=object(),
        round_index=0,
    )

    assert engine.added_request_ids == [
        "round0-req0",
        "round0-req1",
        "round0-req2",
    ]
    assert engine.scheduler.schedule_waiting_sizes == [3]
    assert result.waiting_before_first_step == 3
    assert result.first_step_scheduled_requests == 3
    assert result.first_step_scheduled_tokens == 48
    assert result.finished_requests == 3
    assert result.step_count == 1


def test_add_cli_args_uses_prompt_seed_without_conflicting_with_engine_seed():
    parser = FlexibleArgumentParser()

    add_cli_args(parser)

    args = parser.parse_args([
        "--model",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--prompt-seed",
        "7",
    ])

    assert args.prompt_seed == 7
    assert args.seed == 0
