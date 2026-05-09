from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path


BALANCED_TRACE_0 = (
    "rank=0 CPU attention trace mode=balanced thread_id=0 task_idx=0 "
    "thread_offset=0 kv_head_idx=0 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=64 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

BALANCED_TRACE_1 = (
    "rank=0 CPU attention trace mode=balanced thread_id=2 task_idx=1 "
    "thread_offset=1 kv_head_idx=0 workitem_group_idx=1 req_id=0 "
    "q_token_id_start=64 q_token_num=32 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

BALANCED_TRACE_2 = (
    "rank=0 CPU attention trace mode=balanced thread_id=3 task_idx=2 "
    "thread_offset=0 kv_head_idx=1 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=64 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

ACC_TRACE_0 = (
    "rank=0 CPU attention trace mode=acc-local-l3 thread_id=0 subgroup_id=0 "
    "legacy_thread_offset=0 kv_head_idx=0 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=64 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

ACC_TRACE_1 = (
    "rank=0 CPU attention trace mode=acc-local-l3 thread_id=2 subgroup_id=1 "
    "legacy_thread_offset=0 kv_head_idx=1 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=64 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)


def _rank_results_json() -> str:
    return (
        '{ "rank_results": ['
        "{"
        '"rank": 0,'
        '"omp_cpuids": "256,257,272,273",'
        '"locality_groups": ['
        '{"numa_node":0,"socket_id":0,"l3_cache_id":0,"cpu_ids":[256,257]},'
        '{"numa_node":0,"socket_id":0,"l3_cache_id":4,"cpu_ids":[272,273]}'
        "]"
        "}"
        "] }"
    )


class TestAttnMappingCompareToHtml(unittest.TestCase):

    def test_build_mode_summary_counts_ccd_kv_matrix(self):
        module = importlib.import_module("tools.p3_attn_only.attn_mapping_compare_to_html")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "balanced.log"
            log_path.write_text(
                "\n".join(
                    [
                        BALANCED_TRACE_0,
                        BALANCED_TRACE_1,
                        BALANCED_TRACE_2,
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            summary = module.build_mode_summary(log_path, rank=0)

        self.assertEqual("balanced", summary.mode)
        self.assertEqual([0, 1], summary.kv_heads)
        self.assertEqual([0, 4], [group.l3_cache_id for group in summary.l3_groups])
        self.assertEqual([[1, 0], [1, 1]], summary.matrix)
        self.assertEqual({0: 2, 1: 1}, summary.kv_span_counts)
        self.assertEqual({0: 1, 4: 2}, summary.l3_kv_counts)

    def test_render_html_contains_matrices_and_summaries(self):
        module = importlib.import_module("tools.p3_attn_only.attn_mapping_compare_to_html")

        with tempfile.TemporaryDirectory() as tmpdir:
            balanced_log = Path(tmpdir) / "balanced.log"
            acc_log = Path(tmpdir) / "acc.log"
            balanced_log.write_text(
                "\n".join(
                    [
                        BALANCED_TRACE_0,
                        BALANCED_TRACE_1,
                        BALANCED_TRACE_2,
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )
            acc_log.write_text(
                "\n".join(
                    [
                        ACC_TRACE_0,
                        ACC_TRACE_1,
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            balanced = module.build_mode_summary(balanced_log, rank=0)
            acc = module.build_mode_summary(acc_log, rank=0)
            html = module.render_html(balanced, acc)

        self.assertIn("CCD × KV Head", html)
        self.assertIn("balanced", html)
        self.assertIn("acc-local-l3", html)
        self.assertIn("kv0", html)
        self.assertIn("l3_0", html)
        self.assertIn("KV Head 跨 CCD 数", html)
        self.assertIn("CCD 同时覆盖的 KV Head 数", html)

    def test_main_writes_output_file(self):
        module = importlib.import_module("tools.p3_attn_only.attn_mapping_compare_to_html")

        with tempfile.TemporaryDirectory() as tmpdir:
            balanced_log = Path(tmpdir) / "balanced.log"
            acc_log = Path(tmpdir) / "acc.log"
            output = Path(tmpdir) / "mapping.html"
            balanced_log.write_text(
                "\n".join(
                    [
                        BALANCED_TRACE_0,
                        BALANCED_TRACE_1,
                        BALANCED_TRACE_2,
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )
            acc_log.write_text(
                "\n".join(
                    [
                        ACC_TRACE_0,
                        ACC_TRACE_1,
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            output_path = module.main(
                [
                    "--balanced-log",
                    str(balanced_log),
                    "--acc-log",
                    str(acc_log),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(output, output_path)
            self.assertTrue(output.exists())
            html = output.read_text(encoding="utf-8")
            self.assertIn("balanced", html)
            self.assertIn("acc-local-l3", html)


if __name__ == "__main__":
    unittest.main()
