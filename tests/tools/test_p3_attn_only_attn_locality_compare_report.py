from __future__ import annotations

import html as html_lib
import importlib
import re
import tempfile
import unittest
from pathlib import Path


def _trace_block(
    trace_line: str,
    q_tile_num: int,
    default_q_tile_token_num: int = 32,
) -> str:
    lines = [
        trace_line,
        (
            "  tile_plan: "
            f"q_tile_num={q_tile_num}, "
            f"default_q_tile_token_num={default_q_tile_token_num}"
        ),
    ]
    q_start = int(trace_line.split("q_token_id_start=")[1].split()[0])
    q_token_num = int(trace_line.split("q_token_num=")[1].split()[0])
    step = max(1, q_token_num // q_tile_num)
    for idx in range(q_tile_num):
        tile_q_start = q_start + idx * step
        tile_q_end = q_start + q_token_num if idx == q_tile_num - 1 else tile_q_start + step
        kv_end = ((tile_q_end + 31) // 32) * 32
        lines.append(
            "    "
            f"q_tile {idx}: q_range=[{tile_q_start},{tile_q_end}), "
            f"kv_range=[0,{kv_end}), kv_tile_size=64, kv_tile_num={idx + 1}"
        )
    return "\n".join(lines)


BALANCED_TRACE_A = (
    "rank=0 CPU attention trace mode=balanced thread_id=0 task_idx=0 "
    "thread_offset=0 kv_head_idx=0 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=64 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

BALANCED_TRACE_B = (
    "rank=0 CPU attention trace mode=balanced thread_id=0 task_idx=1 "
    "thread_offset=0 kv_head_idx=0 workitem_group_idx=1 req_id=0 "
    "q_token_id_start=64 q_token_num=32 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

BALANCED_TRACE_C = (
    "rank=0 CPU attention trace mode=balanced thread_id=2 task_idx=2 "
    "thread_offset=0 kv_head_idx=1 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=96 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

BALANCED_TRACE_D = (
    "rank=0 CPU attention trace mode=balanced thread_id=3 task_idx=3 "
    "thread_offset=1 kv_head_idx=0 workitem_group_idx=2 req_id=0 "
    "q_token_id_start=96 q_token_num=128 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

ACC_TRACE_A = (
    "rank=0 CPU attention trace mode=acc-local-l3 thread_id=0 subgroup_id=0 "
    "legacy_thread_offset=0 kv_head_idx=0 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=64 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

ACC_TRACE_B = (
    "rank=0 CPU attention trace mode=acc-local-l3 thread_id=2 subgroup_id=1 "
    "legacy_thread_offset=0 kv_head_idx=1 workitem_group_idx=0 req_id=0 "
    "q_token_id_start=0 q_token_num=96 kv_split_pos_start=0 "
    "kv_split_pos_end=256 split_id=-1 local_split_id=0"
)

ACC_TRACE_C = (
    "rank=0 CPU attention trace mode=acc-local-l3 thread_id=2 subgroup_id=1 "
    "legacy_thread_offset=0 kv_head_idx=1 workitem_group_idx=1 req_id=0 "
    "q_token_id_start=96 q_token_num=32 kv_split_pos_start=0 "
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
        "},"
        "{"
        '"rank": 1,'
        '"omp_cpuids": "384,385,400,401",'
        '"locality_groups": ['
        '{"numa_node":1,"socket_id":1,"l3_cache_id":8,"cpu_ids":[384,385]},'
        '{"numa_node":1,"socket_id":1,"l3_cache_id":12,"cpu_ids":[400,401]}'
        "]"
        "}"
        "] }"
    )


def _extract_runtime_rows(html_text: str, ccd: int) -> list[list[str]]:
    marker = f'<div class="ccd-title">CCD {ccd}</div>'
    section = html_text.split(marker, 1)[1].split("</section>", 1)[0]
    tbody = section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]

    rows: list[list[str]] = []
    for row_html in re.findall(r"<tr>(.*?)</tr>", tbody, flags=re.S):
        cells = []
        for cell_html in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.S):
            text = re.sub(r"<[^>]+>", "", cell_html)
            cells.append(html_lib.unescape(text).strip())
        rows.append(cells)
    return rows


class TestAttnLocalityCompareReport(unittest.TestCase):

    def test_build_core_kv_rows_aggregates_q_tiles(self):
        module = importlib.import_module(
            "tools.p3_attn_only.attn_locality_compare_report"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "balanced.log"
            log_path.write_text(
                "\n".join(
                    [
                        _trace_block(BALANCED_TRACE_A, q_tile_num=2),
                        _trace_block(BALANCED_TRACE_B, q_tile_num=1),
                        _trace_block(BALANCED_TRACE_C, q_tile_num=3),
                        _trace_block(BALANCED_TRACE_D, q_tile_num=4),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            rows = module.build_core_kv_rows(log_path, rank=0)

        rows_by_key = {
            (row.ccd, row.core, row.kv_head_idx): row
            for row in rows
        }
        self.assertEqual(3, rows_by_key[(0, 256, 0)].total_q_tiles)
        self.assertEqual(2, rows_by_key[(0, 256, 0)].task_rows)
        self.assertEqual(96, rows_by_key[(0, 256, 0)].total_q_tokens)
        self.assertEqual(3, rows_by_key[(4, 272, 1)].total_q_tiles)
        self.assertEqual(4, rows_by_key[(4, 273, 0)].total_q_tiles)

    def test_build_core_task_rows_keeps_multiple_tasks_on_same_core_separate(self):
        module = importlib.import_module(
            "tools.p3_attn_only.attn_locality_compare_report"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "balanced.log"
            log_path.write_text(
                "\n".join(
                    [
                        _trace_block(BALANCED_TRACE_A, q_tile_num=2),
                        _trace_block(BALANCED_TRACE_B, q_tile_num=1),
                        _trace_block(BALANCED_TRACE_C, q_tile_num=3),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            rows = module.build_core_task_rows(log_path, rank=0)

        core_256_rows = [row for row in rows if row.ccd == 0 and row.core == 256]
        self.assertEqual(2, len(core_256_rows))
        self.assertEqual([0, 1], [row.workitem_group_idx for row in core_256_rows])
        self.assertEqual([2, 1], [row.q_tile_num for row in core_256_rows])

    def test_render_html_contains_mapping_and_runtime_sections(self):
        module = importlib.import_module(
            "tools.p3_attn_only.attn_locality_compare_report"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            balanced_log = Path(tmpdir) / "balanced.log"
            acc_log = Path(tmpdir) / "acc.log"
            output = Path(tmpdir) / "compare.html"
            balanced_log.write_text(
                "\n".join(
                    [
                        _trace_block(BALANCED_TRACE_A, q_tile_num=2),
                        _trace_block(BALANCED_TRACE_B, q_tile_num=1),
                        _trace_block(BALANCED_TRACE_C, q_tile_num=3),
                        _trace_block(BALANCED_TRACE_D, q_tile_num=4),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )
            acc_log.write_text(
                "\n".join(
                    [
                        _trace_block(ACC_TRACE_A, q_tile_num=2),
                        _trace_block(ACC_TRACE_B, q_tile_num=3),
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

        self.assertIn("CCD × KV Head", html)
        self.assertIn("CCD 内核心执行明细", html)
        self.assertIn("total q_tile", html)
        self.assertIn("仅展示同时存在 trace 的 rank", html)
        self.assertIn("Task Nums", html)
        self.assertIn("Workitem", html)
        self.assertIn("balanced", html)
        self.assertIn("acc-local-l3", html)

    def test_main_accepts_case_dir_and_auto_discovers_logs(self):
        module = importlib.import_module(
            "tools.p3_attn_only.attn_locality_compare_report"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "case"
            balanced_dir = case_dir / "balanced"
            acc_dir = case_dir / "acc-local-l3"
            balanced_dir.mkdir(parents=True)
            acc_dir.mkdir(parents=True)

            balanced_log = balanced_dir / "debug.log"
            acc_log = acc_dir / "debug.log"
            balanced_log.write_text(
                "\n".join(
                    [
                        _trace_block(BALANCED_TRACE_A, q_tile_num=2),
                        _trace_block(BALANCED_TRACE_B, q_tile_num=1),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )
            acc_log.write_text(
                "\n".join(
                    [
                        _trace_block(ACC_TRACE_A, q_tile_num=2),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            output_path = module.main([str(case_dir)])

            self.assertEqual(case_dir / "attn_locality_compare.html", output_path)
            self.assertTrue(output_path.exists())
            html = output_path.read_text(encoding="utf-8")

        self.assertIn(str(balanced_log), html)
        self.assertIn(str(acc_log), html)
        self.assertIn("balanced vs acc-local-l3 对比报告", html)

    def test_render_html_aligns_runtime_rows_by_core_across_modes(self):
        module = importlib.import_module(
            "tools.p3_attn_only.attn_locality_compare_report"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            balanced_log = Path(tmpdir) / "balanced.log"
            acc_log = Path(tmpdir) / "acc.log"
            output = Path(tmpdir) / "compare.html"
            balanced_log.write_text(
                "\n".join(
                    [
                        _trace_block(BALANCED_TRACE_A, q_tile_num=2),
                        _trace_block(BALANCED_TRACE_B, q_tile_num=1),
                        _trace_block(BALANCED_TRACE_C, q_tile_num=3),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )
            acc_log.write_text(
                "\n".join(
                    [
                        _trace_block(ACC_TRACE_A, q_tile_num=2),
                        _trace_block(ACC_TRACE_B, q_tile_num=3),
                        _trace_block(ACC_TRACE_C, q_tile_num=1),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            module.main(
                [
                    "--balanced-log",
                    str(balanced_log),
                    "--acc-log",
                    str(acc_log),
                    "--output",
                    str(output),
                ]
            )
            html = output.read_text(encoding="utf-8")

        self.assertIn("balanced Task Nums", html)
        self.assertIn("acc-local-l3 Task Nums", html)
        self.assertIn("balanced KV Head", html)
        self.assertIn("acc-local-l3 KV Head", html)

        ccd0_rows = _extract_runtime_rows(html, ccd=0)
        self.assertEqual(
            [
                [
                    "256",
                    "2",
                    "kv0",
                    "2",
                    "64",
                    "0",
                    "[0,64)",
                    "1",
                    "kv0",
                    "2",
                    "64",
                    "0",
                    "[0,64)",
                ],
                [
                    "",
                    "",
                    "kv0",
                    "1",
                    "32",
                    "1",
                    "[64,96)",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            ],
            ccd0_rows,
        )

        ccd4_rows = _extract_runtime_rows(html, ccd=4)
        self.assertEqual(
            [
                [
                    "272",
                    "1",
                    "kv1",
                    "3",
                    "96",
                    "0",
                    "[0,96)",
                    "2",
                    "kv1",
                    "3",
                    "96",
                    "0",
                    "[0,96)",
                ],
                [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "kv1",
                    "1",
                    "32",
                    "1",
                    "[96,128)",
                ],
            ],
            ccd4_rows,
        )

    def test_render_html_makes_core_column_sticky(self):
        module = importlib.import_module(
            "tools.p3_attn_only.attn_locality_compare_report"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            balanced_log = Path(tmpdir) / "balanced.log"
            acc_log = Path(tmpdir) / "acc.log"
            output = Path(tmpdir) / "compare.html"
            balanced_log.write_text(
                "\n".join(
                    [
                        _trace_block(BALANCED_TRACE_A, q_tile_num=2),
                        _trace_block(BALANCED_TRACE_B, q_tile_num=1),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )
            acc_log.write_text(
                "\n".join(
                    [
                        _trace_block(ACC_TRACE_A, q_tile_num=2),
                        _rank_results_json(),
                    ]
                ),
                encoding="utf-8",
            )

            module.main(
                [
                    "--balanced-log",
                    str(balanced_log),
                    "--acc-log",
                    str(acc_log),
                    "--output",
                    str(output),
                ]
            )
            html = output.read_text(encoding="utf-8")

        self.assertIn('<th class="shared-head core-head">Core</th>', html)
        self.assertIn('class="core-cell"', html)
        self.assertIn(".runtime-compare-table .core-head,", html)
        self.assertIn("position: sticky;", html)
        self.assertIn("left: 0;", html)


if __name__ == "__main__":
    unittest.main()
