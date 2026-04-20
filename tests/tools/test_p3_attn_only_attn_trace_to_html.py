import importlib
import tempfile
import unittest
from pathlib import Path


BALANCED_TRACE = (
    "rank=0 CPU attention trace mode=balanced thread_id=10 task_idx=20 "
    "thread_offset=5 kv_head_idx=1 workitem_group_idx=5 req_id=0 "
    "q_token_id_start=580 q_token_num=56 kv_split_pos_start=0 "
    "kv_split_pos_end=1024 split_id=-1 local_split_id=0"
)

ACC_TRACE = (
    "rank=0 CPU attention trace mode=acc-local-l3 thread_id=14 subgroup_id=0 "
    "legacy_thread_offset=1 kv_head_idx=0 workitem_group_idx=1 req_id=0 "
    "q_token_id_start=248 q_token_num=112 kv_split_pos_start=0 "
    "kv_split_pos_end=1024 split_id=-1 local_split_id=0"
)


def _rank_results_json(
    rank_to_cpu_ids: dict[int, list[int]],
    rank_to_locality_groups: dict[int, list[dict[str, object]]] | None = None,
) -> str:
    rank_entries = []
    for rank, cpu_ids in rank_to_cpu_ids.items():
        cpu_ids_str = ",".join(str(cpu_id) for cpu_id in cpu_ids)
        locality_groups = (rank_to_locality_groups or {}).get(rank, [])
        locality_groups_json = ",".join(
            (
                "{"
                f'"numa_node": {group.get("numa_node", 0)},'
                f'"socket_id": {group.get("socket_id", 0)},'
                f'"l3_cache_id": {group["l3_cache_id"]},'
                '"cpu_ids": [' + ",".join(str(cpu_id) for cpu_id in group["cpu_ids"]) + "]"
                "}"
            )
            for group in locality_groups
        )
        rank_entries.append(
            "{"
            f'"rank": {rank},'
            f'"omp_cpuids": "{cpu_ids_str}",'
            f'"locality_groups": [{locality_groups_json}]'
            "}"
        )
    return '{ "rank_results": [' + ",".join(rank_entries) + "] }"


class TestAttnTraceToHtml(unittest.TestCase):

    def test_parse_trace_line_handles_balanced_trace(self):
        module = importlib.import_module("tools.p3_attn_only.attn_trace_to_html")

        record = module.parse_trace_line(BALANCED_TRACE)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(0, record["rank"])
        self.assertEqual("balanced", record["mode"])
        self.assertEqual(10, record["thread_id"])
        self.assertEqual(20, record["task_idx"])
        self.assertEqual(5, record["thread_offset"])
        self.assertEqual(1, record["kv_head_idx"])
        self.assertEqual(0, record["req_id"])
        self.assertEqual(580, record["q_token_id_start"])
        self.assertNotIn("subgroup_id", record)

    def test_parse_trace_line_handles_acc_locality_trace(self):
        module = importlib.import_module("tools.p3_attn_only.attn_trace_to_html")

        record = module.parse_trace_line(ACC_TRACE)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual("acc-local-l3", record["mode"])
        self.assertEqual(14, record["thread_id"])
        self.assertEqual(0, record["subgroup_id"])
        self.assertEqual(1, record["legacy_thread_offset"])
        self.assertNotIn("task_idx", record)

    def test_load_trace_records_ignores_non_trace_lines(self):
        module = importlib.import_module("tools.p3_attn_only.attn_trace_to_html")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trace.log"
            log_path.write_text(
                "\n".join(
                    [
                        "INFO something else",
                        BALANCED_TRACE,
                        "CPU attention balanced runtime summary",
                        ACC_TRACE,
                    ]
                ),
                encoding="utf-8",
            )

            records = module.load_trace_records(log_path)

        self.assertEqual(2, len(records))
        self.assertEqual([0, 1], [record["line_no"] for record in records])

    def test_load_trace_records_attaches_core_from_rank_results(self):
        module = importlib.import_module("tools.p3_attn_only.attn_trace_to_html")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trace.log"
            log_path.write_text(
                "\n".join(
                    [
                        BALANCED_TRACE,
                        ACC_TRACE,
                        _rank_results_json(
                            {
                                0: list(range(256, 280)),
                            },
                            {
                                0: [
                                    {
                                        "l3_cache_id": 0,
                                        "cpu_ids": list(range(256, 272)),
                                    },
                                    {
                                        "l3_cache_id": 4,
                                        "cpu_ids": list(range(272, 280)),
                                    },
                                ]
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            records = module.load_trace_records(log_path)

        self.assertEqual(266, records[0]["core"])
        self.assertEqual(270, records[1]["core"])
        self.assertEqual(0, records[0]["ccd"])
        self.assertEqual(0, records[1]["ccd"])

    def test_build_request_kv_coverage_groups_by_req_and_kv_head(self):
        module = importlib.import_module("tools.p3_attn_only.attn_trace_to_html")
        records = [
            module.parse_trace_line(BALANCED_TRACE),
            module.parse_trace_line(
                BALANCED_TRACE.replace("thread_id=10", "thread_id=11")
            ),
            module.parse_trace_line(
                BALANCED_TRACE.replace("kv_head_idx=1", "kv_head_idx=2")
            ),
        ]
        assert records[0] is not None
        assert records[1] is not None
        assert records[2] is not None
        records[0]["core"] = 266
        records[1]["core"] = 276
        records[2]["core"] = 266
        records[0]["ccd"] = 0
        records[1]["ccd"] = 4
        records[2]["ccd"] = 0

        rows = module.build_request_kv_coverage(records)

        self.assertEqual(2, len(rows))
        self.assertEqual(0, rows[0]["req_id"])
        self.assertEqual(1, rows[0]["kv_head_idx"])
        self.assertEqual([266, 276], rows[0]["cores"])
        self.assertEqual([0, 4], rows[0]["ccds"])
        self.assertEqual([10, 11], rows[0]["threads"])
        self.assertEqual(2, rows[0]["task_rows"])

    def test_render_html_groups_records_by_thread(self):
        module = importlib.import_module("tools.p3_attn_only.attn_trace_to_html")
        records = [
            module.parse_trace_line(BALANCED_TRACE),
            module.parse_trace_line(
                BALANCED_TRACE.replace("thread_id=10", "thread_id=10").replace(
                    "req_id=0", "req_id=1"
                )
            ),
            module.parse_trace_line(ACC_TRACE.replace("thread_id=14", "thread_id=7")),
        ]
        for idx, record in enumerate(records):
            assert record is not None
            record["line_no"] = idx
        records[0]["core"] = 266
        records[0]["ccd"] = 0
        records[1]["core"] = 266
        records[1]["ccd"] = 0
        records[2]["core"] = 263
        records[2]["ccd"] = 0

        html = module.render_html(
            records=[record for record in records if record is not None],
            source_log=Path("sample.log"),
        )

        self.assertIn("Attention Trace Viewer", html)
        self.assertIn("sample.log", html)
        self.assertIn("Thread 10", html)
        self.assertIn("Thread 7", html)
        self.assertIn("Total Trace Rows", html)
        self.assertIn("Request × KV Head Coverage", html)
        self.assertIn("core=266", html)
        self.assertIn("CCD", html)
        self.assertIn("CCD=0", html)
        self.assertIn("task_idx", html)
        self.assertIn("legacy_thread_offset", html)
        self.assertIn("req_id=1", html)

    def test_main_writes_default_html_next_to_log(self):
        module = importlib.import_module("tools.p3_attn_only.attn_trace_to_html")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trace.log"
            log_path.write_text(BALANCED_TRACE + "\n", encoding="utf-8")

            output_path = module.main(["--log", str(log_path)])

            self.assertEqual(log_path.with_suffix(".html"), output_path)
            self.assertTrue(output_path.exists())
            html = output_path.read_text(encoding="utf-8")
            self.assertIn("Thread 10", html)


if __name__ == "__main__":
    unittest.main()
