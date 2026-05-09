import csv
import importlib
import json
import tempfile
import textwrap
import unittest
from pathlib import Path


def _write_profile(case_dir: Path,
                   slowest_rank_mean_ms: float = 100.0,
                   call_count: int = 100,
                   task_count: int = 12800,
                   task_body_avg_ns: float = 1000.0) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
        "profile_summary": {
            "runtime": {
                "call_count": call_count,
                "attention_task_count": task_count,
                "attention_task_body_avg_ns": task_body_avg_ns,
            },
        },
    }
    (case_dir / "profile.json").write_text(json.dumps(profile),
                                           encoding="utf-8")


def _write_custom_pcm(case_dir: Path,
                      l3_miss: float = 2000.0,
                      l3_access: float = 8000.0,
                      l3_miss_per_second: float = 100.0) -> None:
    report_dir = case_dir / "pcm_l3_source_latency_raw" / "session_a"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report-cumulative.csv").write_text(
        textwrap.dedent(f"""\
        AMDuProfPcm Report

        L3 METRICS
        Metric,System (Aggregated),Package (Aggregated)-0
        L3 Access,{l3_access},0
        L3 Miss,{l3_miss},0
        L3 Miss / second,{l3_miss_per_second},0
        L3 Miss %,25.00,0
        """),
        encoding="utf-8",
    )


def _write_custom_pcm_json(case_dir: Path) -> None:
    report_dir = case_dir / "pcm_l3_source_latency_raw" / "session_json"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": [
            {
                "name": "SAMPLE_INTERVAL (ms)",
                "value": "1000",
            },
        ],
        "metrics": [
            {
                "name": "L3 Miss",
                "group": [{
                    "l3": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "sum": 3000.0,
                    "average": 1500.0,
                },
                "series": [1000.0, 2000.0],
            },
            {
                "name": "L3 Access",
                "group": [{
                    "l3": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "sum": 9000.0,
                    "average": 4500.0,
                },
                "series": [4000.0, 5000.0],
            },
            {
                "name": "L3 Miss / second",
                "group": [{
                    "l3": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "sum": 3000.0,
                    "average": 1500.0,
                },
                "series": [1000.0, 2000.0],
            },
        ],
    }
    (report_dir / "report.json").write_text(json.dumps(report),
                                            encoding="utf-8")


def _write_standard_pcm(case_dir: Path) -> None:
    report_dir = case_dir / "pcm_l3_dc" / "session_a"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": [
            {
                "name": "SAMPLE_INTERVAL (ms)",
                "value": "1000",
            },
        ],
        "metrics": [
            {
                "name": "IPC (Sys + User)",
                "group": [{
                    "ipc": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 2.5,
                },
                "series": [2.0, 3.0],
            },
            {
                "name": "Giga Instructions Per Sec",
                "group": [{
                    "ipc": [{
                        "name": "system",
                    }],
                }],
                "series": [2.0, 2.0],
            },
            {
                "name": "L2 Access (pti)",
                "group": [{
                    "l2": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 10.0,
                },
                "series": [9.0, 11.0],
            },
            {
                "name": "L2 Miss (pti)",
                "group": [{
                    "l2": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 1.5,
                },
                "series": [1.0, 2.0],
            },
            {
                "name": "L3 Miss (pti)",
                "group": [{
                    "l3": [{
                        "name": "system",
                    }],
                }],
                "series": [1.0, 1.0],
            },
            {
                "name": "L3 Access (pti)",
                "group": [{
                    "l3": [{
                        "name": "system",
                    }],
                }],
                "series": [4.0, 4.0],
            },
            {
                "name": "All Demand DC Fills (pti)",
                "group": [{
                    "dc": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 8.0,
                },
                "series": [7.0, 9.0],
            },
            {
                "name": "Demand DC Fills From Local L2 (pti)",
                "group": [{
                    "dc": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 7.0,
                },
                "series": [6.0, 8.0],
            },
            {
                "name": "Demand DC Fills From Local L3 or different L2 in same CCX (pti)",
                "group": [{
                    "dc": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 0.6,
                },
                "series": [0.5, 0.7],
            },
            {
                "name": "Demand DC Fills From another CCX in same node (pti)",
                "group": [{
                    "dc": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 0.3,
                },
                "series": [0.2, 0.4],
            },
            {
                "name": "Demand DC Fills From Local Memory or I/O (pti)",
                "group": [{
                    "dc": [{
                        "name": "system",
                    }],
                }],
                "aggregated": {
                    "average": 0.1,
                },
                "series": [0.0, 0.2],
            },
        ],
    }
    (report_dir / "report.json").write_text(json.dumps(report),
                                            encoding="utf-8")


class TestP3AttnOnlyPcmL3TaskMetrics(unittest.TestCase):

    def test_span_layout_case_dirs_are_discovered(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_task_metrics")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "span1" / "q16384_kv16384" / "balanced"
            _write_profile(case_dir)
            _write_custom_pcm(case_dir)

            rows = module.build_rows(root)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("q16384_kv16384", row["case"])
        self.assertEqual("balanced", row["mode"])
        self.assertTrue(str(case_dir) in row["case_dir"])

    def test_custom_pcm_cumulative_csv_is_preferred(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_task_metrics")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "q16384_kv16384" / "balanced"
            _write_profile(case_dir)
            _write_custom_pcm(case_dir)
            _write_standard_pcm(case_dir)

            rows = module.build_rows(root)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("custom_cumulative_csv", row["pcm_source"])
        self.assertEqual(20.0, row["collected_s"])
        self.assertEqual(25600.0, row["estimated_sampled_task_count"])
        self.assertEqual(2000.0, row["l3_miss"])
        self.assertAlmostEqual(0.078125, row["l3_miss_per_task"])
        self.assertAlmostEqual(0.3125, row["l3_access_per_task"])

    def test_custom_pcm_can_be_supplemented_with_standard_pcm_averages(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_task_metrics")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "q16384_kv16384" / "balanced"
            _write_profile(case_dir)
            _write_custom_pcm(case_dir)
            _write_standard_pcm(case_dir)

            rows = module.build_rows(root)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("custom_cumulative_csv", row["pcm_source"])
        self.assertEqual(2.5, row["ipc_sys_user"])
        self.assertEqual(10.0, row["l2_access_pti"])
        self.assertEqual(1.5, row["l2_miss_pti"])
        self.assertEqual(4.0, row["l3_access_pti"])
        self.assertEqual(1.0, row["l3_miss_pti"])
        self.assertEqual(8.0, row["all_demand_dc_fills_pti"])
        self.assertEqual(7.0, row["demand_dc_fills_local_l2_pti"])
        self.assertEqual(0.6, row["demand_dc_fills_local_l3_same_ccx_pti"])
        self.assertEqual(0.3, row["demand_dc_fills_another_ccx_same_node_pti"])
        self.assertEqual(0.1, row["demand_dc_fills_local_memory_io_pti"])

    def test_custom_pcm_report_json_is_used_when_cumulative_csv_is_missing(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_task_metrics")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "q16384_kv16384" / "balanced"
            _write_profile(case_dir)
            _write_custom_pcm_json(case_dir)
            _write_standard_pcm(case_dir)

            rows = module.build_rows(root)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("custom_report_json", row["pcm_source"])
        self.assertEqual(2.0, row["collected_s"])
        self.assertEqual(2560.0, row["estimated_sampled_task_count"])
        self.assertEqual(3000.0, row["l3_miss"])
        self.assertEqual(9000.0, row["l3_access"])
        self.assertAlmostEqual(1.171875, row["l3_miss_per_task"])
        self.assertAlmostEqual(3.515625, row["l3_access_per_task"])

    def test_standard_pcm_report_json_fallback(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_task_metrics")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "q8192_kv8192" / "acc-local-l3"
            _write_profile(case_dir,
                           slowest_rank_mean_ms=10.0,
                           call_count=100,
                           task_count=1000)
            _write_standard_pcm(case_dir)

            rows = module.build_rows(root)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("standard_report_json", row["pcm_source"])
        self.assertEqual(2.0, row["collected_s"])
        self.assertEqual(2000.0, row["estimated_sampled_task_count"])
        self.assertEqual(4_000_000.0, row["l3_miss"])
        self.assertEqual(16_000_000.0, row["l3_access"])
        self.assertEqual(2000.0, row["l3_miss_per_task"])
        self.assertEqual(8000.0, row["l3_access_per_task"])
        self.assertEqual(2.5, row["ipc_sys_user"])
        self.assertEqual(10.0, row["l2_access_pti"])
        self.assertEqual(1.5, row["l2_miss_pti"])
        self.assertEqual(4.0, row["l3_access_pti"])
        self.assertEqual(1.0, row["l3_miss_pti"])
        self.assertEqual(8.0, row["all_demand_dc_fills_pti"])
        self.assertEqual(7.0, row["demand_dc_fills_local_l2_pti"])
        self.assertEqual(0.6, row["demand_dc_fills_local_l3_same_ccx_pti"])
        self.assertEqual(0.3, row["demand_dc_fills_another_ccx_same_node_pti"])
        self.assertEqual(0.1, row["demand_dc_fills_local_memory_io_pti"])

    def test_case_without_profile_is_still_discovered_from_pcm_reports(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_task_metrics")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "q32768_kv32768" / "balanced"
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_standard_pcm(case_dir)

            rows = module.build_rows(root)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("q32768_kv32768", row["case"])
        self.assertEqual("balanced", row["mode"])
        self.assertEqual("standard_report_json", row["pcm_source"])
        self.assertIsNone(row["benchmark_runtime_s"])
        self.assertEqual(4_000_000.0, row["l3_miss"])
        self.assertIsNone(row["estimated_sampled_task_count"])
        self.assertIsNone(row["l3_miss_per_task"])

    def test_write_csv(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_task_metrics")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "q16384_kv16384" / "balanced"
            _write_profile(case_dir)
            _write_custom_pcm(case_dir)
            out_csv = root / "l3_task_metrics.csv"

            module.write_csv(module.build_rows(root), out_csv)

            with out_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual("q16384_kv16384", rows[0]["case"])
        self.assertEqual("balanced", rows[0]["mode"])
        self.assertEqual("0.08", rows[0]["l3_miss_per_task"])


if __name__ == "__main__":
    unittest.main()
