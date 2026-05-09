import csv
import importlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


SYSTEM_GROUP = [{"l3": [{"id": -1, "name": "system"}]}]


def _metric(name: str, average: float, unit: str = "%"):
    return {
        "name": name,
        "source": "L3",
        "unit": unit,
        "title": name,
        "group": SYSTEM_GROUP,
        "aggregated": {
            "average": average,
            "sum": average,
            "minimum": average,
            "maximum": average,
            "95th percentile": average,
        },
        "series": [average],
    }


def _write_case(root: Path,
                q_len: int,
                kv_len: int,
                mode_dir: str,
                mode_name: str,
                slowest_rank_mean_ms: float,
                session_name: str | None,
                metrics: dict[str, float] | None,
                group_span: int = 4,
                profile_payload: dict | None = None):
    case_dir = (root / "prefill-like" / "global-fixed" / "batch_16" /
                f"q{q_len}_kv{kv_len}" / mode_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    dry_run_summary = {
        "workload": "prefill-like",
        "batch_size": 16,
        "requested_lengths": {
            "q_len": q_len,
            "kv_len": kv_len,
        },
        "resolved_lengths": {
            "q_len": q_len,
            "kv_len": kv_len,
        },
        "result_shape_dirname": f"batch_16/q{q_len}_KV_{kv_len}",
        "head_plan": {
            "partition_mode": "global-fixed",
            "tp_size": 2,
            "global_num_query_heads": 32,
            "global_num_kv_heads": 4,
            "local_num_query_heads": 16,
            "local_num_kv_heads": 2,
        },
        "attn_locality_mode": mode_name,
        "attn_locality_group_span": group_span,
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
        "rank_results": [],
    }
    (case_dir / "dry_run_summary.json").write_text(
        json.dumps(dry_run_summary), encoding="utf-8")

    if session_name is None or metrics is None:
        if profile_payload is not None:
            (case_dir / "profile.json").write_text(
                json.dumps(profile_payload), encoding="utf-8")
        return case_dir

    report_dir = case_dir / "pcm_l3_dc" / session_name
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": [],
        "hierarchy": [],
        "metric-groups": [],
        "sections": [],
        "precision": 2,
        "metrics": [
            _metric(name, value, unit="pti" if "(pti)" in name else
                    ("ns" if "(ns)" in name else "%"))
            for name, value in metrics.items()
        ],
    }
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if profile_payload is not None:
        (case_dir / "profile.json").write_text(
            json.dumps(profile_payload), encoding="utf-8")
    return case_dir


def _profile_payload(mode_name: str,
                     slowest_rank_mean_ms: float,
                     scheduler_metadata_avg_ns: float,
                     attention_task_body_avg_ns: float,
                     execute_attention_avg_ns: float,
                     attention_task_non_execute_pct: float,
                     slowest_rank_attention_parallelism: float,
                     slowest_rank_execute_parallelism: float,
                     legacy_scheduler_metadata_avg_ns: float | None = None,
                     locality_metadata_build_avg_ns: float | None = None):
    scheduler_call_count = 2
    runtime_call_count = 4
    attention_task_body_ns = 14_400_000_000
    execute_attention_ns = attention_task_body_ns * (
        1.0 - attention_task_non_execute_pct / 100.0)
    slowest_elapsed_ms = 100.0
    slowest_attention_task_body_ns = (
        slowest_rank_attention_parallelism * slowest_elapsed_ms * 1e6)
    slowest_execute_attention_ns = (
        slowest_rank_execute_parallelism * slowest_elapsed_ms * 1e6)
    scheduler = {
        "call_count": scheduler_call_count,
        "scheduler_metadata_ns":
        scheduler_metadata_avg_ns * scheduler_call_count,
        "scheduler_metadata_avg_ns": scheduler_metadata_avg_ns,
    }
    if legacy_scheduler_metadata_avg_ns is not None:
        scheduler["legacy_scheduler_metadata_ns"] = (
            legacy_scheduler_metadata_avg_ns * scheduler_call_count)
    if locality_metadata_build_avg_ns is not None:
        scheduler["locality_metadata_build_ns"] = (
            locality_metadata_build_avg_ns * scheduler_call_count)

    runtime = {
        "call_count": runtime_call_count,
        "attention_task_body_ns": attention_task_body_ns,
        "attention_task_count": (
            attention_task_body_ns / attention_task_body_avg_ns),
        "attention_task_body_avg_ns": attention_task_body_avg_ns,
        "reduction_task_body_ns": 0,
        "reduction_task_count": 0,
        "reduction_task_body_avg_ns": None,
        "execute_attention_ns": execute_attention_ns,
        "execute_attention_count": (
            execute_attention_ns / execute_attention_avg_ns),
        "execute_attention_avg_ns": execute_attention_avg_ns,
        "attention_task_non_execute_ns":
        attention_task_body_ns - execute_attention_ns,
    }
    if mode_name == "acc-local-l3":
        runtime["attention_counter_fetch_ns"] = 0
        runtime["attention_counter_fetch_count"] = 0
        runtime["reduction_counter_fetch_ns"] = 0
        runtime["reduction_counter_fetch_count"] = 0

    return {
        "workload": "prefill-like",
        "batch_size": 16,
        "requested_lengths": {
            "q_len": 64,
            "kv_len": 64,
        },
        "resolved_lengths": {
            "q_len": 64,
            "kv_len": 64,
        },
        "result_shape_dirname": "batch_16/q64_KV_64",
        "head_plan": {
            "partition_mode": "global-fixed",
            "tp_size": 2,
            "global_num_query_heads": 32,
            "global_num_kv_heads": 4,
            "local_num_query_heads": 16,
            "local_num_kv_heads": 2,
        },
        "attn_locality_mode": mode_name,
        "attn_locality_group_span": 4,
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
        "profile_summary": {
            "mode": mode_name,
            "scheduler": scheduler,
            "runtime": runtime,
        },
        "rank_results": [
            {
                "rank": 0,
                "elapsed_ms": slowest_elapsed_ms,
                "result": {
                    "time_mean_ms": slowest_rank_mean_ms,
                },
                "profile": {
                    "mode": mode_name,
                    "scheduler": scheduler,
                    "runtime": {
                        "call_count": runtime_call_count / 2,
                        "attention_task_body_ns":
                        slowest_attention_task_body_ns,
                        "attention_task_count": (
                            slowest_attention_task_body_ns /
                            attention_task_body_avg_ns),
                        "attention_task_body_avg_ns":
                        attention_task_body_avg_ns,
                        "reduction_task_body_ns": 0,
                        "reduction_task_count": 0,
                        "reduction_task_body_avg_ns": None,
                        "execute_attention_ns":
                        slowest_execute_attention_ns,
                        "execute_attention_count": (
                            slowest_execute_attention_ns /
                            execute_attention_avg_ns),
                        "execute_attention_avg_ns":
                        execute_attention_avg_ns,
                        "attention_task_non_execute_ns":
                        slowest_attention_task_body_ns -
                        slowest_execute_attention_ns,
                    },
                },
            },
            {
                "rank": 1,
                "elapsed_ms": 90.0,
                "result": {
                    "time_mean_ms": slowest_rank_mean_ms * 0.95,
                },
                "profile": {
                    "mode": mode_name,
                    "scheduler": scheduler,
                    "runtime": {
                        "call_count": runtime_call_count / 2,
                        "attention_task_body_ns":
                        slowest_attention_task_body_ns * 0.9,
                        "attention_task_count": (
                            slowest_attention_task_body_ns * 0.9 /
                            attention_task_body_avg_ns),
                        "attention_task_body_avg_ns":
                        attention_task_body_avg_ns,
                        "reduction_task_body_ns": 0,
                        "reduction_task_count": 0,
                        "reduction_task_body_avg_ns": None,
                        "execute_attention_ns":
                        slowest_execute_attention_ns * 0.9,
                        "execute_attention_count": (
                            slowest_execute_attention_ns * 0.9 /
                            execute_attention_avg_ns),
                        "execute_attention_avg_ns":
                        execute_attention_avg_ns,
                        "attention_task_non_execute_ns":
                        (slowest_attention_task_body_ns -
                         slowest_execute_attention_ns) * 0.9,
                    },
                },
            },
        ],
    }


class TestP3AttnOnlyPcmCompare(unittest.TestCase):

    def test_parse_args_uses_default_result_root(self):
        module = importlib.import_module("tools.p3_attn_only.pcm_compare")

        with mock.patch("sys.argv", ["pcm_compare.py"]):
            args = module.parse_args()

        self.assertIsInstance(args, Namespace)
        self.assertEqual(
            Path(
                "test_results/P3_AttnOnly/Qwen3-30B-A3B/"
                "qhead_32_kvhead_16/NPS1_TP2/"
                "prefill-like/global-fixed/batch_1"),
            args.result_root,
        )

    def test_discover_cases_includes_mode_and_missing_pcm(self):
        module = importlib.import_module("tools.p3_attn_only.pcm_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="balanced_span4",
                mode_name="balanced",
                slowest_rank_mean_ms=0.10,
                session_name="session_a",
                metrics={
                    "L3 Access (pti)": 6.0,
                    "DC Fills From Local L2 (pti)": 0.3,
                    "Demand DC Fills From another CCX in same node (pti)": 1.2,
                },
            )
            _write_case(
                root,
                q_len=128,
                kv_len=128,
                mode_dir="acc-local-l3_span4",
                mode_name="acc-local-l3",
                slowest_rank_mean_ms=0.20,
                session_name=None,
                metrics=None,
            )

            cases = module.discover_cases(root)

            self.assertEqual(2, len(cases))
            self.assertEqual("balanced_span4", cases[0]["mode_dir"])
            self.assertEqual("acc-local-l3_span4", cases[1]["mode_dir"])
            self.assertIsNotNone(cases[0]["report_json"])
            self.assertIsNone(cases[1]["report_json"])

    def test_build_compare_rows_pivots_balanced_vs_acc_metrics(self):
        module = importlib.import_module("tools.p3_attn_only.pcm_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common_balanced = {
                "All DC Fills (pti)": 30.98,
                "DC Fills From Local L2 (pti)": 10.01,
                "DC Fills From Local L3 or different L2 in same CCX (pti)": 5.11,
                "DC Fills From another CCX in same node (pti)": 1.91,
                "DC Fills From Local Memory or I/O (pti)": 0.31,
                "DC Fills From another CCX in remote node (pti)": 0.01,
                "DC Fills From Remote Memory or I/O (pti)": 0.02,
                "All Demand DC Fills (pti)": 13.15,
                "Demand DC Fills From Local L2 (pti)": 6.21,
                "L3 Access (pti)": 6.20,
                "L3 Miss (pti)": 2.59,
                "L3 Miss %": 41.79,
                "Ave L3 Miss Latency (ns)": 478.60,
                "Demand DC Fills From Local L3 or different L2 in same CCX (pti)":
                0.50,
                "Demand DC Fills From another CCX in same node (pti)": 1.20,
                "Demand DC Fills From Local Memory or I/O (pti)": 0.10,
                "Demand DC Fills From another CCX in remote node (pti)": 0.01,
                "Demand DC Fills From Remote memory or I/O (pti)": 0.02,
                "L3 Miss Latency From Local Memory or I/O (%)": 0.65,
                "L3 Miss Latency From another CCX in same node (%)": 99.30,
                "L3 Miss Latency From another CCX in remote node (%)": 0.04,
                "L3 Miss Latency From Remote Memory or I/O (%)": 0.03,
                "IPC (Sys + User)": 1.41,
                "IPC (Sys)": 0.22,
                "IPC (User)": 1.50,
                "CPI (Sys + User)": 0.71,
                "CPI (Sys)": 4.47,
                "CPI (User)": 0.66,
            }
            common_acc = {
                "All DC Fills (pti)": 8.92,
                "DC Fills From Local L2 (pti)": 4.02,
                "DC Fills From Local L3 or different L2 in same CCX (pti)": 0.41,
                "DC Fills From another CCX in same node (pti)": 0.52,
                "DC Fills From Local Memory or I/O (pti)": 0.16,
                "DC Fills From another CCX in remote node (pti)": 0.01,
                "DC Fills From Remote Memory or I/O (pti)": 0.01,
                "All Demand DC Fills (pti)": 1.76,
                "Demand DC Fills From Local L2 (pti)": 1.11,
                "L3 Access (pti)": 0.90,
                "L3 Miss (pti)": 0.88,
                "L3 Miss %": 97.91,
                "Ave L3 Miss Latency (ns)": 154.21,
                "Demand DC Fills From Local L3 or different L2 in same CCX (pti)":
                0.05,
                "Demand DC Fills From another CCX in same node (pti)": 0.80,
                "Demand DC Fills From Local Memory or I/O (pti)": 0.02,
                "Demand DC Fills From another CCX in remote node (pti)": 0.01,
                "Demand DC Fills From Remote memory or I/O (pti)": 0.01,
                "L3 Miss Latency From Local Memory or I/O (%)": 0.72,
                "L3 Miss Latency From another CCX in same node (%)": 98.72,
                "L3 Miss Latency From another CCX in remote node (%)": 0.40,
                "L3 Miss Latency From Remote Memory or I/O (%)": 0.17,
                "IPC (Sys + User)": 1.52,
                "IPC (Sys)": 0.28,
                "IPC (User)": 1.61,
                "CPI (Sys + User)": 0.66,
                "CPI (Sys)": 3.61,
                "CPI (User)": 0.62,
            }
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="balanced_span4",
                mode_name="balanced",
                slowest_rank_mean_ms=0.074393,
                session_name="session_a",
                metrics=common_balanced,
            )
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="acc-local-l3_span4",
                mode_name="acc-local-l3",
                slowest_rank_mean_ms=0.067254,
                session_name="session_b",
                metrics=common_acc,
            )
            _write_case(
                root,
                q_len=128,
                kv_len=128,
                mode_dir="balanced_span4",
                mode_name="balanced",
                slowest_rank_mean_ms=0.189787,
                session_name="session_c",
                metrics=common_balanced,
            )
            _write_case(
                root,
                q_len=128,
                kv_len=128,
                mode_dir="acc-local-l3_span4",
                mode_name="acc-local-l3",
                slowest_rank_mean_ms=0.170897,
                session_name=None,
                metrics=None,
            )

            rows = module.build_compare_rows(root)

            self.assertEqual(2, len(rows))
            row_64 = rows[0]
            self.assertEqual(64, row_64["q_len"])
            self.assertAlmostEqual(-9.596332988318803,
                                   row_64["slowest_rank_mean_ms_delta_pct"])
            self.assertEqual(
                1.2,
                row_64["balanced_demand_another_ccx_same_node_pti"],
            )
            self.assertEqual(
                0.8,
                row_64["acc_local_l3_demand_another_ccx_same_node_pti"],
            )
            self.assertEqual(1.41, row_64["balanced_ipc_sys_user"])
            self.assertEqual(0.66, row_64["acc_local_l3_cpi_sys_user"])

            row_128 = rows[1]
            self.assertIsNone(row_128["acc_local_l3_l3_access_pti"])
            self.assertEqual("missing report.json", row_128["acc_local_l3_note"])

    def test_build_compare_rows_includes_profile_analysis_metrics(self):
        module = importlib.import_module("tools.p3_attn_only.pcm_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="balanced",
                mode_name="balanced",
                slowest_rank_mean_ms=100.0,
                session_name=None,
                metrics=None,
                group_span=1,
                profile_payload=_profile_payload(
                    mode_name="balanced",
                    slowest_rank_mean_ms=100.0,
                    scheduler_metadata_avg_ns=100.0,
                    attention_task_body_avg_ns=1000.0,
                    execute_attention_avg_ns=900.0,
                    attention_task_non_execute_pct=10.0,
                    slowest_rank_attention_parallelism=80.0,
                    slowest_rank_execute_parallelism=72.0,
                ),
            )
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="acc-local-l3",
                mode_name="acc-local-l3",
                slowest_rank_mean_ms=108.0,
                session_name=None,
                metrics=None,
                group_span=1,
                profile_payload=_profile_payload(
                    mode_name="acc-local-l3",
                    slowest_rank_mean_ms=108.0,
                    scheduler_metadata_avg_ns=112.0,
                    attention_task_body_avg_ns=1050.0,
                    execute_attention_avg_ns=945.0,
                    attention_task_non_execute_pct=10.0,
                    slowest_rank_attention_parallelism=72.0,
                    slowest_rank_execute_parallelism=68.4,
                    legacy_scheduler_metadata_avg_ns=105.0,
                    locality_metadata_build_avg_ns=7.0,
                ),
            )

            rows = module.build_compare_rows(root)

            self.assertEqual(1, len(rows))
            row = rows[0]
            self.assertEqual(100.0, row["balanced_profile_scheduler_metadata_avg_ns"])
            self.assertEqual(
                112.0, row["acc_local_l3_profile_scheduler_metadata_avg_ns"])
            self.assertEqual(
                105.0,
                row["acc_local_l3_profile_legacy_scheduler_metadata_avg_ns"],
            )
            self.assertEqual(
                7.0,
                row["acc_local_l3_profile_locality_metadata_build_avg_ns"],
            )
            self.assertEqual(
                1000.0, row["balanced_profile_attention_task_body_avg_ns"])
            self.assertEqual(
                945.0, row["acc_local_l3_profile_execute_attention_avg_ns"])
            self.assertEqual(
                80.0,
                row["balanced_profile_slowest_rank_attention_parallelism"],
            )
            self.assertEqual(
                68.4,
                row["acc_local_l3_profile_slowest_rank_execute_parallelism"],
            )
            self.assertAlmostEqual(
                12.0,
                row["profile_scheduler_metadata_avg_ns_delta_pct"],
            )
            self.assertAlmostEqual(
                5.0,
                row["profile_execute_attention_avg_ns_delta_pct"],
            )
            self.assertAlmostEqual(
                -10.0,
                row["profile_slowest_rank_attention_parallelism_delta_pct"],
            )

    def test_build_outputs_writes_profile_analysis_rows(self):
        module = importlib.import_module("tools.p3_attn_only.pcm_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="balanced",
                mode_name="balanced",
                slowest_rank_mean_ms=100.0,
                session_name=None,
                metrics=None,
                group_span=1,
                profile_payload=_profile_payload(
                    mode_name="balanced",
                    slowest_rank_mean_ms=100.0,
                    scheduler_metadata_avg_ns=100.0,
                    attention_task_body_avg_ns=1000.0,
                    execute_attention_avg_ns=900.0,
                    attention_task_non_execute_pct=10.0,
                    slowest_rank_attention_parallelism=80.0,
                    slowest_rank_execute_parallelism=72.0,
                ),
            )
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="acc-local-l3",
                mode_name="acc-local-l3",
                slowest_rank_mean_ms=108.0,
                session_name=None,
                metrics=None,
                group_span=1,
                profile_payload=_profile_payload(
                    mode_name="acc-local-l3",
                    slowest_rank_mean_ms=108.0,
                    scheduler_metadata_avg_ns=112.0,
                    attention_task_body_avg_ns=1050.0,
                    execute_attention_avg_ns=945.0,
                    attention_task_non_execute_pct=10.0,
                    slowest_rank_attention_parallelism=72.0,
                    slowest_rank_execute_parallelism=68.4,
                    legacy_scheduler_metadata_avg_ns=105.0,
                    locality_metadata_build_avg_ns=7.0,
                ),
            )

            summary_csv, compare_csv, summary_md = module.build_outputs(root)

            with Path(summary_csv).open(encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            by_metric = {row["metric"]: row for row in summary_rows}
            self.assertEqual(
                "100.00",
                by_metric["Scheduler Metadata Avg (ns)"]["q64_kv64_balanced"],
            )
            self.assertEqual(
                "112.00",
                by_metric["Scheduler Metadata Avg (ns)"][
                    "q64_kv64_acc-local-l3"],
            )
            self.assertEqual(
                "-10.00",
                by_metric["Slowest Rank Attention Effective Threads Delta (%)"][
                    "q64_kv64_acc-local-l3"],
            )

            with Path(compare_csv).open(encoding="utf-8") as f:
                compare_rows = list(csv.DictReader(f))
            compare_by_metric = {row["metric"]: row for row in compare_rows}
            self.assertEqual(
                "105.00",
                compare_by_metric["Legacy Scheduler Metadata Avg (ns)"][
                    "q64_kv64_acc-local-l3"],
            )
            self.assertEqual(
                "5.00",
                compare_by_metric["Execute Attention Avg Delta (%)"][
                    "q64_kv64_acc-local-l3"],
            )

            markdown = Path(summary_md).read_text(encoding="utf-8")
            self.assertIn(
                "| Scheduler Metadata Avg (ns) | 100.00 | 112.00 |",
                markdown,
            )
            self.assertIn(
                "| Slowest Rank Attention Effective Threads Delta (%) | - | -10.00 |",
                markdown,
            )

    def test_build_outputs_writes_csv_and_markdown_with_dc_columns(self):
        module = importlib.import_module("tools.p3_attn_only.pcm_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="balanced_span4",
                mode_name="balanced",
                slowest_rank_mean_ms=0.074393,
                session_name="session_a",
                metrics={
                    "All DC Fills (pti)": 30.98,
                    "DC Fills From Local L2 (pti)": 10.01,
                    "DC Fills From Local L3 or different L2 in same CCX (pti)": 5.11,
                    "DC Fills From another CCX in same node (pti)": 1.91,
                    "DC Fills From Local Memory or I/O (pti)": 0.31,
                    "DC Fills From another CCX in remote node (pti)":
                    0.01,
                    "DC Fills From Remote Memory or I/O (pti)": 0.02,
                    "All Demand DC Fills (pti)": 13.15,
                    "Demand DC Fills From Local L2 (pti)": 6.21,
                    "L3 Access (pti)": 6.20,
                    "L3 Miss (pti)": 2.59,
                    "L3 Miss %": 41.79,
                    "Ave L3 Miss Latency (ns)": 478.60,
                    "Demand DC Fills From Local L3 or different L2 in same CCX (pti)":
                    0.50,
                    "Demand DC Fills From another CCX in same node (pti)": 1.20,
                    "Demand DC Fills From Local Memory or I/O (pti)": 0.10,
                    "Demand DC Fills From another CCX in remote node (pti)":
                    0.01,
                    "Demand DC Fills From Remote memory or I/O (pti)": 0.02,
                    "L3 Miss Latency From Local Memory or I/O (%)": 0.65,
                    "L3 Miss Latency From another CCX in same node (%)": 99.30,
                    "L3 Miss Latency From another CCX in remote node (%)": 0.04,
                    "L3 Miss Latency From Remote Memory or I/O (%)": 0.03,
                    "IPC (Sys + User)": 1.41,
                    "IPC (Sys)": 0.22,
                    "IPC (User)": 1.50,
                    "CPI (Sys + User)": 0.71,
                    "CPI (Sys)": 4.47,
                    "CPI (User)": 0.66,
                },
            )
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="acc-local-l3_span4",
                mode_name="acc-local-l3",
                slowest_rank_mean_ms=0.067254,
                session_name="session_b",
                metrics={
                    "All DC Fills (pti)": 8.92,
                    "DC Fills From Local L2 (pti)": 4.02,
                    "DC Fills From Local L3 or different L2 in same CCX (pti)":
                    0.41,
                    "DC Fills From another CCX in same node (pti)": 0.52,
                    "DC Fills From Local Memory or I/O (pti)": 0.16,
                    "DC Fills From another CCX in remote node (pti)":
                    0.01,
                    "DC Fills From Remote Memory or I/O (pti)": 0.01,
                    "All Demand DC Fills (pti)": 1.76,
                    "Demand DC Fills From Local L2 (pti)": 1.11,
                    "L3 Access (pti)": 0.90,
                    "L3 Miss (pti)": 0.88,
                    "L3 Miss %": 97.91,
                    "Ave L3 Miss Latency (ns)": 154.21,
                    "Demand DC Fills From Local L3 or different L2 in same CCX (pti)":
                    0.05,
                    "Demand DC Fills From another CCX in same node (pti)": 0.80,
                    "Demand DC Fills From Local Memory or I/O (pti)": 0.02,
                    "Demand DC Fills From another CCX in remote node (pti)":
                    0.01,
                    "Demand DC Fills From Remote memory or I/O (pti)": 0.01,
                    "L3 Miss Latency From Local Memory or I/O (%)": 0.72,
                    "L3 Miss Latency From another CCX in same node (%)": 98.72,
                    "L3 Miss Latency From another CCX in remote node (%)": 0.40,
                    "L3 Miss Latency From Remote Memory or I/O (%)": 0.17,
                    "IPC (Sys + User)": 1.52,
                    "IPC (Sys)": 0.28,
                    "IPC (User)": 1.61,
                    "CPI (Sys + User)": 0.66,
                    "CPI (Sys)": 3.61,
                    "CPI (User)": 0.62,
                },
            )

            summary_csv, compare_csv, summary_md = module.build_outputs(root)

            with Path(summary_csv).open(encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(summary_rows), 25)
            self.assertEqual(
                {
                    "metric",
                    "q64_kv64_balanced_span4",
                    "q64_kv64_acc-local-l3_span4",
                },
                set(summary_rows[0].keys()),
            )
            by_metric = {row["metric"]: row for row in summary_rows}
            summary_metric_order = [row["metric"] for row in summary_rows]
            demand_start = summary_metric_order.index("All Demand DC Fills (pti)")
            self.assertEqual(
                [
                    "All Demand DC Fills (pti)",
                    "Demand DC Fills From Local L2 (pti)",
                    "Demand DC Fills From Local L3 or different L2 in same CCX (pti)",
                    "Demand DC Fills From another CCX in same node (pti)",
                    "Demand DC Fills From Local Memory or I/O (pti)",
                    "Demand DC Fills From another CCX in remote node (pti)",
                    "Demand DC Fills From Remote memory or I/O (pti)",
                ],
                summary_metric_order[demand_start:demand_start + 7],
            )
            self.assertEqual(
                "1.20",
                by_metric["Demand DC Fills From another CCX in same node (pti)"][
                    "q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "30.98",
                by_metric["All DC Fills (pti)"]["q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "1.76",
                by_metric["All Demand DC Fills (pti)"][
                    "q64_kv64_acc-local-l3_span4"],
            )
            self.assertEqual(
                "10.01",
                by_metric["DC Fills From Local L2 (pti)"][
                    "q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "1.11",
                by_metric["Demand DC Fills From Local L2 (pti)"][
                    "q64_kv64_acc-local-l3_span4"],
            )
            self.assertEqual("1.4100",
                             by_metric["IPC (Sys + User)"][
                                 "q64_kv64_balanced_span4"])
            self.assertEqual("0.6600",
                             by_metric["CPI (Sys + User)"][
                                 "q64_kv64_acc-local-l3_span4"])

            with Path(compare_csv).open(encoding="utf-8") as f:
                compare_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(compare_rows), 5)
            self.assertEqual(
                {
                    "metric",
                    "q64_kv64_balanced_span4",
                    "q64_kv64_acc-local-l3_span4",
                },
                set(compare_rows[0].keys()),
            )
            compare_by_metric = {row["metric"]: row for row in compare_rows}
            self.assertEqual(
                "0.074393",
                compare_by_metric["slowest_rank_mean_ms"][
                    "q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "-9.60",
                compare_by_metric["slowest_rank_mean_delta_pct"][
                    "q64_kv64_acc-local-l3_span4"],
            )
            self.assertEqual(
                "latest PCM session",
                compare_by_metric["note"]["q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "1.4100",
                compare_by_metric["IPC (Sys + User)"][
                    "q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "0.6600",
                compare_by_metric["CPI (Sys + User)"][
                    "q64_kv64_acc-local-l3_span4"],
            )

            detail_summary_csv = Path(root) / "detail_summary.csv"
            with detail_summary_csv.open(encoding="utf-8") as f:
                detail_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(detail_rows), 10)
            self.assertEqual(
                {
                    "metric",
                    "q64_kv64_balanced_span4",
                    "q64_kv64_acc-local-l3_span4",
                },
                set(detail_rows[0].keys()),
            )
            detail_by_metric = {row["metric"]: row for row in detail_rows}
            detail_metric_order = [row["metric"] for row in detail_rows]
            demand_detail_start = detail_metric_order.index(
                "All Demand DC Fills (pti)")
            self.assertEqual(
                [
                    "All Demand DC Fills (pti)",
                    "Demand DC Fills From Local L2 (pti)",
                    "Demand DC Fills From Local L3 or different L2 in same CCX (pti)",
                    "Demand DC Fills From another CCX in same node (pti)",
                    "Demand DC Fills From Local Memory or I/O (pti)",
                    "Demand DC Fills From another CCX in remote node (pti)",
                    "Demand DC Fills From Remote memory or I/O (pti)",
                ],
                detail_metric_order[demand_detail_start:demand_detail_start + 7],
            )
            self.assertEqual(
                "balanced_span4",
                detail_by_metric["mode_dir"]["q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "acc-local-l3_span4",
                detail_by_metric["mode_dir"]["q64_kv64_acc-local-l3_span4"],
            )
            self.assertEqual(
                "30.98",
                detail_by_metric["All DC Fills (pti)"][
                    "q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "1.76",
                detail_by_metric["All Demand DC Fills (pti)"][
                    "q64_kv64_acc-local-l3_span4"],
            )
            self.assertEqual(
                "1.41",
                detail_by_metric["IPC (Sys + User)"][
                    "q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "0.66",
                detail_by_metric["CPI (Sys + User)"][
                    "q64_kv64_acc-local-l3_span4"],
            )

            markdown = Path(summary_md).read_text(encoding="utf-8")
            self.assertIn(
                "| metric | q64_kv64_balanced_span4 | q64_kv64_acc-local-l3_span4 |",
                markdown,
            )
            self.assertIn(
                "| Demand DC Fills From another CCX in same node (pti) | 1.20 | 0.80 |",
                markdown,
            )
            self.assertIn(
                "| DC Fills From Local L2 (pti) | 10.01 | 4.02 |",
                markdown,
            )
            self.assertIn(
                "| IPC (Sys + User) | 1.4100 | 1.5200 |",
                markdown,
            )
            self.assertIn(
                "| CPI (Sys + User) | 0.7100 | 0.6600 |",
                markdown,
            )

    def test_build_outputs_use_actual_mode_dir_headers_for_group_span1(self):
        module = importlib.import_module("tools.p3_attn_only.pcm_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="balanced",
                mode_name="balanced",
                slowest_rank_mean_ms=0.074393,
                session_name=None,
                metrics=None,
                group_span=1,
            )
            _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="acc-local-l3",
                mode_name="acc-local-l3",
                slowest_rank_mean_ms=0.067254,
                session_name=None,
                metrics=None,
                group_span=1,
            )

            summary_csv, compare_csv, summary_md = module.build_outputs(root)

            with Path(summary_csv).open(encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertEqual(
                {
                    "metric",
                    "q64_kv64_balanced",
                    "q64_kv64_acc-local-l3",
                },
                set(summary_rows[0].keys()),
            )

            with Path(compare_csv).open(encoding="utf-8") as f:
                compare_rows = list(csv.DictReader(f))
            self.assertEqual(
                {
                    "metric",
                    "q64_kv64_balanced",
                    "q64_kv64_acc-local-l3",
                },
                set(compare_rows[0].keys()),
            )

            markdown = Path(summary_md).read_text(encoding="utf-8")
            self.assertIn(
                "| metric | q64_kv64_balanced | q64_kv64_acc-local-l3 |",
                markdown,
            )


if __name__ == "__main__":
    unittest.main()
