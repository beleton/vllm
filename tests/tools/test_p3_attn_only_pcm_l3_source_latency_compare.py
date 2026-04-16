import csv
import importlib
import json
import tempfile
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


def _write_case(root: Path,
                q_len: int,
                kv_len: int,
                mode_dir: str,
                mode_name: str,
                slowest_rank_mean_ms: float,
                session_name: str | None,
                metrics: dict[str, float] | None) -> Path:
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
        "head_plan": {
            "partition_mode": "global-fixed",
        },
        "attn_locality_mode": mode_name,
        "attn_locality_group_span": 4,
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
        "rank_results": [],
    }
    (case_dir / "dry_run_summary.json").write_text(
        json.dumps(dry_run_summary), encoding="utf-8")

    if session_name is None or metrics is None:
        return case_dir

    report_dir = case_dir / "pcm_l3_source_latency_raw" / session_name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report-cumulative.csv"
    report_path.write_text(
        textwrap.dedent(f"""\
        AMDuProfPcm Report

        CORE METRICS
        Metric,System (Aggregated),Package (Aggregated)-0
        IPC (Sys + User),{metrics["IPC (Sys + User)"]:.2f},0.00
        IPC (Sys),{metrics["IPC (Sys)"]:.2f},0.00
        IPC (User),{metrics["IPC (User)"]:.2f},0.00
        CPI (Sys + User),{metrics["CPI (Sys + User)"]:.2f},0.00
        CPI (Sys),{metrics["CPI (Sys)"]:.2f},0.00
        CPI (User),{metrics["CPI (User)"]:.2f},0.00
        Utilization (%),49.91,0.00

        L3 METRICS
        Metric,System (Aggregated),Package (Aggregated)-0
        Raw L3SampledLatencyAll,{metrics["Raw L3SampledLatencyAll"]:.2f},0.00
        Raw L3SampledLatencyRequestsAll,{metrics["Raw L3SampledLatencyRequestsAll"]:.2f},0.00
        Derived Avg L3 Miss Latency (ns),{metrics["Derived Avg L3 Miss Latency (ns)"]:.2f},0.00
        Raw L3SampledLatencyFromLocalMemory,{metrics["Raw L3SampledLatencyFromLocalMemory"]:.2f},0.00
        Raw L3SampledLatencyRequestsFromLocalMemory,{metrics["Raw L3SampledLatencyRequestsFromLocalMemory"]:.2f},0.00
        Derived Local Memory Avg L3 Miss Latency (ns),{metrics["Derived Local Memory Avg L3 Miss Latency (ns)"]:.2f},0.00
        Derived Local Memory L3 Miss Latency Share (%),{metrics["Derived Local Memory L3 Miss Latency Share (%)"]:.2f},0.00
        Raw L3SampledLatencyFromRemoteMemory,{metrics["Raw L3SampledLatencyFromRemoteMemory"]:.2f},0.00
        Raw L3SampledLatencyRequestsFromRemoteMemory,{metrics["Raw L3SampledLatencyRequestsFromRemoteMemory"]:.2f},0.00
        Derived Remote Memory Avg L3 Miss Latency (ns),{metrics["Derived Remote Memory Avg L3 Miss Latency (ns)"]:.2f},0.00
        Derived Remote Memory L3 Miss Latency Share (%),{metrics["Derived Remote Memory L3 Miss Latency Share (%)"]:.2f},0.00
        Raw L3SampledLatencyFromExternalCacheLocal,{metrics["Raw L3SampledLatencyFromExternalCacheLocal"]:.2f},0.00
        Raw L3SampledLatencyRequestsFromExternalCacheLocal,{metrics["Raw L3SampledLatencyRequestsFromExternalCacheLocal"]:.2f},0.00
        Derived another CCX in same node Avg L3 Miss Latency (ns),{metrics["Derived another CCX in same node Avg L3 Miss Latency (ns)"]:.2f},0.00
        Derived another CCX in same node L3 Miss Latency Share (%),{metrics["Derived another CCX in same node L3 Miss Latency Share (%)"]:.2f},0.00
        Raw L3SampledLatencyFromExternalCacheRemote,{metrics["Raw L3SampledLatencyFromExternalCacheRemote"]:.2f},0.00
        Raw L3SampledLatencyRequestsFromExternalCacheRemote,{metrics["Raw L3SampledLatencyRequestsFromExternalCacheRemote"]:.2f},0.00
        Derived another CCX in remote node Avg L3 Miss Latency (ns),{metrics["Derived another CCX in remote node Avg L3 Miss Latency (ns)"]:.2f},0.00
        Derived another CCX in remote node L3 Miss Latency Share (%),{metrics["Derived another CCX in remote node L3 Miss Latency Share (%)"]:.2f},0.00
        """),
        encoding="utf-8",
    )
    return case_dir


class TestP3AttnOnlyPcmL3SourceLatencyCompare(unittest.TestCase):

    def test_parse_args_uses_default_result_root(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_source_latency_compare")

        with mock.patch("sys.argv", ["pcm_l3_source_latency_compare.py"]):
            args = module.parse_args()

        self.assertIsInstance(args, Namespace)
        self.assertEqual(
            Path(
                "test_results/P3_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/"
                "prefill-like/global-fixed/batch_16"),
            args.result_root,
        )

    def test_extract_system_metrics_from_cumulative_csv(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_source_latency_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = _write_case(
                root,
                q_len=64,
                kv_len=64,
                mode_dir="balanced_span4",
                mode_name="balanced",
                slowest_rank_mean_ms=0.10,
                session_name="session_a",
                metrics={
                    "IPC (Sys + User)": 1.82,
                    "IPC (Sys)": 0.75,
                    "IPC (User)": 1.84,
                    "CPI (Sys + User)": 0.55,
                    "CPI (Sys)": 1.34,
                    "CPI (User)": 0.54,
                    "Raw L3SampledLatencyAll": 90755800163.0,
                    "Raw L3SampledLatencyRequestsAll": 1932407338.0,
                    "Derived Avg L3 Miss Latency (ns)": 469.65,
                    "Raw L3SampledLatencyFromLocalMemory": 541084448.0,
                    "Raw L3SampledLatencyRequestsFromLocalMemory": 23913020.0,
                    "Derived Local Memory Avg L3 Miss Latency (ns)": 226.27,
                    "Derived Local Memory L3 Miss Latency Share (%)": 0.60,
                    "Raw L3SampledLatencyFromRemoteMemory": 46569103.0,
                    "Raw L3SampledLatencyRequestsFromRemoteMemory": 1626560.0,
                    "Derived Remote Memory Avg L3 Miss Latency (ns)": 286.30,
                    "Derived Remote Memory L3 Miss Latency Share (%)": 0.05,
                    "Raw L3SampledLatencyFromExternalCacheLocal": 89793045927.0,
                    "Raw L3SampledLatencyRequestsFromExternalCacheLocal":
                    1901512297.0,
                    "Derived another CCX in same node Avg L3 Miss Latency (ns)":
                    472.22,
                    "Derived another CCX in same node L3 Miss Latency Share (%)":
                    98.94,
                    "Raw L3SampledLatencyFromExternalCacheRemote": 19140016.0,
                    "Raw L3SampledLatencyRequestsFromExternalCacheRemote":
                    606023.0,
                    "Derived another CCX in remote node Avg L3 Miss Latency (ns)":
                    315.83,
                    "Derived another CCX in remote node L3 Miss Latency Share (%)":
                    0.02,
                },
            )

            metrics = module.extract_system_metrics(
                case_dir / "pcm_l3_source_latency_raw" / "session_a" /
                "report-cumulative.csv")

            self.assertEqual(1.82, metrics["IPC (Sys + User)"])
            self.assertEqual(469.65,
                             metrics["Derived Avg L3 Miss Latency (ns)"])
            self.assertEqual(
                472.22,
                metrics["Derived another CCX in same node Avg L3 Miss Latency (ns)"],
            )
            self.assertNotIn("Utilization (%)", metrics)

    def test_build_outputs_writes_summary_and_detail_csv(self):
        module = importlib.import_module(
            "tools.p3_attn_only.pcm_l3_source_latency_compare")

        common_balanced = {
            "IPC (Sys + User)": 1.82,
            "IPC (Sys)": 0.75,
            "IPC (User)": 1.84,
            "CPI (Sys + User)": 0.55,
            "CPI (Sys)": 1.34,
            "CPI (User)": 0.54,
            "Raw L3SampledLatencyAll": 90755800163.0,
            "Raw L3SampledLatencyRequestsAll": 1932407338.0,
            "Derived Avg L3 Miss Latency (ns)": 469.65,
            "Raw L3SampledLatencyFromLocalMemory": 541084448.0,
            "Raw L3SampledLatencyRequestsFromLocalMemory": 23913020.0,
            "Derived Local Memory Avg L3 Miss Latency (ns)": 226.27,
            "Derived Local Memory L3 Miss Latency Share (%)": 0.60,
            "Raw L3SampledLatencyFromRemoteMemory": 46569103.0,
            "Raw L3SampledLatencyRequestsFromRemoteMemory": 1626560.0,
            "Derived Remote Memory Avg L3 Miss Latency (ns)": 286.30,
            "Derived Remote Memory L3 Miss Latency Share (%)": 0.05,
            "Raw L3SampledLatencyFromExternalCacheLocal": 89793045927.0,
            "Raw L3SampledLatencyRequestsFromExternalCacheLocal": 1901512297.0,
            "Derived another CCX in same node Avg L3 Miss Latency (ns)": 472.22,
            "Derived another CCX in same node L3 Miss Latency Share (%)": 98.94,
            "Raw L3SampledLatencyFromExternalCacheRemote": 19140016.0,
            "Raw L3SampledLatencyRequestsFromExternalCacheRemote": 606023.0,
            "Derived another CCX in remote node Avg L3 Miss Latency (ns)":
            315.83,
            "Derived another CCX in remote node L3 Miss Latency Share (%)":
            0.02,
        }
        common_acc = {
            **common_balanced,
            "IPC (Sys + User)": 1.52,
            "CPI (Sys + User)": 0.66,
            "Derived Avg L3 Miss Latency (ns)": 154.21,
            "Derived Local Memory Avg L3 Miss Latency (ns)": 157.12,
            "Derived another CCX in same node Avg L3 Miss Latency (ns)":
            153.62,
        }

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
                session_name=None,
                metrics=None,
            )

            summary_csv, detail_csv = module.build_outputs(root)

            with Path(summary_csv).open(encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(summary_rows), 20)
            self.assertEqual(
                {
                    "metric",
                    "q64_kv64_balanced_span4",
                    "q64_kv64_acc-local-l3_span4",
                    "q128_kv128_balanced_span4",
                },
                set(summary_rows[0].keys()),
            )
            by_metric = {row["metric"]: row for row in summary_rows}
            self.assertNotIn("case_dir", by_metric)
            self.assertEqual(
                "1.8200",
                by_metric["IPC (Sys + User)"]["q64_kv64_balanced_span4"],
            )
            self.assertEqual(
                "154.21",
                by_metric["Derived Avg L3 Miss Latency (ns)"][
                    "q64_kv64_acc-local-l3_span4"],
            )
            self.assertEqual(
                "-",
                by_metric["Derived Avg L3 Miss Latency (ns)"][
                    "q128_kv128_balanced_span4"],
            )
            self.assertEqual(
                "missing report-cumulative.csv",
                by_metric["note"]["q128_kv128_balanced_span4"],
            )

            with Path(detail_csv).open(encoding="utf-8") as f:
                detail_rows = list(csv.DictReader(f))
            self.assertEqual(3, len(detail_rows))
            self.assertEqual("balanced_span4", detail_rows[0]["mode_dir"])
            self.assertEqual("latest PCM session", detail_rows[0]["note"])
            self.assertEqual(
                "472.22",
                detail_rows[0][
                    "Derived another CCX in same node Avg L3 Miss Latency (ns)"
                ],
            )
            self.assertEqual(
                "",
                detail_rows[2]["Derived Avg L3 Miss Latency (ns)"],
            )
