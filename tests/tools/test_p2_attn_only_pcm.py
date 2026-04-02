import csv
import importlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SYSTEM_GROUP = [{"l3": [{"id": -1, "name": "system"}]}]


def _metric(name: str,
            average: float,
            group=None,
            series=None,
            source: str = "L3",
            unit: str = "%"):
    return {
        "name": name,
        "source": source,
        "unit": unit,
        "title": name,
        "group": group if group is not None else SYSTEM_GROUP,
        "aggregated": {
            "average": average,
            "sum": average,
            "minimum": average,
            "maximum": average,
            "95th percentile": average,
        },
        "series": series if series is not None else [average],
    }


def _write_case(root: Path,
                workload: str,
                partition_mode: str,
                batch_size: int,
                q_len: int,
                kv_len: int,
                session_name: str,
                slowest_rank_mean_ms: float = 1.23):
    case_dir = (root / workload / partition_mode / f"batch_{batch_size}" /
                f"q{q_len}_kv{kv_len}")
    report_dir = case_dir / "pcm_l3_dc" / session_name
    report_dir.mkdir(parents=True, exist_ok=True)

    dry_run_summary = {
        "workload": workload,
        "batch_size": batch_size,
        "requested_lengths": {
            "q_len": q_len,
            "kv_len": kv_len,
        },
        "resolved_lengths": {
            "q_len": q_len,
            "kv_len": kv_len,
        },
        "result_shape_dirname": f"batch_{batch_size}/q{q_len}_KV_{kv_len}",
        "head_plan": {
            "partition_mode": partition_mode,
            "tp_size": 2,
            "global_num_query_heads": 32,
            "global_num_kv_heads": 4,
            "local_num_query_heads": 16,
            "local_num_kv_heads": 2,
        },
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
        "rank_results": [],
    }
    (case_dir / "dry_run_summary.json").write_text(
        json.dumps(dry_run_summary), encoding="utf-8")

    report = {
        "metadata": [],
        "hierarchy": [],
        "metric-groups": [],
        "sections": [],
        "precision": 2,
        "metrics": [
            _metric("L3 Access (pti)", 24.5, unit="pti"),
            _metric("L3 Miss (pti)", 1.6, unit="pti"),
            _metric("L3 Miss %", 46.2),
            _metric("Ave L3 Miss Latency (ns)", 256.2, unit="ns"),
            _metric("L3 Miss Latency From Local Memory or I/O (%)", 2.3),
            _metric("L3 Miss Latency From Remote Memory or I/O (%)", 0.1),
            _metric("L3 Miss Latency From another CCX in same node (%)", 97.4),
            _metric("L3 Miss Latency From another CCX in remote node (%)",
                    0.2),
            _metric("L3 Miss Latency From Local Extension Memory (CXL) (%)",
                    0.0),
            _metric("L3 Miss Latency From Remote Extension Memory (CXL) (%)",
                    0.0),
            _metric(
                "L3 Miss %",
                99.0,
                group=[{
                    "l3": [{
                        "id": 0,
                        "name": "ccx"
                    }]
                }],
            ),
        ],
    }
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return case_dir, report_dir


class TestP2AttnOnlyPcmSummary(unittest.TestCase):

    def test_discover_cases_prefers_latest_session_and_parses_lengths(self):
        module = importlib.import_module("tools.p2_attn_only.pcm_summary")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_case(
                root,
                workload="decode-like",
                partition_mode="global-fixed",
                batch_size=16,
                q_len=1,
                kv_len=1024,
                session_name="AMDuProfPcm-Multi_Apr-02-2026_13-40-49",
            )
            _, latest_report_dir = _write_case(
                root,
                workload="decode-like",
                partition_mode="global-fixed",
                batch_size=16,
                q_len=1,
                kv_len=1024,
                session_name="AMDuProfPcm-Multi_Apr-02-2026_15-40-49",
                slowest_rank_mean_ms=2.34,
            )

            cases = module.discover_cases(root)

            self.assertEqual(1, len(cases))
            case = cases[0]
            self.assertEqual("decode-like", case["workload"])
            self.assertEqual("global-fixed", case["partition_mode"])
            self.assertEqual(16, case["batch_size"])
            self.assertEqual(1, case["q_len"])
            self.assertEqual(1024, case["kv_len"])
            self.assertEqual(str(latest_report_dir / "report.json"),
                             case["report_json"])

    def test_extract_system_metrics_ignores_ccx_rows(self):
        module = importlib.import_module("tools.p2_attn_only.pcm_summary")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, report_dir = _write_case(
                root,
                workload="prefill-like",
                partition_mode="global-fixed",
                batch_size=16,
                q_len=256,
                kv_len=256,
                session_name="AMDuProfPcm-Multi_Apr-02-2026_15-35-37",
            )

            metrics = module.extract_system_metrics(report_dir / "report.json")

            self.assertEqual(24.5, metrics["L3 Access (pti)"])
            self.assertEqual(46.2, metrics["L3 Miss %"])
            self.assertEqual(256.2, metrics["Ave L3 Miss Latency (ns)"])
            self.assertEqual(
                97.4,
                metrics["L3 Miss Latency From another CCX in same node (%)"],
            )

    def test_build_summary_rows_merges_dry_run_and_report_metrics(self):
        module = importlib.import_module("tools.p2_attn_only.pcm_summary")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_case(
                root,
                workload="prefill-like",
                partition_mode="global-fixed",
                batch_size=16,
                q_len=512,
                kv_len=512,
                session_name="AMDuProfPcm-Multi_Apr-02-2026_15-28-58",
                slowest_rank_mean_ms=5.67,
            )

            rows = module.build_summary_rows(root)

            self.assertEqual(1, len(rows))
            row = rows[0]
            self.assertEqual(5.67, row["slowest_rank_mean_ms"])
            self.assertEqual("global-fixed", row["partition_mode"])
            self.assertEqual("batch_16/q512_KV_512", row["result_shape_dirname"])
            self.assertEqual(1.6, row["L3 Miss (pti)"])
            self.assertEqual(
                0.0,
                row["L3 Miss Latency From Remote Extension Memory (CXL) (%)"],
            )


class TestP2AttnOnlyPcmPlot(unittest.TestCase):

    def test_sort_rows_for_plot_uses_workload_specific_length_axis(self):
        module = importlib.import_module("tools.p2_attn_only.plot_pcm_summary")

        decode_rows = [
            {
                "workload": "decode-like",
                "q_len": 1,
                "kv_len": 2048,
            },
            {
                "workload": "decode-like",
                "q_len": 1,
                "kv_len": 256,
            },
        ]
        prefill_rows = [
            {
                "workload": "prefill-like",
                "q_len": 1024,
                "kv_len": 1024,
            },
            {
                "workload": "prefill-like",
                "q_len": 256,
                "kv_len": 256,
            },
        ]

        self.assertEqual(
            [256, 2048],
            [r["kv_len"] for r in module.sort_rows_for_plot(decode_rows)],
        )
        self.assertEqual(
            [256, 1024],
            [r["q_len"] for r in module.sort_rows_for_plot(prefill_rows)],
        )

    def test_latency_breakdown_sum_is_100_percent(self):
        module = importlib.import_module("tools.p2_attn_only.plot_pcm_summary")

        row = {
            "L3 Miss Latency From Local Memory or I/O (%)": 2.3,
            "L3 Miss Latency From Remote Memory or I/O (%)": 0.1,
            "L3 Miss Latency From another CCX in same node (%)": 97.4,
            "L3 Miss Latency From another CCX in remote node (%)": 0.2,
            "L3 Miss Latency From Local Extension Memory (CXL) (%)": 0.0,
            "L3 Miss Latency From Remote Extension Memory (CXL) (%)": 0.0,
        }

        total = sum(module.get_latency_breakdown_values(row))
        self.assertAlmostEqual(100.0, total, places=6)

    def test_load_summary_csv_reads_numeric_columns(self):
        module = importlib.import_module("tools.p2_attn_only.plot_pcm_summary")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "workload",
                        "q_len",
                        "kv_len",
                        "L3 Miss %",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "workload": "decode-like",
                    "q_len": "1",
                    "kv_len": "1024",
                    "L3 Miss %": "46.2",
                })

            rows = module.load_summary_rows(csv_path)

            self.assertEqual(1, rows[0]["q_len"])
            self.assertEqual(1024, rows[0]["kv_len"])
            self.assertEqual(46.2, rows[0]["L3 Miss %"])

    def test_plot_latency_breakdown_uses_stacked_bars(self):
        module = importlib.import_module("tools.p2_attn_only.plot_pcm_summary")

        rows = [
            {
                "workload": "prefill-like",
                "q_len": 256,
                "kv_len": 256,
                "L3 Miss Latency From Local Memory or I/O (%)": 5.0,
                "L3 Miss Latency From Remote Memory or I/O (%)": 1.0,
                "L3 Miss Latency From another CCX in same node (%)": 90.0,
                "L3 Miss Latency From another CCX in remote node (%)": 4.0,
                "L3 Miss Latency From Local Extension Memory (CXL) (%)": 0.0,
                "L3 Miss Latency From Remote Extension Memory (CXL) (%)": 0.0,
            },
            {
                "workload": "prefill-like",
                "q_len": 512,
                "kv_len": 512,
                "L3 Miss Latency From Local Memory or I/O (%)": 10.0,
                "L3 Miss Latency From Remote Memory or I/O (%)": 2.0,
                "L3 Miss Latency From another CCX in same node (%)": 80.0,
                "L3 Miss Latency From another CCX in remote node (%)": 8.0,
                "L3 Miss Latency From Local Extension Memory (CXL) (%)": 0.0,
                "L3 Miss Latency From Remote Extension Memory (CXL) (%)": 0.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "breakdown.png"
            with mock.patch.object(module.plt, "bar") as bar_mock, \
                 mock.patch.object(module.plt, "stackplot") as stackplot_mock, \
                 mock.patch.object(module.plt, "figure"), \
                 mock.patch.object(module.plt, "xlabel"), \
                 mock.patch.object(module.plt, "ylabel"), \
                 mock.patch.object(module.plt, "title"), \
                 mock.patch.object(module.plt, "ylim"), \
                 mock.patch.object(module.plt, "legend"), \
                 mock.patch.object(module.plt, "tight_layout"), \
                 mock.patch.object(module.plt, "savefig"), \
                 mock.patch.object(module.plt, "close"):
                module.plot_latency_breakdown(rows, "prefill-like", output_path)

            self.assertEqual(len(module.LATENCY_BREAKDOWN_METRIC_NAMES),
                             bar_mock.call_count)
            self.assertFalse(stackplot_mock.called)


if __name__ == "__main__":
    unittest.main()
