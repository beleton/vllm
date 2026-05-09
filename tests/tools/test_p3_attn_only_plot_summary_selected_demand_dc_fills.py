import importlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SUMMARY_CSV = """metric,q64_kv64_balanced,q64_kv64_acc-local-l3,q128_kv128_balanced,q128_kv128_acc-local-l3,q256_kv256_balanced,q256_kv256_acc-local-l3
slowest rank mean (ms),0.027242,0.031338,0.029414,0.031510,0.055479,0.062016
All Demand DC Fills (pti),8.80,8.12,7.63,7.15,-,-
Demand DC Fills From Local L3 or different L2 in same CCX (pti),0.19,0.19,0.13,0.13,-,-
Demand DC Fills From Local Memory or I/O (pti),0.07,0.06,0.06,0.04,-,-
"""

SUMMARY_CSV_WITH_ONE_SIDED_GAP = """metric,q64_kv64_balanced,q64_kv64_acc-local-l3,q128_kv128_balanced,q128_kv128_acc-local-l3,q256_kv256_balanced,q256_kv256_acc-local-l3
slowest rank mean (ms),0.027242,0.031338,0.029414,0.031510,0.055479,0.062016
All Demand DC Fills (pti),8.80,8.12,7.63,7.15,-,6.65
Demand DC Fills From Local L3 or different L2 in same CCX (pti),0.19,0.19,0.13,0.13,-,0.08
Demand DC Fills From Local Memory or I/O (pti),0.07,0.06,0.06,0.04,-,0.03
"""


class TestPlotSummarySelectedDemandDCFills(unittest.TestCase):

    def test_load_metric_series_reads_all_target_metrics(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_summary_selected_demand_dc_fills")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV, encoding="utf-8")

            series = module.load_metric_series(csv_path)

        self.assertEqual(
            [(64, 64, 8.80), (128, 128, 7.63)],
            series["All Demand DC Fills (pti)"]["balanced"],
        )
        self.assertEqual(
            [(64, 64, 8.12), (128, 128, 7.15)],
            series["All Demand DC Fills (pti)"]["acc-local-l3"],
        )
        self.assertEqual(
            [(64, 64, 0.19), (128, 128, 0.13)],
            series["Demand DC Fills From Local L3 or different L2 in same CCX (pti)"]["balanced"],
        )
        self.assertEqual(
            [(64, 64, 0.06), (128, 128, 0.04)],
            series["Demand DC Fills From Local Memory or I/O (pti)"]["acc-local-l3"],
        )

    def test_generate_plot_writes_default_png_next_to_summary_csv(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_summary_selected_demand_dc_fills")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV, encoding="utf-8")

            output_path = module.generate_plot(csv_path)

            self.assertEqual(
                csv_path.with_name(
                    "selected_demand_dc_fills_balanced_vs_acc-local-l3.png"),
                output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_generate_plot_writes_explicit_png_with_expected_size(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_summary_selected_demand_dc_fills")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV, encoding="utf-8")
            output_path = Path(tmpdir) / "plot.png"

            generated_path = module.generate_plot(csv_path, output_path)

            self.assertEqual(output_path, generated_path)
            self.assertTrue(output_path.exists())
            with Image.open(output_path) as image:
                self.assertEqual((1360, 500), image.size)

    def test_generate_plot_uses_only_shared_shapes(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_summary_selected_demand_dc_fills")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV_WITH_ONE_SIDED_GAP,
                                encoding="utf-8")

            output_path = module.generate_plot(csv_path)

            self.assertTrue(output_path.exists())

    def test_build_stacked_case_series_groups_by_shape_and_mode(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_summary_selected_demand_dc_fills")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV_WITH_ONE_SIDED_GAP,
                                encoding="utf-8")

            series = module.load_metric_series(csv_path)
            cases = module.build_stacked_case_series(series)

        self.assertEqual(["64", "128"], [case["shape_label"] for case in cases])
        self.assertEqual(
            {
                "mode": "balanced",
                "segments": [8.54, 0.19, 0.07],
                "total": 8.80,
            },
            cases[0]["bars"][0],
        )
        self.assertEqual(
            {
                "mode": "acc-local-l3",
                "segments": [6.98, 0.13, 0.04],
                "total": 7.15,
            },
            cases[1]["bars"][1],
        )

    def test_compute_bar_layout_prefers_tighter_spacing(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_summary_selected_demand_dc_fills")

        layout = module.compute_bar_layout(1200, 4)

        self.assertEqual(
            {
                "cell_width": 300.0,
                "group_width": 210.0,
                "gap": 14.7,
                "bar_width": 97.65,
            },
            layout,
        )

    def test_make_plot_title_uses_workload_and_batch_dir(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_summary_selected_demand_dc_fills")

        summary_csv = Path(
            "test_results/P3_AttnOnly/Qwen3-30B-A3B/qhead_32_kvhead_16/"
            "NPS1_TP2/prefill-like/global-fixed/batch_4/summary.csv")

        title = module.make_plot_title(summary_csv)

        self.assertEqual(
            "Prefill-like batch_4: selected demand DC fills",
            title,
        )


if __name__ == "__main__":
    unittest.main()
