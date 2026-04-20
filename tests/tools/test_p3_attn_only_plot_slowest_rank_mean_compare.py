import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SUMMARY_CSV = """metric,q64_kv64_balanced,q64_kv64_acc-local-l3,q128_kv128_balanced,q128_kv128_acc-local-l3,q256_kv256_balanced,q256_kv256_acc-local-l3
slowest rank mean (ms),0.027242,0.031338,0.029414,0.031510,0.055479,0.062016
slowest rank mean delta (%),-,15.04,-,7.13,-,11.78
"""

SUMMARY_CSV_WITH_SPAN = """metric,q64_kv64_balanced_span4,q64_kv64_acc-local-l3_span4,q128_kv128_balanced_span4,q128_kv128_acc-local-l3_span4
slowest rank mean (ms),0.074821,0.067005,0.191925,0.170473
slowest rank mean delta (%),-,-10.45,-,-11.18
"""


class TestPlotSlowestRankMeanCompare(unittest.TestCase):

    def test_load_slowest_rank_mean_series_reads_two_modes(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_slowest_rank_mean_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV, encoding="utf-8")

            series = module.load_slowest_rank_mean_series(csv_path)

        self.assertEqual(
            [(64, 64, 0.027242), (128, 128, 0.029414), (256, 256, 0.055479)],
            series["balanced"],
        )
        self.assertEqual(
            [(64, 64, 0.031338), (128, 128, 0.03151), (256, 256, 0.062016)],
            series["acc-local-l3"],
        )

    def test_generate_plot_writes_default_png_next_to_summary_csv(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_slowest_rank_mean_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV, encoding="utf-8")

            output_path = module.generate_plot(csv_path)

            self.assertEqual(
                csv_path.with_name(
                    "slowest_rank_mean_balanced_vs_acc-local-l3.png"),
                output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_generate_plot_uses_grouped_bars_instead_of_lines(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_slowest_rank_mean_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV, encoding="utf-8")
            output_path = Path(tmpdir) / "plot.png"
            fig_mock = mock.Mock()
            ax_mock = mock.Mock()

            with mock.patch.object(module.plt, "subplots",
                                   return_value=(fig_mock, ax_mock)), \
                 mock.patch.object(module.plt, "close"):
                module.generate_plot(csv_path, output_path)

        self.assertEqual(2, ax_mock.bar.call_count)
        self.assertFalse(ax_mock.plot.called)

    def test_load_slowest_rank_mean_series_accepts_mode_suffix(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_slowest_rank_mean_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            csv_path.write_text(SUMMARY_CSV_WITH_SPAN, encoding="utf-8")

            series = module.load_slowest_rank_mean_series(csv_path)

        self.assertEqual(
            [(64, 64, 0.074821), (128, 128, 0.191925)],
            series["balanced"],
        )
        self.assertEqual(
            [(64, 64, 0.067005), (128, 128, 0.170473)],
            series["acc-local-l3"],
        )

    def test_make_plot_title_uses_workload_and_batch_dir(self):
        module = importlib.import_module(
            "tools.p3_attn_only.plot_slowest_rank_mean_compare")

        summary_csv = Path(
            "test_results/P3_AttnOnly/Qwen3-30B-A3B/qhead_32_kvhead_4/"
            "NPS1_TP2/prefill-like/global-fixed/batch_16_old_acc/summary.csv")

        title = module.make_plot_title(summary_csv)

        self.assertEqual(
            "Prefill-like batch_16_old_acc: slowest rank mean (ms)",
            title,
        )


if __name__ == "__main__":
    unittest.main()
