import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path


def _write_dry_run(case_dir: Path,
                   *,
                   q_len: int,
                   kv_len: int,
                   mode: str,
                   slowest_rank_mean_ms: float) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "workload": "prefill-like",
        "batch_size": 1,
        "requested_lengths": {
            "q_len": q_len,
            "kv_len": kv_len,
        },
        "resolved_lengths": {
            "q_len": q_len,
            "kv_len": kv_len,
        },
        "attn_locality_mode": mode,
        "attn_locality_group_span": 1,
        "head_plan": {
            "partition_mode": "global-fixed",
        },
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
    }
    (case_dir / "dry_run_summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


class TestP3AttnOnlyCatCompare(unittest.TestCase):

    def test_collect_rows_aligns_baseline_and_cat_masks(self):
        module = importlib.import_module("tools.p3_attn_only.cat_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "span1"
            _write_dry_run(
                root / "q1024_kv1024" / "balanced",
                q_len=1024,
                kv_len=1024,
                mode="balanced",
                slowest_rank_mean_ms=10.0,
            )
            _write_dry_run(
                root / "q1024_kv1024" / "acc-local-l3",
                q_len=1024,
                kv_len=1024,
                mode="acc-local-l3",
                slowest_rank_mean_ms=12.0,
            )
            _write_dry_run(
                root / "cat" / "q1024_kv1024" / "balanced" / "ffff",
                q_len=1024,
                kv_len=1024,
                mode="balanced",
                slowest_rank_mean_ms=10.2,
            )
            _write_dry_run(
                root / "cat" / "q1024_kv1024" / "balanced" / "0001",
                q_len=1024,
                kv_len=1024,
                mode="balanced",
                slowest_rank_mean_ms=10.8,
            )
            _write_dry_run(
                root / "cat" / "q1024_kv1024" / "acc-local-l3" / "ffff",
                q_len=1024,
                kv_len=1024,
                mode="acc-local-l3",
                slowest_rank_mean_ms=12.6,
            )
            _write_dry_run(
                root / "cat" / "q1024_kv1024" / "acc-local-l3" / "0001",
                q_len=1024,
                kv_len=1024,
                mode="acc-local-l3",
                slowest_rank_mean_ms=13.2,
            )

            rows = module.collect_rows(root)

        self.assertEqual(2, len(rows))
        self.assertEqual("balanced", rows[0]["mode"])
        self.assertEqual(10.0, rows[0]["baseline_slowest_rank_mean_ms"])
        self.assertEqual(10.2, rows[0]["cat_ffff_slowest_rank_mean_ms"])
        self.assertEqual(10.8, rows[0]["cat_0001_slowest_rank_mean_ms"])
        self.assertAlmostEqual(2.0, rows[0]["cat_ffff_delta_pct"])
        self.assertAlmostEqual(8.0, rows[0]["cat_0001_delta_pct"])

        self.assertEqual("acc-local-l3", rows[1]["mode"])
        self.assertEqual(12.0, rows[1]["baseline_slowest_rank_mean_ms"])
        self.assertEqual(12.6, rows[1]["cat_ffff_slowest_rank_mean_ms"])
        self.assertEqual(13.2, rows[1]["cat_0001_slowest_rank_mean_ms"])
        self.assertAlmostEqual(5.0, rows[1]["cat_ffff_delta_pct"])
        self.assertAlmostEqual(10.0, rows[1]["cat_0001_delta_pct"])

    def test_write_summary_creates_csv_in_root_by_default(self):
        module = importlib.import_module("tools.p3_attn_only.cat_compare")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "span1"
            _write_dry_run(
                root / "q4096_kv4096" / "balanced",
                q_len=4096,
                kv_len=4096,
                mode="balanced",
                slowest_rank_mean_ms=20.0,
            )
            _write_dry_run(
                root / "cat" / "q4096_kv4096" / "balanced" / "ffff",
                q_len=4096,
                kv_len=4096,
                mode="balanced",
                slowest_rank_mean_ms=20.4,
            )
            _write_dry_run(
                root / "cat" / "q4096_kv4096" / "balanced" / "0001",
                q_len=4096,
                kv_len=4096,
                mode="balanced",
                slowest_rank_mean_ms=21.0,
            )

            output_path = module.write_summary(root)

            self.assertEqual(root / "cat_compare_summary.csv", output_path)
            self.assertTrue(output_path.exists())
            with output_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(1, len(rows))
        self.assertEqual("4096", rows[0]["q_len"])
        self.assertEqual("balanced", rows[0]["mode"])
        self.assertEqual("20.000000", rows[0]["baseline_slowest_rank_mean_ms"])
        self.assertEqual("20.400000", rows[0]["cat_ffff_slowest_rank_mean_ms"])
        self.assertEqual("2.000000", rows[0]["cat_ffff_delta_pct"])
        self.assertEqual("21.000000", rows[0]["cat_0001_slowest_rank_mean_ms"])
        self.assertEqual("5.000000", rows[0]["cat_0001_delta_pct"])


if __name__ == "__main__":
    unittest.main()
