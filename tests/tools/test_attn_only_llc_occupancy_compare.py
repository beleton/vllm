import json
import tempfile
import unittest
from pathlib import Path


MB = 1_000_000


def _make_samples(scale_a: float, scale_b: float) -> list[tuple[int, str, int]]:
    samples: list[tuple[int, str, int]] = []
    for ts_ms, scale in ((100, scale_a), (200, scale_b)):
        for idx in range(16):
            value = int(scale * (idx + 1) * MB)
            samples.append((ts_ms, f"mon_L3_{idx:02d}", value))
    return samples


def _write_case(root: Path,
                *,
                q_len: int,
                kv_len: int,
                mode: str,
                mask: str,
                slowest_rank_mean_ms: float,
                samples: list[tuple[int, str, int]]) -> Path:
    case_dir = root / f"q{q_len}_kv{kv_len}" / mode / mask / "span1"
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
        "head_plan": {
            "partition_mode": "global-fixed",
        },
        "attn_locality_mode": mode,
        "attn_locality_group_span": 1,
        "slowest_rank_mean_ms": slowest_rank_mean_ms,
    }
    (case_dir / "occupancy_summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with (case_dir / "llc_occupancy.csv").open("w",
                                               newline="",
                                               encoding="utf-8") as f:
        f.write("ts_ms,domain,llc_occupancy_bytes\n")
        for ts_ms, domain, value in samples:
            f.write(f"{ts_ms},{domain},{value}\n")
    return case_dir


class TestAttnOnlyLlcOccupancyCompare(unittest.TestCase):

    def test_write_html_creates_single_page_with_sections(self):
        module = __import__("tools.attn_only.llc_occupancy_compare",
                            fromlist=["write_html"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "occupancy"
            for q_len, base_runtime, acc_runtime in ((1024, 10.0, 11.0),
                                                     (4096, 20.0, 22.0)):
                for mode, runtime in (("balanced", base_runtime),
                                      ("acc-local-l3", acc_runtime)):
                    _write_case(
                        root,
                        q_len=q_len,
                        kv_len=q_len,
                        mode=mode,
                        mask="ffff",
                        slowest_rank_mean_ms=runtime,
                        samples=_make_samples(1.0, 2.0),
                    )
                    _write_case(
                        root,
                        q_len=q_len,
                        kv_len=q_len,
                        mode=mode,
                        mask="0001",
                        slowest_rank_mean_ms=runtime + 0.5,
                        samples=_make_samples(0.5, 1.0),
                    )

            output_path = module.write_html(root)
            html = output_path.read_text(encoding="utf-8")

        self.assertEqual(root / "occupancy_compare.html", output_path)
        self.assertIn("<html", html)
        self.assertIn("Total runtime compare", html)
        self.assertIn("Total occupancy across lengths", html)
        self.assertIn("q1024_kv1024", html)
        self.assertIn("q4096_kv4096", html)
        self.assertIn("ffff", html)
        self.assertIn("0001", html)
        self.assertIn("balanced mean", html)
        self.assertIn("acc-local-l3 peak", html)
        self.assertIn("slowest_rank_mean_ms", html)
        self.assertIn("L3_00", html)
        self.assertIn("Total", html)
        self.assertGreaterEqual(html.count("<svg"), 7)

    def test_write_html_embeds_length_runtime_comparison(self):
        module = __import__("tools.attn_only.llc_occupancy_compare",
                            fromlist=["write_html"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "occupancy"
            for q_len, runtime in ((8192, 30.0), (16384, 31.0)):
                for mode in ("balanced", "acc-local-l3"):
                    for mask in ("ffff", "0001"):
                        _write_case(
                            root,
                            q_len=q_len,
                            kv_len=q_len,
                            mode=mode,
                            mask=mask,
                            slowest_rank_mean_ms=runtime + (0.1 if mode == "acc-local-l3" else 0.0) + (0.5 if mask == "0001" else 0.0),
                            samples=_make_samples(1.0, 2.0),
                        )

            output_path = module.write_html(root)
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("q8192_kv8192", html)
        self.assertIn("q16384_kv16384", html)
        self.assertIn("balanced-ffff", html)
        self.assertIn("balanced-0001", html)
        self.assertIn("acc-local-l3-ffff", html)
        self.assertIn("acc-local-l3-0001", html)

    def test_write_html_splits_runtime_chart_by_length_range(self):
        module = __import__("tools.attn_only.llc_occupancy_compare",
                            fromlist=["write_html"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "occupancy"
            for q_len in (1024, 4096, 8192, 16384, 32768, 65536):
                for mode in ("balanced", "acc-local-l3"):
                    for mask in ("ffff", "0001"):
                        _write_case(
                            root,
                            q_len=q_len,
                            kv_len=q_len,
                            mode=mode,
                            mask=mask,
                            slowest_rank_mean_ms=float(q_len),
                            samples=_make_samples(1.0, 2.0),
                        )

            output_path = module.write_html(root)
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("slowest_rank_mean_ms q1024-q8192", html)
        self.assertIn("slowest_rank_mean_ms q16384-q65536", html)
        short_chart = html.split("slowest_rank_mean_ms q1024-q8192", 1)[1].split(
            "slowest_rank_mean_ms q16384-q65536", 1)[0]
        long_chart = html.split("slowest_rank_mean_ms q16384-q65536", 1)[1].split(
            "Total occupancy across lengths", 1)[0]
        self.assertIn(">1024<", short_chart)
        self.assertIn(">4096<", short_chart)
        self.assertIn(">8192<", short_chart)
        self.assertNotIn(">16384<", short_chart)
        self.assertIn(">16384<", long_chart)
        self.assertIn(">32768<", long_chart)
        self.assertIn(">65536<", long_chart)
        self.assertNotIn(">8192<", long_chart)

    def test_write_html_runtime_tooltip_uses_ms_unit(self):
        module = __import__("tools.attn_only.llc_occupancy_compare",
                            fromlist=["build_html", "CaseOccupancy", "DomainOccupancy", "CaseKey"])

        case_index = {
            module.CaseKey(q_len=1024, mode="balanced", mask="ffff"):
                module.CaseOccupancy(
                    workload="prefill-like",
                    batch_size=1,
                    q_len=1024,
                    kv_len=1024,
                    partition_mode="global-fixed",
                    mode="balanced",
                    mask="ffff",
                    group_span=1,
                    slowest_rank_mean_ms=12.345,
                    case_dir=Path("/tmp/q1024_kv1024/balanced/ffff/span1"),
                    domains=[module.DomainOccupancy(label=f"L3_{idx:02d}",
                                                   mean_mb=1.0,
                                                   peak_mb=2.0)
                             for idx in range(16)],
                    total_mean_mb=3.0,
                    total_peak_mb=4.0,
                )
        }

        html = module.build_html(case_index)

        self.assertIn("balanced-ffff 1024: 12.345 ms", html)
        self.assertNotIn("balanced-ffff 1024: 12.345 MB", html)


if __name__ == "__main__":
    unittest.main()
