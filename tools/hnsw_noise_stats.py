#!/usr/bin/env python3
"""Calibrate query noise for synthetic HNSW embedding workloads.

The script does not modify HNSW experiment code. It imports the shared
``data_gen.py`` module, samples base vectors, adds Gaussian query noise, and
reports cosine-distance statistics between each base vector and its noisy query.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def parse_noise_stds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def cosine_distance_stats(base, noise_std, seed, chunk_size):
    rng = np.random.default_rng(seed)
    n, dim = base.shape
    distances = np.empty(n, dtype=np.float32)
    noise_norms = np.empty(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        rows = base[start:end]
        noise = rng.standard_normal((end - start, dim), dtype=np.float32) * noise_std
        noisy = rows + noise

        row_norm = np.linalg.norm(rows, axis=1)
        noisy_norm = np.linalg.norm(noisy, axis=1)
        dot = np.einsum("ij,ij->i", rows, noisy)
        cosine = dot / np.maximum(row_norm * noisy_norm, 1e-12)
        distances[start:end] = 1.0 - np.clip(cosine, -1.0, 1.0)
        noise_norms[start:end] = np.linalg.norm(noise, axis=1)

    return summarize(distances), summarize(noise_norms)


def choose_recommendation(rows, target_center, target_min, target_max):
    in_range = [
        row for row in rows
        if target_min <= row["cosine_distance"]["p50"] <= target_max
    ]
    candidates = in_range if in_range else rows
    return min(
        candidates,
        key=lambda row: abs(row["cosine_distance"]["p50"] - target_center),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-gen-dir", default="/home/zjj/HNSW/scripts")
    parser.add_argument("--num-elements", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-mode", default="realistic",
                        choices=["uniform", "clustered", "realistic"])
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--noise-stds", default="0.05,0.1,0.2,0.3,0.5,1.0",
                        help="Comma-separated noise std values")
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--target-min", type=float, default=0.05)
    parser.add_argument("--target-max", type=float, default=0.15)
    parser.add_argument("--target-center", type=float, default=0.10)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    sys.path.insert(0, args.data_gen_dir)
    from data_gen import make_data

    data, _ = make_data(
        args.num_elements,
        args.dim,
        args.seed,
        mode=args.data_mode,
        num_clusters=args.num_clusters,
    )

    base_norm = summarize(np.linalg.norm(data, axis=1))
    rows = []
    for noise_std in parse_noise_stds(args.noise_stds):
        distance, noise_norm = cosine_distance_stats(
            data,
            noise_std,
            args.seed + 2000,
            args.chunk_size,
        )
        rows.append({
            "noise_std": noise_std,
            "cosine_distance": distance,
            "noise_norm": noise_norm,
        })

    recommendation = choose_recommendation(
        rows,
        args.target_center,
        args.target_min,
        args.target_max,
    )

    payload = {
        "config": {
            "num_elements": args.num_elements,
            "dim": args.dim,
            "seed": args.seed,
            "data_mode": args.data_mode,
            "num_clusters": args.num_clusters,
            "target_min": args.target_min,
            "target_max": args.target_max,
            "target_center": args.target_center,
        },
        "base_norm": base_norm,
        "rows": rows,
        "recommendation": recommendation,
    }

    print("base_norm mean={mean:.4f} p50={p50:.4f} p95={p95:.4f}".format(**base_norm))
    print("noise_std  cos_dist_mean  p50     p90     p95     p99     noise_norm_mean")
    for row in rows:
        d = row["cosine_distance"]
        nn = row["noise_norm"]
        print(
            f"{row['noise_std']:>9.4f}  "
            f"{d['mean']:>13.5f}  "
            f"{d['p50']:>6.5f}  "
            f"{d['p90']:>6.5f}  "
            f"{d['p95']:>6.5f}  "
            f"{d['p99']:>6.5f}  "
            f"{nn['mean']:>15.4f}"
        )
    print(
        "recommended noise_std={:.4f} "
        "(p50 cosine_distance={:.5f})".format(
            recommendation["noise_std"],
            recommendation["cosine_distance"]["p50"],
        )
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "noise_std",
                "cosine_distance_mean",
                "cosine_distance_p50",
                "cosine_distance_p90",
                "cosine_distance_p95",
                "cosine_distance_p99",
                "noise_norm_mean",
            ])
            for row in rows:
                d = row["cosine_distance"]
                nn = row["noise_norm"]
                writer.writerow([
                    row["noise_std"],
                    d["mean"],
                    d["p50"],
                    d["p90"],
                    d["p95"],
                    d["p99"],
                    nn["mean"],
                ])


if __name__ == "__main__":
    main()
