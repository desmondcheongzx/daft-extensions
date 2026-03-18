"""Benchmark: extension FFI overhead isolation.

Compares the extension against a no-op baseline to measure the raw FFI
round-trip cost (Series -> Arrow C Data Interface -> arrow-rs -> back).

Run from the h3 extension directory:
  .venv/bin/python bench_ffi_overhead.py
"""
from __future__ import annotations

import time

import numpy as np

import daft
import daft_h3
from daft import col
from daft.session import Session


def gen_hex_df(n: int):
    """Generate a DataFrame with hex string H3 cells."""
    rng = np.random.default_rng(42)
    lats = rng.uniform(-90, 90, n).tolist()
    lngs = rng.uniform(-180, 180, n).tolist()

    sess = Session()
    sess.load_extension(daft_h3)
    with sess:
        df = daft.from_pydict({"lat": lats, "lng": lngs})
        cell_df = df.select(
            daft_h3.h3_latlng_to_cell(col("lat"), col("lng"), 7).alias("cell")
        ).collect()
        hex_df = cell_df.select(
            daft_h3.h3_cell_to_str(col("cell")).alias("hex")
        ).collect()
    return cell_df, hex_df


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    p10 = sorted(times)[len(times) // 10]
    return med * 1000, p10 * 1000


def fmt(ms):
    return f"{ms:.1f}ms"


def main():
    sess = Session()
    sess.load_extension(daft_h3)

    for n in [100_000, 1_000_000, 10_000_000]:
        print(f"\n{'='*60}")
        print(f"  {n:,} rows")
        print(f"{'='*60}")

        cell_df, hex_df = gen_hex_df(n)

        with sess:
            # Cheap operations that isolate FFI overhead
            # (resolution is just a bit shift - nearly zero compute)
            med, p10 = bench(
                lambda: cell_df.select(daft_h3.h3_cell_resolution(col("cell"))).collect()
            )
            throughput = n / (med / 1000) / 1e6
            print(f"  cell_resolution (u64)  median: {fmt(med):>8}  p10: {fmt(p10):>8}  {throughput:.1f}M rows/s")

            med, p10 = bench(
                lambda: hex_df.select(daft_h3.h3_cell_resolution(col("hex"))).collect()
            )
            throughput = n / (med / 1000) / 1e6
            print(f"  cell_resolution (str)  median: {fmt(med):>8}  p10: {fmt(p10):>8}  {throughput:.1f}M rows/s")

            med, p10 = bench(
                lambda: cell_df.select(daft_h3.h3_cell_is_valid(col("cell"))).collect()
            )
            throughput = n / (med / 1000) / 1e6
            print(f"  cell_is_valid (u64)    median: {fmt(med):>8}  p10: {fmt(p10):>8}  {throughput:.1f}M rows/s")

            # Heavier operations
            med, p10 = bench(
                lambda: hex_df.select(daft_h3.h3_cell_to_lat(col("hex"))).collect()
            )
            throughput = n / (med / 1000) / 1e6
            print(f"  cell_to_lat (str)      median: {fmt(med):>8}  p10: {fmt(p10):>8}  {throughput:.1f}M rows/s")

            med, p10 = bench(
                lambda: hex_df.select(daft_h3.h3_str_to_cell(col("hex"))).collect()
            )
            throughput = n / (med / 1000) / 1e6
            print(f"  str_to_cell            median: {fmt(med):>8}  p10: {fmt(p10):>8}  {throughput:.1f}M rows/s")


if __name__ == "__main__":
    main()
