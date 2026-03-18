"""Benchmark: daft-h3 extension vs UDF wrapping h3 Python library.

Run: python bench_h3.py
"""
from __future__ import annotations

import time

import h3
import numpy as np

import daft
import daft_h3
from daft import col
from daft.session import Session


# --- UDF definitions (what users write today) ---

@daft.func.batch(return_dtype=daft.DataType.string())
def udf_latlng_to_cell(lat, lng):
    return [h3.latlng_to_cell(la, ln, 7) for la, ln in zip(lat.to_pylist(), lng.to_pylist())]


@daft.func.batch(return_dtype=daft.DataType.float64())
def udf_cell_to_lat(hex_str):
    return [h3.cell_to_latlng(c)[0] if c is not None else None for c in hex_str.to_pylist()]


@daft.func.batch(return_dtype=daft.DataType.string())
def udf_cell_to_parent(hex_str):
    return [h3.cell_to_parent(c, 5) if c is not None else None for c in hex_str.to_pylist()]


@daft.func.batch(return_dtype=daft.DataType.int64())
def udf_str_to_cell(hex_str):
    return [h3.str_to_int(s) if s is not None else None for s in hex_str.to_pylist()]


@daft.func.batch(return_dtype=daft.DataType.int32())
def udf_resolution(hex_str):
    return [h3.get_resolution(c) if c is not None else None for c in hex_str.to_pylist()]


# --- Benchmark harness ---

def gen_data(n: int) -> dict:
    rng = np.random.default_rng(42)
    return {
        "lat": rng.uniform(-90, 90, n).tolist(),
        "lng": rng.uniform(-180, 180, n).tolist(),
    }


def bench(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def fmt(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f}us"
    return f"{ms:.1f}ms"


def run_comparison(label, native_fn, udf_fn):
    native_ms = bench(native_fn) * 1000
    udf_ms = bench(udf_fn) * 1000
    speedup = udf_ms / native_ms
    print(f"  {label:<22} ext: {fmt(native_ms):>8}   udf: {fmt(udf_ms):>8}   speedup: {speedup:.1f}x")


def main():
    sess = Session()
    sess.load_extension(daft_h3)

    for n in [10_000, 100_000, 1_000_000]:
        print(f"\n{'='*70}")
        print(f"  {n:,} rows (all operations on hex string columns)")
        print(f"{'='*70}")
        data = gen_data(n)

        with sess:
            df = daft.from_pydict(data)

            # --- latlng_to_cell ---
            run_comparison(
                "latlng_to_cell",
                lambda: df.select(daft_h3.h3_latlng_to_cell(col("lat"), col("lng"), 7)).collect(),
                lambda: df.select(udf_latlng_to_cell(col("lat"), col("lng"))).collect(),
            )

            # Build hex string column for downstream benchmarks
            hex_df = (
                df.select(daft_h3.h3_latlng_to_cell(col("lat"), col("lng"), 7).alias("cell"))
                .collect()
                .select(daft_h3.h3_cell_to_str(col("cell")).alias("hex"))
                .collect()
            )

            # --- cell_to_lat (string in) ---
            run_comparison(
                "cell_to_lat (str in)",
                lambda: hex_df.select(daft_h3.h3_cell_to_lat(col("hex"))).collect(),
                lambda: hex_df.select(udf_cell_to_lat(col("hex"))).collect(),
            )

            # --- cell_parent (string in -> string out) ---
            run_comparison(
                "cell_parent (str->str)",
                lambda: hex_df.select(daft_h3.h3_cell_parent(col("hex"), 5)).collect(),
                lambda: hex_df.select(udf_cell_to_parent(col("hex"))).collect(),
            )

            # --- resolution (string in) ---
            run_comparison(
                "cell_resolution (str)",
                lambda: hex_df.select(daft_h3.h3_cell_resolution(col("hex"))).collect(),
                lambda: hex_df.select(udf_resolution(col("hex"))).collect(),
            )

            # --- str_to_cell ---
            run_comparison(
                "str_to_cell",
                lambda: hex_df.select(daft_h3.h3_str_to_cell(col("hex"))).collect(),
                lambda: hex_df.select(udf_str_to_cell(col("hex"))).collect(),
            )


if __name__ == "__main__":
    main()
