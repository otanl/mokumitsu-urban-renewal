"""Check physical-grid sensitivity for one representative Mokumitsu district."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from houdini_xlb import BACKEND_SIGNATURE, XlbConfig, analyze_heightmap
from scipy.ndimage import zoom

from mokumitsu import Thresholds, generate_mokumitsu

ROOT = Path(__file__).resolve().parents[1]


def _config(level: int, args: argparse.Namespace) -> XlbConfig:
    dx = 100.0 / level
    steps = round(level / args.wind * args.flow_throughs)
    return XlbConfig(
        grid_x=level,
        grid_y=level,
        grid_z=round(args.domain_height_m / dx),
        steps=steps,
        wind=args.wind,
        reynolds=args.reynolds,
        domain_length_x_m=100.0,
        domain_length_y_m=100.0,
        domain_height_m=args.domain_height_m,
        reference_height_m=args.reference_height_m,
        pedestrian_height_m=args.pedestrian_height_m,
        average_window=max(1, round(steps * args.average_fraction)),
        average_every=max(1, round(steps * args.average_fraction / 20)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--levels", default="128,160,200")
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--flow-throughs", type=float, default=1.0)
    parser.add_argument("--average-fraction", type=float, default=0.25)
    parser.add_argument("--domain-height-m", type=float, default=60.0)
    parser.add_argument("--reference-height-m", type=float, default=10.0)
    parser.add_argument("--pedestrian-height-m", type=float, default=1.5)
    parser.add_argument("--wind", type=float, default=0.05)
    parser.add_argument("--reynolds", type=float, default=8000.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "artifacts" / "houdini" / "cache" / "residential-grid-check",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "residential_grid_independence.json",
    )
    args = parser.parse_args()
    levels = tuple(int(item) for item in args.levels.split(","))
    if (
        len(levels) < 2
        or any(level < 32 for level in levels)
        or tuple(sorted(set(levels))) != levels
    ):
        raise ValueError("provide at least two strictly increasing unique lattice levels >= 32")
    if (
        args.res < 16
        or args.flow_throughs <= 0
        or not 0 < args.average_fraction <= 1
        or args.tolerance <= 0
    ):
        raise ValueError("res, flow-throughs, average-fraction and tolerance must be positive")

    district = generate_mokumitsu(args.seed)
    heightmap = district.heightmap(args.res, args.domain_height_m)
    outdoor = heightmap <= 1e-7
    threshold = Thresholds().dead
    rows = []
    fields = []
    for level in levels:
        config = _config(level, args)
        xlb_heightmap = district.heightmap(level, args.domain_height_m)
        empty = analyze_heightmap(
            np.zeros_like(xlb_heightmap),
            config,
            cache_dir=args.cache,
        )
        result = analyze_heightmap(xlb_heightmap, config, cache_dir=args.cache)
        margin = max(1, round(5.0 / config.cell_sizes_m[0]))
        u0 = float(empty.speed[margin:-margin, margin:-margin].mean())
        if not np.isfinite(u0) or u0 <= 0:
            raise RuntimeError("empty-domain XLB reference speed is not positive and finite")
        ratio_lattice = result.speed / u0
        ratio = zoom(
            ratio_lattice,
            (args.res / config.grid_y, args.res / config.grid_x),
            order=1,
        ).astype(np.float32)
        if ratio.shape != heightmap.shape or not np.isfinite(ratio).all():
            raise RuntimeError("grid-comparison resampling produced an invalid field")
        ratio[~outdoor] = 0.0
        values = ratio[outdoor]
        fields.append(ratio)
        row = {
            "grid_xyz": list(config.grid_xyz),
            "cell_sizes_m": list(config.cell_sizes_m),
            "xlb_config": config.to_dict(),
            "steps": config.steps,
            "resolved_pedestrian_height_m": config.resolved_pedestrian_height_m,
            "empty_reference_speed": u0,
            "mean_index": float(values.mean()),
            "dead_fraction": float(np.mean(values < threshold)),
            "peak_speed_ratio": float(result.speed.max() / config.wind),
            "building_coverage": float(np.mean(xlb_heightmap > 1e-7)),
            "cache_hit": bool(result.cache_hit and empty.cache_hit),
        }
        rows.append(row)
        print(
            f"grid={config.grid_xyz} steps={config.steps} "
            f"z={config.resolved_pedestrian_height_m:.3f} "
            f"mean={row['mean_index']:.4f} dead={row['dead_fraction']:.4f}",
            flush=True,
        )

    finest = fields[-1]
    finest_mean = max(float(finest[outdoor].mean()), 1e-12)
    for row, field in zip(rows, fields, strict=True):
        row["rms_vs_finest"] = float(
            np.sqrt(np.mean((field[outdoor] - finest[outdoor]) ** 2)) / finest_mean
        )
    means = np.asarray([row["mean_index"] for row in rows])
    dead = np.asarray([row["dead_fraction"] for row in rows])
    report = {
        "protocol_version": 1,
        "backend_signature": BACKEND_SIGNATURE,
        "seed": args.seed,
        "levels": list(levels),
        "comparison_grid": [args.res, args.res],
        "flow_throughs": args.flow_throughs,
        "average_fraction": args.average_fraction,
        "dead_threshold_u_over_u0": threshold,
        "domain_height_m": args.domain_height_m,
        "height_encoding": "fraction_of_domain_height",
        "rows": rows,
        "mean_index_drift": float(means.max() - means.min()),
        "dead_fraction_drift": float(dead.max() - dead.min()),
        "tolerance": args.tolerance,
    }
    report["passed"] = bool(
        report["mean_index_drift"] <= args.tolerance
        and report["dead_fraction_drift"] <= args.tolerance
    )
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"mean drift={report['mean_index_drift']:.4f}; "
        f"dead drift={report['dead_fraction_drift']:.4f}; saved={output}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
