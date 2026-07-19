"""Verify an FNO-selected mokumitsu joint-renewal plan with GPU XLB.

The fast residential FNO selects parcel pairs and massing placements.  This script
reconstructs that deterministic plan, freezes the before/after height maps, and runs
both through the realistic XLB solver.  An empty-domain XLB run supplies XLB's own
reference speed, so weak/strong-wind fractions are not compared across incompatible
absolute velocity scales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

from houdini_xlb import XlbConfig, analyze_heightmap  # noqa: E402

from mokumitsu import Thresholds, generate_mokumitsu  # noqa: E402
from mokumitsu.joint_renewal import (  # noqa: E402
    JointRenewalPolicy,
    optimize_joint_renewal,
)
from mokumitsu.model import load_model  # noqa: E402
from mokumitsu.wind import (  # noqa: E402
    SummerWindScenario,
    WindDirection,
    district_wind_masks,
)


def _evaluation_frame(
    district,
    res: int,
    direction_deg: float,
    edge_distance_m: float,
    domain_height_m: float,
):
    turns = int(round(direction_deg / 90.0)) % 4
    heightmap = np.rot90(
        district.heightmap(res, domain_height_m),
        k=turns,
    ).copy()
    masks = {
        name: np.rot90(mask, k=turns).copy()
        for name, mask in district_wind_masks(
            district,
            res,
            building_edge_distance_m=edge_distance_m,
            domain_height_m=domain_height_m,
        ).items()
    }
    return _safe_heightmap(heightmap), masks


def _safe_heightmap(heightmap: np.ndarray) -> np.ndarray:
    """Keep solids off XLB's boundary-condition cells after lattice upsampling."""
    heightmap = np.asarray(heightmap, dtype=np.float32).copy()
    heightmap[:2, :] = 0.0
    heightmap[-2:, :] = 0.0
    heightmap[:, :2] = 0.0
    heightmap[:, -2:] = 0.0
    return heightmap


def _xlb_heightmap(district, config: XlbConfig, direction_deg: float) -> np.ndarray:
    if config.grid_x != config.grid_y:
        raise ValueError("cardinal Mokumitsu verification requires a square XLB plan grid")
    turns = int(round(direction_deg / 90.0)) % 4
    heightmap = district.heightmap(config.grid_x, config.domain_height_m)
    return _safe_heightmap(np.rot90(heightmap, k=turns).copy())


def _display_speed(speed: np.ndarray, display_heightmap: np.ndarray) -> np.ndarray:
    output = zoom(
        speed,
        (
            display_heightmap.shape[0] / speed.shape[0],
            display_heightmap.shape[1] / speed.shape[1],
        ),
        order=1,
    ).astype(np.float32)
    output[display_heightmap > 1e-7] = 0.0
    return output


def _run_xlb(heightmap: np.ndarray, config: XlbConfig, cache_dir: Path):
    return analyze_heightmap(
        heightmap,
        config,
        cache_dir=cache_dir,
    ).speed


def _predict_fno(model, heightmap: np.ndarray) -> np.ndarray:
    speed = np.maximum(np.asarray(model.predict(heightmap), dtype=np.float32), 0.0)
    speed[heightmap > 1e-7] = 0.0
    return speed


def _zone_metrics(
    speed: np.ndarray,
    heightmap: np.ndarray,
    masks: dict[str, np.ndarray],
    reference_speed: float,
    thresholds: Thresholds,
) -> dict[str, dict[str, float | int]]:
    out = {}
    for name, region in masks.items():
        mask = np.asarray(region, dtype=bool) & (heightmap <= 1e-7)
        values = speed[mask]
        if values.size == 0:
            out[name] = {
                "cells": 0,
                "mean_index": 0.0,
                "dead_fraction": 0.0,
                "strong_fraction": 0.0,
                "comfortable_fraction": 0.0,
            }
            continue
        low = thresholds.dead * reference_speed
        high = thresholds.strong * reference_speed
        out[name] = {
            "cells": int(values.size),
            "mean_index": float(values.mean() / reference_speed),
            "dead_fraction": float(np.mean(values < low)),
            "strong_fraction": float(np.mean(values > high)),
            "comfortable_fraction": float(np.mean((values >= low) & (values <= high))),
        }
    return out


def _normalized_rms(
    fno_speed: np.ndarray,
    xlb_speed: np.ndarray,
    heightmap: np.ndarray,
    fno_u0: float,
    xlb_u0: float,
) -> float:
    mask = heightmap <= 1e-7
    delta = fno_speed[mask] / fno_u0 - xlb_speed[mask] / xlb_u0
    return float(np.sqrt(np.mean(delta**2)))


def _metric_changes(initial: dict, final: dict) -> dict[str, dict[str, float]]:
    fields = ("mean_index", "dead_fraction", "strong_fraction")
    return {
        zone: {field: float(final[zone][field] - initial[zone][field]) for field in fields}
        for zone in initial
    }


def _plot(
    heightmaps: tuple[np.ndarray, np.ndarray],
    fno_fields: tuple[np.ndarray, np.ndarray],
    xlb_fields: tuple[np.ndarray, np.ndarray],
    references: tuple[float, float],
    metrics: dict,
    path: Path,
) -> Path:
    hm0, hm1 = heightmaps
    f0, f1 = (fno_fields[0] / references[0], fno_fields[1] / references[0])
    x0, x1 = (xlb_fields[0] / references[1], xlb_fields[1] / references[1])
    fdelta = f1 - f0
    xdelta = x1 - x0
    delta_limit = max(
        0.10,
        float(np.nanpercentile(np.abs(np.concatenate((fdelta.ravel(), xdelta.ravel()))), 97)),
    )

    figure, axes = plt.subplots(2, 3, figsize=(13.6, 8.4), constrained_layout=True)
    engines = (
        ("FNO screening", f0, f1, fdelta, metrics["fno"]),
        ("XLB verification", x0, x1, xdelta, metrics["xlb"]),
    )
    for row, (engine, initial, final, delta, engine_metrics) in enumerate(engines):
        for col, (field, title, hm) in enumerate(((initial, "Before", hm0), (final, "After", hm1))):
            image = axes[row, col].imshow(
                field,
                origin="lower",
                cmap="turbo",
                vmin=0.0,
                vmax=1.5,
            )
            axes[row, col].contour(hm > 1e-7, levels=[0.5], colors="white", linewidths=0.45)
            yard = engine_metrics[title.lower()]["parcel_open_space"]["dead_fraction"]
            axes[row, col].set_title(f"{engine}: {title}\nyard weak = {yard:.3f}")
            figure.colorbar(image, ax=axes[row, col], fraction=0.046, label="U / U0")

        image = axes[row, 2].imshow(
            delta,
            origin="lower",
            cmap="RdBu_r",
            vmin=-delta_limit,
            vmax=delta_limit,
        )
        axes[row, 2].contour(hm1 > 1e-7, levels=[0.5], colors="black", linewidths=0.45)
        yard_delta = engine_metrics["change"]["parcel_open_space"]["dead_fraction"]
        axes[row, 2].set_title(f"{engine}: After - Before\nΔ yard weak = {yard_delta:+.3f}")
        figure.colorbar(image, ax=axes[row, 2], fraction=0.046, label="Δ(U / U0)")

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Mokumitsu joint renewal — FNO selection checked on frozen geometry with XLB\n"
        "Wind is evaluated toward +X; white/black contours are building footprints."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projects", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--target-coverage", type=float, default=0.30)
    parser.add_argument("--minimum-floors", type=int, default=3)
    parser.add_argument("--maximum-floors", type=int, default=6)
    parser.add_argument("--year", type=int, default=2036)
    parser.add_argument(
        "--wind-dir",
        type=float,
        choices=(0.0, 90.0, 180.0, 270.0),
        default=0.0,
    )
    parser.add_argument("--wind-model", default="residential")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument(
        "--gridxy",
        type=int,
        default=0,
        help="XLB horizontal lattice; 0 uses twice the FNO resolution",
    )
    parser.add_argument(
        "--gridz",
        type=int,
        default=0,
        help="vertical lattice; 0 derives near-cubic cells from the physical domain",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/houdini/cache/mokumitsu-joint-verify"),
    )
    parser.add_argument("--out", default="outputs/joint_renewal_xlb.json")
    parser.add_argument("--fig", default="outputs/joint_renewal_xlb.png")
    args = parser.parse_args()

    district = generate_mokumitsu(seed=args.seed)
    model = load_model(args.wind_model)
    scenario = SummerWindScenario(
        directions=(WindDirection(args.wind_dir),),
        model_name=args.wind_model,
    )
    policy = JointRenewalPolicy(
        year=args.year,
        projects=args.projects,
        candidate_limit=args.candidate_limit,
        target_building_coverage=args.target_coverage,
        minimum_floors=args.minimum_floors,
        maximum_floors=args.maximum_floors,
    )
    plan = optimize_joint_renewal(district, policy=policy, scenario=scenario, model=model)

    res = int(model.ny)
    if model.nx != res:
        raise ValueError("joint-renewal XLB verification requires a square surrogate grid")
    gridxy = args.gridxy or 2 * res
    dx = district.width_m / gridxy
    grid_y = round(district.height_m / dx)
    grid_z = args.gridz or round(scenario.domain_height_m / dx)
    xlb_config = XlbConfig(
        grid_x=gridxy,
        grid_y=grid_y,
        grid_z=grid_z,
        steps=args.steps,
        domain_length_x_m=district.width_m,
        domain_length_y_m=district.height_m,
        domain_height_m=scenario.domain_height_m,
        reference_height_m=scenario.reference_height_m,
        pedestrian_height_m=scenario.pedestrian_height_m,
        average_window=min(800, args.steps),
        average_every=100,
    )
    hm0, masks0 = _evaluation_frame(
        district,
        res,
        args.wind_dir,
        scenario.building_edge_distance_m,
        scenario.domain_height_m,
    )
    hm1, masks1 = _evaluation_frame(
        plan.district,
        res,
        args.wind_dir,
        scenario.building_edge_distance_m,
        scenario.domain_height_m,
    )

    fno_u0 = float(model.reference_speed())
    fno_fields = (_predict_fno(model, hm0), _predict_fno(model, hm1))
    print(
        f"selected {len(plan.steps)} projects with {args.wind_model} FNO; "
        f"running empty/before/after XLB on {xlb_config.grid_xyz} "
        f"for {args.steps} steps at z={xlb_config.resolved_pedestrian_height_m:.2f} m",
        flush=True,
    )
    empty_xlb = _run_xlb(
        np.zeros((xlb_config.grid_y, xlb_config.grid_x), np.float32),
        xlb_config,
        args.cache,
    )
    margin = max(1, round(5.0 / xlb_config.cell_sizes_m[0]))
    xlb_u0 = float(empty_xlb[margin:-margin, margin:-margin].mean())
    xlb_fields = (
        _display_speed(
            _run_xlb(
                _xlb_heightmap(district, xlb_config, args.wind_dir),
                xlb_config,
                args.cache,
            ),
            hm0,
        ),
        _display_speed(
            _run_xlb(
                _xlb_heightmap(plan.district, xlb_config, args.wind_dir),
                xlb_config,
                args.cache,
            ),
            hm1,
        ),
    )

    thresholds = scenario.thresholds
    fno_before = _zone_metrics(fno_fields[0], hm0, masks0, fno_u0, thresholds)
    fno_after = _zone_metrics(fno_fields[1], hm1, masks1, fno_u0, thresholds)
    xlb_before = _zone_metrics(xlb_fields[0], hm0, masks0, xlb_u0, thresholds)
    xlb_after = _zone_metrics(xlb_fields[1], hm1, masks1, xlb_u0, thresholds)
    metrics = {
        "fno": {
            "before": fno_before,
            "after": fno_after,
            "change": _metric_changes(fno_before, fno_after),
        },
        "xlb": {
            "before": xlb_before,
            "after": xlb_after,
            "change": _metric_changes(xlb_before, xlb_after),
        },
    }
    agreement = {
        zone: bool(
            np.sign(metrics["fno"]["change"][zone]["dead_fraction"])
            == np.sign(metrics["xlb"]["change"][zone]["dead_fraction"])
        )
        for zone in ("all_outdoor", "roads", "parcel_open_space")
    }
    result = {
        "seed": args.seed,
        "wind_direction_deg": args.wind_dir,
        "wind_model": args.wind_model,
        "model": model.provenance(),
        "resolution": res,
        "xlb_config": xlb_config.to_dict(),
        "xlb_resolved_pedestrian_height_m": xlb_config.resolved_pedestrian_height_m,
        "xlb_steps": args.steps,
        "reference_speed": {"fno": fno_u0, "xlb": xlb_u0},
        "plan": plan.to_dict(include_district=False),
        "metrics": metrics,
        "dead_fraction_direction_agreement": agreement,
        "normalized_field_rms": {
            "before": _normalized_rms(fno_fields[0], xlb_fields[0], hm0, fno_u0, xlb_u0),
            "after": _normalized_rms(fno_fields[1], xlb_fields[1], hm1, fno_u0, xlb_u0),
        },
    }

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    figure = _plot(
        (hm0, hm1),
        fno_fields,
        xlb_fields,
        (fno_u0, xlb_u0),
        metrics,
        ROOT / args.fig,
    )
    for zone in ("all_outdoor", "roads", "parcel_open_space"):
        fno_delta = metrics["fno"]["change"][zone]["dead_fraction"]
        xlb_delta = metrics["xlb"]["change"][zone]["dead_fraction"]
        print(
            f"{zone}: weak-wind Δ FNO={fno_delta:+.4f}, XLB={xlb_delta:+.4f}, "
            f"direction_agrees={agreement[zone]}"
        )
    print(f"saved {output}")
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
