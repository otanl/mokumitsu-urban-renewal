"""Verify a selected connected-parcel renewal alternative with GPU XLB.

The Pareto search is deliberately performed with the residential FNO.  This
script reads the frozen recommended district from that search, reconstructs the
seeded baseline, and compares both districts with XLB for every 90-degree wind
direction in the saved summer wind rose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

from verify_joint_renewal_xlb import (  # noqa: E402
    _evaluation_frame,
    _metric_changes,
    _normalized_rms,
    _predict_fno,
    _run_xlb,
    _zone_metrics,
)

from mokumitsu import MokumitsuDistrict, Thresholds, generate_mokumitsu  # noqa: E402
from mokumitsu.model import load_model  # noqa: E402
from mokumitsu.wind import SummerWindScenario  # noqa: E402


def _load_selection(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    recommended_index = int(data["recommended_index"])
    recommended = data["alternatives"][recommended_index]
    if "district" not in recommended:
        raise ValueError(
            "selection JSON does not contain the recommended district; "
            "rerun evaluate_cluster_renewal.py with --include-recommended-district"
        )
    district = MokumitsuDistrict.from_dict(recommended["district"])
    directions = tuple(
        (float(item["direction_deg"]), float(item["weight"]))
        for item in data["baseline"]["wind"]["directions"]
    )
    if not directions or sum(weight for _, weight in directions) <= 0:
        raise ValueError("selection JSON contains no positive wind-rose weight")
    for direction, weight in directions:
        if weight < 0 or not np.isclose(direction % 90.0, 0.0):
            raise ValueError("XLB verification currently accepts non-negative 90-degree wind bins")
    total_weight = sum(weight for _, weight in directions)
    directions = tuple((direction, weight / total_weight) for direction, weight in directions)
    return data, recommended, district, directions


def _weighted_zone_metrics(
    metrics: list[dict[str, dict[str, float | int]]],
    weights: tuple[float, ...],
) -> dict[str, dict[str, float | int]]:
    fields = ("mean_index", "dead_fraction", "strong_fraction", "comfortable_fraction")
    return {
        zone: {
            "cells": int(metrics[0][zone]["cells"]),
            **{
                field: float(
                    sum(
                        weight * float(item[zone][field])
                        for item, weight in zip(metrics, weights, strict=True)
                    )
                )
                for field in fields
            },
        }
        for zone in metrics[0]
    }


def _wind_penalty(zones: dict[str, dict[str, float | int]]) -> float:
    return float(
        0.55 * float(zones["parcel_open_space"]["dead_fraction"])
        + 0.25 * float(zones["all_outdoor"]["dead_fraction"])
        + 0.10 * float(zones["roads"]["dead_fraction"])
        + 0.35 * float(zones["building_edge"]["strong_fraction"])
        + 0.15 * float(zones["all_outdoor"]["strong_fraction"])
    )


def _reduction_percent(before: float, after: float) -> float:
    return float(100.0 * (before - after) / before) if before else 0.0


def _plot(
    rows: list[dict],
    xlb_u0: float,
    weighted_penalties: dict[str, dict[str, float]],
    path: Path,
) -> Path:
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13.2, 4.0 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, row in enumerate(rows):
        before = row["xlb_fields"][0] / xlb_u0
        after = row["xlb_fields"][1] / xlb_u0
        delta = after - before
        delta_limit = max(float(np.quantile(np.abs(delta), 0.99)), 0.05)
        image0 = axes[row_index, 0].imshow(before, origin="lower", cmap="magma", vmin=0, vmax=1.6)
        axes[row_index, 1].imshow(after, origin="lower", cmap="magma", vmin=0, vmax=1.6)
        image2 = axes[row_index, 2].imshow(
            delta,
            origin="lower",
            cmap="coolwarm",
            vmin=-delta_limit,
            vmax=delta_limit,
        )
        direction = row["direction_deg"]
        weight = row["weight"]
        axes[row_index, 0].set_title(f"{direction:g}° ({weight:.0%}) · XLB before")
        axes[row_index, 1].set_title(f"{direction:g}° ({weight:.0%}) · XLB after")
        axes[row_index, 2].set_title("after − before, normalized speed")
        for axis in axes[row_index]:
            axis.set_xticks(())
            axis.set_yticks(())
        figure.colorbar(
            image0, ax=axes[row_index, :2], shrink=0.78, label="speed / empty-domain speed"
        )
        figure.colorbar(image2, ax=axes[row_index, 2], shrink=0.78)

    fno = weighted_penalties["fno"]
    xlb = weighted_penalties["xlb"]
    figure.suptitle(
        "Connected-parcel renewal: wind-rose XLB verification\n"
        "weighted wind-penalty reduction · "
        f"FNO {_reduction_percent(fno['before'], fno['after']):.2f}% "
        f"/ XLB {_reduction_percent(xlb['before'], xlb['after']):.2f}%"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=145)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="outputs/cluster_renewal_pareto.json")
    parser.add_argument("--wind-model", default="")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument(
        "--gridxy",
        type=int,
        default=0,
        help="XLB horizontal lattice; 0 uses twice the FNO resolution",
    )
    parser.add_argument("--gridz", type=int, default=64)
    parser.add_argument("--out", default="outputs/cluster_renewal_xlb.json")
    parser.add_argument("--fig", default="outputs/cluster_renewal_xlb.png")
    args = parser.parse_args()

    selection_path = ROOT / args.selection
    selection, recommended, final_district, directions = _load_selection(selection_path)
    initial_district = generate_mokumitsu(seed=final_district.seed)
    model_name = args.wind_model or selection["baseline"]["wind"]["model_name"]
    model = load_model(model_name)
    res = int(model.ny)
    if model.nx != res:
        raise ValueError("cluster-renewal XLB verification requires a square surrogate grid")
    gridxy = args.gridxy or 2 * res
    grid_xyz = (gridxy, gridxy, args.gridz)
    scenario_defaults = SummerWindScenario()
    edge_distance_m = scenario_defaults.building_edge_distance_m
    thresholds: Thresholds = scenario_defaults.thresholds
    fno_u0 = float(model.reference_speed())

    frames = [
        (
            direction,
            weight,
            _evaluation_frame(initial_district, res, direction, edge_distance_m),
            _evaluation_frame(final_district, res, direction, edge_distance_m),
        )
        for direction, weight in directions
    ]
    print(
        f"recommended option {recommended['index']} "
        f"({len(recommended['candidate']['parcel_ids'])} parcels); "
        f"running empty + {2 * len(frames)} before/after XLB cases on {grid_xyz} "
        f"for {args.steps} steps",
        flush=True,
    )
    empty_xlb = _run_xlb(np.zeros((res, res), np.float32), grid_xyz, args.steps)
    xlb_u0 = float(empty_xlb.mean())

    rows = []
    for direction, weight, initial_frame, final_frame in frames:
        hm0, masks0 = initial_frame
        hm1, masks1 = final_frame
        fno_fields = (_predict_fno(model, hm0), _predict_fno(model, hm1))
        xlb_fields = (_run_xlb(hm0, grid_xyz, args.steps), _run_xlb(hm1, grid_xyz, args.steps))
        fno_before = _zone_metrics(fno_fields[0], hm0, masks0, fno_u0, thresholds)
        fno_after = _zone_metrics(fno_fields[1], hm1, masks1, fno_u0, thresholds)
        xlb_before = _zone_metrics(xlb_fields[0], hm0, masks0, xlb_u0, thresholds)
        xlb_after = _zone_metrics(xlb_fields[1], hm1, masks1, xlb_u0, thresholds)
        agreement = {
            zone: bool(
                np.sign(fno_after[zone]["dead_fraction"] - fno_before[zone]["dead_fraction"])
                == np.sign(xlb_after[zone]["dead_fraction"] - xlb_before[zone]["dead_fraction"])
            )
            for zone in ("all_outdoor", "roads", "parcel_open_space")
        }
        rows.append(
            {
                "direction_deg": direction,
                "weight": weight,
                "heightmaps": (hm0, hm1),
                "fno_fields": fno_fields,
                "xlb_fields": xlb_fields,
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
                "dead_fraction_direction_agreement": agreement,
                "normalized_field_rms": {
                    "before": _normalized_rms(fno_fields[0], xlb_fields[0], hm0, fno_u0, xlb_u0),
                    "after": _normalized_rms(fno_fields[1], xlb_fields[1], hm1, fno_u0, xlb_u0),
                },
            }
        )

    weights = tuple(row["weight"] for row in rows)
    weighted = {}
    penalties = {}
    for engine in ("fno", "xlb"):
        before = _weighted_zone_metrics([row[engine]["before"] for row in rows], weights)
        after = _weighted_zone_metrics([row[engine]["after"] for row in rows], weights)
        weighted[engine] = {
            "before": before,
            "after": after,
            "change": _metric_changes(before, after),
        }
        penalties[engine] = {
            "before": _wind_penalty(before),
            "after": _wind_penalty(after),
        }
        penalties[engine]["reduction_percent"] = _reduction_percent(
            penalties[engine]["before"],
            penalties[engine]["after"],
        )

    serial_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"heightmaps", "fno_fields", "xlb_fields"}
        }
        for row in rows
    ]
    result = {
        "selection": {
            "path": str(selection_path),
            "recommended_index": int(recommended["index"]),
            "recommended_id": recommended["id"],
            "parcel_count": len(recommended["candidate"]["parcel_ids"]),
            "floor_area_retention": float(recommended["floor_area_retention"]),
        },
        "wind_model": model_name,
        "model": model.provenance(),
        "resolution": res,
        "xlb_grid_xyz": grid_xyz,
        "xlb_steps": args.steps,
        "building_edge_distance_m": edge_distance_m,
        "reference_speed": {"fno": fno_u0, "xlb": xlb_u0},
        "wind_rose": [
            {"direction_deg": direction, "weight": weight} for direction, weight in directions
        ],
        "per_direction": serial_rows,
        "weighted_metrics": weighted,
        "weighted_wind_penalty": penalties,
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    figure = _plot(rows, xlb_u0, penalties, ROOT / args.fig)
    print(
        "weighted wind-penalty reduction: "
        f"FNO={penalties['fno']['reduction_percent']:+.2f}%, "
        f"XLB={penalties['xlb']['reduction_percent']:+.2f}%"
    )
    for row in rows:
        zone = "parcel_open_space"
        print(
            f"{row['direction_deg']:g}° parcel-open weak-wind Δ: "
            f"FNO={row['fno']['change'][zone]['dead_fraction']:+.4f}, "
            f"XLB={row['xlb']['change'][zone]['dead_fraction']:+.4f}"
        )
    print(f"saved {output}")
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
