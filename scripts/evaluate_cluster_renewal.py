"""Evaluate connected 2-4 parcel joint-renewal alternatives on a Pareto front."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Polygon as PolygonPatch

ROOT = Path(__file__).resolve().parent.parent

from mokumitsu import generate_mokumitsu  # noqa: E402
from mokumitsu.fire import FireScenario  # noqa: E402
from mokumitsu.joint_renewal import JointRenewalPolicy  # noqa: E402
from mokumitsu.model import load_model  # noqa: E402
from mokumitsu.pareto import (  # noqa: E402
    ParetoRenewalPolicy,
    evaluate_joint_renewal_pareto,
)
from mokumitsu.wind import SummerWindScenario, WindDirection  # noqa: E402


def _parse_wind_rose(spec: str) -> tuple[WindDirection, ...]:
    directions = []
    for item in spec.split(","):
        direction, weight = item.split(":", 1)
        directions.append(WindDirection(float(direction), float(weight)))
    if not directions:
        raise ValueError("wind rose must contain direction:weight pairs")
    return tuple(directions)


def _building_polygon(building) -> np.ndarray:
    ct, st = math.cos(building.theta), math.sin(building.theta)
    corners = []
    for x, y in (
        (-building.width_m / 2, -building.depth_m / 2),
        (building.width_m / 2, -building.depth_m / 2),
        (building.width_m / 2, building.depth_m / 2),
        (-building.width_m / 2, building.depth_m / 2),
    ):
        corners.append((building.cx + ct * x - st * y, building.cy + st * x + ct * y))
    return np.asarray(corners)


def _draw_district(axis, district, title: str, open_spaces=()) -> None:
    for road in district.roads:
        axis.plot(
            (road.start[0], road.end[0]),
            (road.start[1], road.end[1]),
            color="#d1d5db",
            linewidth=max(0.7, road.width_m * 0.75),
            solid_capstyle="round",
            zorder=0,
        )
    space_colours = {
        "road_widening": "#94a3b8",
        "ventilation_corridor": "#22d3ee",
        "shared_court": "#86efac",
        "pocket_park": "#22c55e",
    }
    space_labels = {
        "road_widening": "local road widening",
        "ventilation_corridor": "wind-aligned corridor",
        "shared_court": "shared court",
        "pocket_park": "pocket park",
    }
    for space in open_spaces:
        axis.add_patch(
            PolygonPatch(
                np.asarray(space.polygon),
                closed=True,
                facecolor=space_colours.get(space.kind, "#a7f3d0"),
                edgecolor="#0f766e",
                linewidth=0.8,
                alpha=0.58,
                zorder=1,
            )
        )
    present_space_kinds = tuple(dict.fromkeys(space.kind for space in open_spaces))
    if present_space_kinds:
        axis.legend(
            handles=[
                Patch(
                    facecolor=space_colours.get(kind, "#a7f3d0"),
                    edgecolor="#0f766e",
                    label=space_labels.get(kind, kind),
                    alpha=0.58,
                )
                for kind in present_space_kinds
            ],
            loc="lower left",
            fontsize=7,
            framealpha=0.88,
        )
    for parcel in district.parcels:
        if parcel.kind != "joint_redevelopment":
            continue
        coordinates = np.asarray(parcel.polygon + (parcel.polygon[0],))
        axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color="#f59e0b",
            linewidth=2.0,
            zorder=3,
        )
    for building in district.buildings:
        renewed = building.id.startswith("joint_building_")
        axis.add_patch(
            PolygonPatch(
                _building_polygon(building),
                closed=True,
                facecolor="#dc2626" if renewed else "#374151",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.92,
                zorder=2,
            )
        )
    axis.set(
        xlim=(0, district.width_m),
        ylim=(0, district.height_m),
        aspect="equal",
        title=title,
        xlabel="x [m]",
        ylabel="y [m]",
    )


def _plot(result, path: Path) -> Path:
    recommended = result.recommended
    if recommended is None:
        raise RuntimeError("no feasible cluster-renewal alternative was generated")
    baseline = result.baseline_objectives
    alternatives = result.alternatives
    front = set(result.pareto_indices)
    wind_improvement = np.asarray(
        [
            100
            * (baseline.wind_penalty - item.objectives.wind_penalty)
            / max(baseline.wind_penalty, 1e-9)
            for item in alternatives
        ]
    )
    fire_improvement = np.asarray(
        [
            100
            * (baseline.expected_burned_area_m2 - item.objectives.expected_burned_area_m2)
            / max(baseline.expected_burned_area_m2, 1e-9)
            for item in alternatives
        ]
    )
    footprint_release = np.asarray(
        [
            (baseline.footprint_coverage - item.objectives.footprint_coverage)
            * result.baseline_district.width_m
            * result.baseline_district.height_m
            for item in alternatives
        ]
    )
    rights = np.asarray([item.objectives.rights_count for item in alternatives])
    access = np.asarray([item.objectives.access_poor_resolved for item in alternatives])
    connected_open_space = np.asarray(
        [item.objectives.connected_open_space_m2 for item in alternatives]
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.6), constrained_layout=True)
    _draw_district(axes[0, 0], result.baseline_district, "Before")
    _draw_district(
        axes[0, 1],
        recommended.district,
        (
            f"Balanced Pareto option #{recommended.index}: "
            f"{recommended.candidate.parcel_count} parcels, "
            f"{recommended.placement.floors} floors, "
            f"{recommended.objectives.connected_open_space_m2:.0f} m² connected open space"
        ),
        recommended.placement.open_spaces,
    )

    colors = plt.get_cmap("viridis")((rights - 2) / 2)
    for index in range(len(alternatives)):
        is_front = index in front
        axes[1, 0].scatter(
            wind_improvement[index],
            fire_improvement[index],
            s=55 + 45 * access[index],
            color=colors[index],
            alpha=0.95 if is_front else 0.25,
            edgecolor="black" if is_front else "none",
            linewidth=1.0,
        )
        if is_front:
            axes[1, 0].annotate(str(index), (wind_improvement[index], fire_improvement[index]))
    axes[1, 0].scatter(
        wind_improvement[recommended.index],
        fire_improvement[recommended.index],
        marker="*",
        s=260,
        color="#fbbf24",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
        label="balanced recommendation",
    )
    axes[1, 0].axvline(0, color="#9ca3af", linewidth=0.8)
    axes[1, 0].axhline(0, color="#9ca3af", linewidth=0.8)
    axes[1, 0].set(
        title="Wind–fire trade-off (outlined = Pareto front)",
        xlabel="wind penalty reduction [%]",
        ylabel="expected burned area reduction [%]",
    )
    axes[1, 0].legend(loc="best")

    color_min = float(footprint_release.min())
    color_max = float(footprint_release.max())
    if color_min == color_max:
        color_max = color_min + 1.0
    color_norm = matplotlib.colors.Normalize(vmin=color_min, vmax=color_max)
    color_map = plt.get_cmap("RdYlGn")
    for index in range(len(alternatives)):
        is_front = index in front
        axes[1, 1].scatter(
            rights[index],
            connected_open_space[index],
            s=60 + 45 * access[index],
            color=color_map(color_norm(footprint_release[index])),
            edgecolor="black" if is_front else "none",
            alpha=0.95 if is_front else 0.28,
        )
    axes[1, 1].scatter(
        recommended.objectives.rights_count,
        connected_open_space[recommended.index],
        marker="*",
        s=260,
        color="#fbbf24",
        edgecolor="black",
        zorder=5,
    )
    axes[1, 1].set(
        title="Road-connected open space vs rights burden",
        xlabel="parcels / rights involved",
        ylabel="connected corridor + court / park [m²]",
        xticks=(2, 3, 4),
    )
    color_scale = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    color_scale.set_array(footprint_release)
    figure.colorbar(color_scale, ax=axes[1, 1], label="released building footprint [m²]")
    figure.suptitle(
        "Connected mokumitsu renewal alternatives\n"
        "Marker size = access-poor parcels resolved; colour/outline do not replace project review"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-parcels", type=int, default=2)
    parser.add_argument("--maximum-parcels", type=int, default=4)
    parser.add_argument("--candidates-per-size", type=int, default=4)
    parser.add_argument("--placement-grid", type=int, default=7)
    parser.add_argument("--placement-variants", type=int, default=3)
    parser.add_argument("--target-coverage", type=float, default=0.30)
    parser.add_argument("--minimum-floors", type=int, default=3)
    parser.add_argument("--maximum-floors", type=int, default=6)
    parser.add_argument("--target-road-width", type=float, default=6.0)
    parser.add_argument("--corridor-width", type=float, default=2.0)
    parser.add_argument("--open-space-fraction", type=float, default=0.10)
    parser.add_argument("--year", type=int, default=2036)
    parser.add_argument("--summer-wind-rose", default="0:0.7,90:0.3")
    parser.add_argument("--wind-model", default="residential")
    parser.add_argument("--fire-wind-dir", type=float, default=0.0)
    parser.add_argument("--fire-wind-speed", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--out", default="outputs/cluster_renewal_pareto.json")
    parser.add_argument("--fig", default="outputs/cluster_renewal_pareto.png")
    parser.add_argument("--include-recommended-district", action="store_true")
    args = parser.parse_args()

    district = generate_mokumitsu(seed=args.seed)
    joint_policy = JointRenewalPolicy(
        year=args.year,
        minimum_cluster_parcels=args.minimum_parcels,
        maximum_cluster_parcels=args.maximum_parcels,
        target_building_coverage=args.target_coverage,
        minimum_floors=args.minimum_floors,
        maximum_floors=args.maximum_floors,
        target_road_width_m=args.target_road_width,
        ventilation_corridor_width_m=args.corridor_width,
        shared_open_space_fraction=args.open_space_fraction,
        placement_grid=args.placement_grid,
        placement_variants=args.placement_variants,
    )
    pareto_policy = ParetoRenewalPolicy(
        candidates_per_cluster_size=args.candidates_per_size,
    )
    wind_scenario = SummerWindScenario(
        directions=_parse_wind_rose(args.summer_wind_rose),
        model_name=args.wind_model,
    )
    fire_scenario = FireScenario(
        runs=args.runs,
        seed=args.seed,
        wind_direction_deg=args.fire_wind_dir,
        wind_speed_m_s=args.fire_wind_speed,
    )
    model = load_model(args.wind_model)
    result = evaluate_joint_renewal_pareto(
        district,
        joint_policy=joint_policy,
        pareto_policy=pareto_policy,
        wind_scenario=wind_scenario,
        fire_scenario=fire_scenario,
        model=model,
    )
    payload = result.to_dict(
        include_recommended_district=args.include_recommended_district,
    )
    payload["model"] = model.provenance()
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    figure = _plot(result, ROOT / args.fig)
    recommended = result.recommended
    if recommended is None:
        print("no feasible alternatives")
    else:
        baseline = result.baseline_objectives
        objective = recommended.objectives
        print(
            f"alternatives={len(result.alternatives)} pareto={len(result.pareto_indices)} "
            f"recommended={recommended.index} parcels={objective.rights_count} "
            f"access_resolved={objective.access_poor_resolved} "
            f"wind={baseline.wind_penalty:.4f}->{objective.wind_penalty:.4f} "
            f"burned_area={baseline.expected_burned_area_m2:.1f}"
            f"->{objective.expected_burned_area_m2:.1f}m2 "
            f"footprint={baseline.footprint_coverage:.3f}"
            f"->{objective.footprint_coverage:.3f} "
            f"connected_open_space={objective.connected_open_space_m2:.1f}m2 "
            f"road_dedication={recommended.placement.road_widening_area_m2:.1f}m2"
        )
    print(f"saved {output}")
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
