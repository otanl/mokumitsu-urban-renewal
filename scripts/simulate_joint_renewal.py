"""Screen joint parcel renewal with floor-area transfer, fire risk, and summer wind."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

from mokumitsu import generate_mokumitsu  # noqa: E402
from mokumitsu.fire import FireScenario, simulate_fire_spread  # noqa: E402
from mokumitsu.joint_renewal import (  # noqa: E402
    JointRenewalPolicy,
    optimize_joint_renewal,
)
from mokumitsu.model import load_model  # noqa: E402
from mokumitsu.wind import (  # noqa: E402
    SummerWindScenario,
    WindDirection,
    predict_directional_wind,
)


def _comparison(initial, final, initial_fire, final_fire, plan):
    initial_wind = plan.initial_wind.weighted_zones
    final_wind = plan.final_wind.weighted_zones
    return {
        "projects_completed": len(plan.steps),
        "floor_area_retention": final.gross_floor_area_m2 / max(initial.gross_floor_area_m2, 1e-9),
        "footprint_coverage_delta": final.footprint_coverage - initial.footprint_coverage,
        "yard_dead_fraction_delta": final_wind["parcel_open_space"].dead_fraction
        - initial_wind["parcel_open_space"].dead_fraction,
        "outdoor_dead_fraction_delta": final_wind["all_outdoor"].dead_fraction
        - initial_wind["all_outdoor"].dead_fraction,
        "edge_strong_fraction_delta": final_wind["building_edge"].strong_fraction
        - initial_wind["building_edge"].strong_fraction,
        "expected_burned_buildings_delta": final_fire.expected_burned_buildings
        - initial_fire.expected_burned_buildings,
        "expected_burned_area_m2_delta": final_fire.expected_burned_area_m2
        - initial_fire.expected_burned_area_m2,
    }


def _world_field(district, direction, model, reference_speed):
    speed, heightmap, _, _ = predict_directional_wind(
        district,
        direction_deg=direction,
        model=model,
        reference_speed=reference_speed,
    )
    turns = int(round(direction / 90.0)) % 4
    return (
        np.rot90(speed, k=-turns).copy() / reference_speed,
        np.rot90(heightmap, k=-turns).copy() * max(district.width_m, district.height_m),
    )


def _draw_joint_outlines(ax, district):
    for parcel in district.parcels:
        if parcel.kind != "joint_redevelopment":
            continue
        coordinates = parcel.polygon + (parcel.polygon[0],)
        ax.plot(
            [point[0] for point in coordinates],
            [point[1] for point in coordinates],
            color="#f4d35e",
            linewidth=2.2,
        )
    joint_parcels = {
        parcel.id for parcel in district.parcels if parcel.kind == "joint_redevelopment"
    }
    for building in district.buildings:
        if building.parcel_id not in joint_parcels:
            continue
        coordinates = building.corners + (building.corners[0],)
        ax.plot(
            [point[0] for point in coordinates],
            [point[1] for point in coordinates],
            color="#e63946",
            linewidth=2.0,
        )


def _relative_change(final, initial):
    return (final / max(initial, 1e-9) - 1.0) * 100.0


def _plot(initial_district, plan, initial_fire, final_fire, model, path, direction):
    final_district = plan.district
    reference_speed = plan.initial_wind.reference_speed
    initial_speed, initial_height = _world_field(
        initial_district, direction, model, reference_speed
    )
    final_speed, final_height = _world_field(final_district, direction, model, reference_speed)
    extent = (0, initial_district.width_m, 0, initial_district.height_m)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))

    height_max = max(float(initial_height.max()), float(final_height.max()), 1.0)
    for axis, heightmap, title in (
        (axes[0, 0], initial_height, "Initial massing"),
        (axes[0, 1], final_height, "Joint-renewal massing"),
    ):
        image = axis.imshow(
            heightmap,
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=0,
            vmax=height_max,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
    _draw_joint_outlines(axes[0, 1], final_district)
    figure.colorbar(image, ax=axes[0, :2], label="building height (m)", shrink=0.75)

    speed_image = None
    for axis, speed, title in (
        (axes[1, 0], initial_speed, "Initial summer wind"),
        (axes[1, 1], final_speed, "After joint renewal"),
    ):
        speed_image = axis.imshow(
            speed,
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=0,
            vmax=1.5,
        )
        axis.contour(
            speed,
            levels=(0.30, 1.30),
            origin="lower",
            extent=extent,
            colors=("#14213d", "#d00000"),
            linewidths=0.7,
        )
        axis.set_title(title + " (U/U0)")
        axis.set_aspect("equal")
    _draw_joint_outlines(axes[1, 1], final_district)
    figure.colorbar(speed_image, ax=axes[1, :2], label="U/U0", shrink=0.75)

    initial_summary = initial_district.summary()
    final_summary = final_district.summary()
    initial_zones = plan.initial_wind.weighted_zones
    final_zones = plan.final_wind.weighted_zones
    labels = (
        "Footprint",
        "Yard weak",
        "Outdoor weak",
        "Edge strong",
        "Burned bldgs",
        "Burned area",
    )
    changes = (
        _relative_change(final_summary.footprint_coverage, initial_summary.footprint_coverage),
        _relative_change(
            final_zones["parcel_open_space"].dead_fraction,
            initial_zones["parcel_open_space"].dead_fraction,
        ),
        _relative_change(
            final_zones["all_outdoor"].dead_fraction,
            initial_zones["all_outdoor"].dead_fraction,
        ),
        _relative_change(
            final_zones["building_edge"].strong_fraction,
            initial_zones["building_edge"].strong_fraction,
        ),
        _relative_change(
            final_fire.expected_burned_buildings,
            initial_fire.expected_burned_buildings,
        ),
        _relative_change(
            final_fire.expected_burned_area_m2,
            initial_fire.expected_burned_area_m2,
        ),
    )
    colors = ["#2a9d8f" if value <= 0 else "#e76f51" for value in changes]
    axes[0, 2].barh(labels, changes, color=colors)
    axes[0, 2].axvline(0, color="black", linewidth=0.8)
    axes[0, 2].set_xlabel("relative change (%)")
    axes[0, 2].set_title("Negative is generally better")
    axes[0, 2].grid(axis="x", alpha=0.2)

    axes[1, 2].axis("off")
    rows = [
        ("Projects", f"{len(plan.steps)}"),
        (
            "Parcels / buildings",
            f"{len(initial_district.parcels)} to {len(final_district.parcels)}",
        ),
        (
            "FAR",
            f"{initial_summary.gross_floor_area_ratio:.3f} to "
            f"{final_summary.gross_floor_area_ratio:.3f}",
        ),
        (
            "Footprint coverage",
            f"{initial_summary.footprint_coverage:.3f} to {final_summary.footprint_coverage:.3f}",
        ),
        (
            "Yard weak fraction",
            f"{initial_zones['parcel_open_space'].dead_fraction:.3f} to "
            f"{final_zones['parcel_open_space'].dead_fraction:.3f}",
        ),
        (
            "Edge strong fraction",
            f"{initial_zones['building_edge'].strong_fraction:.3f} to "
            f"{final_zones['building_edge'].strong_fraction:.3f}",
        ),
        (
            "Expected burned",
            f"{initial_fire.expected_burned_buildings:.2f} to "
            f"{final_fire.expected_burned_buildings:.2f}",
        ),
    ]
    axes[1, 2].table(
        cellText=rows,
        colLabels=("Metric", "Initial to final"),
        loc="center",
        cellLoc="left",
    )
    axes[1, 2].set_title("Joint-renewal screen")

    for axis in axes[:, :2].flat:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.suptitle(
        "Joint parcel renewal: retain floor area, reduce footprint, reconnect open space"
    )
    figure.subplots_adjust(wspace=0.28, hspace=0.26, top=0.92)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
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
    parser.add_argument("--wind-dir", type=float, choices=(0.0, 90.0, 180.0, 270.0), default=0.0)
    parser.add_argument("--wind-model", default="residential")
    parser.add_argument("--fire-wind-speed", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=80)
    parser.add_argument("--out", default="outputs/joint_renewal.json")
    parser.add_argument("--fig", default="outputs/joint_renewal.png")
    parser.add_argument("--include-district", action="store_true")
    args = parser.parse_args()

    district = generate_mokumitsu(args.seed)
    policy = JointRenewalPolicy(
        year=args.year,
        projects=args.projects,
        candidate_limit=args.candidate_limit,
        target_building_coverage=args.target_coverage,
        minimum_floors=args.minimum_floors,
        maximum_floors=args.maximum_floors,
    )
    summer = SummerWindScenario(
        directions=(WindDirection(args.wind_dir),),
        model_name=args.wind_model,
    )
    model = load_model(args.wind_model)
    plan = optimize_joint_renewal(district, policy=policy, scenario=summer, model=model)
    fire_scenario = FireScenario(
        runs=args.runs,
        seed=args.seed,
        wind_direction_deg=args.wind_dir,
        wind_speed_m_s=args.fire_wind_speed,
    )
    initial_fire = simulate_fire_spread(district, fire_scenario)
    final_fire = simulate_fire_spread(plan.district, fire_scenario)
    initial_summary = district.summary()
    final_summary = plan.district.summary()
    data = plan.to_dict(include_district=args.include_district)
    data["model"] = model.provenance()
    data["initial_summary"] = asdict(initial_summary)
    data["initial_fire"] = initial_fire.to_dict()
    data["final_fire"] = final_fire.to_dict()
    data["comparison"] = _comparison(
        initial_summary,
        final_summary,
        initial_fire,
        final_fire,
        plan,
    )
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    figure = _plot(
        district,
        plan,
        initial_fire,
        final_fire,
        model,
        ROOT / args.fig,
        args.wind_dir,
    )

    initial_zones = plan.initial_wind.weighted_zones
    final_zones = plan.final_wind.weighted_zones
    print(f"saved {output}")
    print(f"saved {figure}")
    print(
        f"projects={len(plan.steps)} FAR={initial_summary.gross_floor_area_ratio:.3f}->"
        f"{final_summary.gross_floor_area_ratio:.3f} footprint="
        f"{initial_summary.footprint_coverage:.3f}->{final_summary.footprint_coverage:.3f} "
        f"yard_weak={initial_zones['parcel_open_space'].dead_fraction:.3f}->"
        f"{final_zones['parcel_open_space'].dead_fraction:.3f} edge_strong="
        f"{initial_zones['building_edge'].strong_fraction:.3f}->"
        f"{final_zones['building_edge'].strong_fraction:.3f} expected_burned="
        f"{initial_fire.expected_burned_buildings:.2f}->{final_fire.expected_burned_buildings:.2f}"
    )
    for step in plan.steps:
        print(
            f"step={step.index} parcels={','.join(step.candidate.parcel_ids)} "
            f"placement={step.placement.variant} floors={step.placement.floors} "
            f"wind_objective={step.wind_objective_before:.4f}->"
            f"{step.wind_objective_after:.4f}"
        )


if __name__ == "__main__":
    main()
