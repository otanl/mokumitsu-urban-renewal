from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_SRC))

from mokumitsu import generate_mokumitsu  # noqa: E402
from mokumitsu.feasibility import (  # noqa: E402
    JointRenewalFeasibilityPolicy,
    evaluate_joint_renewal_feasibility,
)
from mokumitsu.joint_renewal import (  # noqa: E402
    JointRenewalPolicy,
    optimize_joint_renewal,
)
from mokumitsu.model import load_model  # noqa: E402
from mokumitsu.wind import (  # noqa: E402
    SummerWindScenario,
    WindDirection,
    weighted_world_wind_ratio,
)


def _block_mean(field: np.ndarray, target_res: int) -> np.ndarray:
    """Area-average a square field for a compact Houdini display payload."""
    source_res = int(field.shape[0])
    if field.shape != (source_res, source_res):
        raise ValueError("wind display field must be square")
    if target_res <= 0 or source_res % target_res:
        raise ValueError(
            f"wind display resolution {target_res} must divide source resolution {source_res}"
        )
    scale = source_res // target_res
    return field.reshape(target_res, scale, target_res, scale).mean(axis=(1, 3))


def _wind_visualization(result, scenario, model, target_res: int) -> dict:
    """Serialize cached, world-oriented FNO fields for baseline plus every phase."""
    districts = (result.initial_district,) + tuple(phase.district for phase in result.phases)
    years = (result.initial_district.reference_year,) + tuple(phase.year for phase in result.phases)
    frames = []
    reference_speed = None
    for phase_index, (district, year) in enumerate(zip(districts, years, strict=True)):
        ratio, heightmap, reference_speed = weighted_world_wind_ratio(
            district,
            scenario=scenario,
            model=model,
            reference_speed=reference_speed,
        )
        ratio_display = _block_mean(ratio, target_res)
        solid_fraction = _block_mean((heightmap > 0).astype(np.float32), target_res)
        outdoor = solid_fraction < 0.25
        frames.append(
            {
                "phase": phase_index,
                "year": int(year),
                "speed_ratio": np.round(ratio_display, 4).tolist(),
                "solid": (~outdoor).tolist(),
                "outdoor_min": round(float(ratio_display[outdoor].min()), 4),
                "outdoor_mean": round(float(ratio_display[outdoor].mean()), 4),
                "outdoor_max": round(float(ratio_display[outdoor].max()), 4),
            }
        )
    total_weight = sum(direction.weight for direction in scenario.directions)
    return {
        "engine": "FNO surrogate",
        "quantity": "wind-rose-weighted scalar speed ratio U/U0",
        "source_resolution": int(model.ny),
        "display_resolution": target_res,
        "reference_speed": float(reference_speed),
        "directions": [
            {
                "direction_deg": float(direction.direction_deg % 360.0),
                "weight": float(direction.weight / total_weight),
            }
            for direction in scenario.directions
        ],
        "thresholds": {
            "weak_below": float(scenario.thresholds.dead),
            "strong_above": float(scenario.thresholds.strong),
        },
        "frames": frames,
    }


def _plot(result, path: Path) -> Path:
    phases = result.phases
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    if not phases:
        for axis in axes.flat:
            axis.text(0.5, 0.5, "No feasible joint project", ha="center", va="center")
            axis.set_axis_off()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    years = np.asarray([phase.year for phase in phases])
    width = max(0.8, result.policy.years_per_phase * 0.55)
    costs = {
        "Construction": np.asarray([phase.cost.construction_jpy for phase in phases]) / 1e6,
        "Demolition + site": np.asarray(
            [
                phase.cost.demolition_jpy
                + phase.cost.road_works_jpy
                + phase.cost.open_space_works_jpy
                for phase in phases
            ]
        )
        / 1e6,
        "Relocation + rights": np.asarray(
            [phase.cost.relocation_jpy + phase.cost.rights_coordination_jpy for phase in phases]
        )
        / 1e6,
        "Design + contingency": np.asarray(
            [phase.cost.design_and_management_jpy + phase.cost.contingency_jpy for phase in phases]
        )
        / 1e6,
    }
    bottom = np.zeros(len(phases))
    for label, values in costs.items():
        axes[0, 0].bar(years, values, width=width, bottom=bottom, label=label)
        bottom += values
    axes[0, 0].set_title("Project cost by phase (scenario assumptions)")
    axes[0, 0].set_ylabel("million JPY")
    axes[0, 0].legend(fontsize=8)

    timeline_years = np.concatenate(([years[0] - result.policy.years_per_phase], years))
    dwellings = np.asarray(
        [result.summary.initial_dwellings] + [phase.district_dwellings_after for phase in phases]
    )
    axes[0, 1].step(timeline_years, dwellings, where="post", marker="o", label="Capacity")
    relocation_axis = axes[0, 1].twinx()
    relocation_axis.bar(
        years,
        [phase.temporary_relocation_dwellings for phase in phases],
        width=width,
        alpha=0.45,
        color="#d95f02",
        label="Temporary relocation",
    )
    axes[0, 1].set_title("Dwelling capacity and temporary relocation")
    axes[0, 1].set_ylabel("district dwelling capacity")
    relocation_axis.set_ylabel("temporary dwellings", color="#d95f02")
    relocation_axis.tick_params(axis="y", labelcolor="#d95f02")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles2, labels2 = relocation_axis.get_legend_handles_labels()
    axes[0, 1].legend(handles + handles2, labels + labels2, fontsize=8)

    axes[1, 0].bar(
        years - width * 0.18,
        [phase.rights_count for phase in phases],
        width=width * 0.36,
        label="Rights converted",
    )
    axes[1, 0].bar(
        years + width * 0.18,
        [phase.replacement_dwellings for phase in phases],
        width=width * 0.36,
        label="Replacement dwellings",
    )
    axes[1, 0].plot(
        years,
        [phase.return_dwellings for phase in phases],
        marker="o",
        color="#1b7837",
        label="Return dwellings",
    )
    axes[1, 0].set_title("Rights conversion and dwelling delivery")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        years,
        [phase.cumulative_cost_jpy / 1e9 for phase in phases],
        marker="o",
        color="#762a83",
        label="Cumulative cost",
    )
    axes[1, 1].set_ylabel("billion JPY", color="#762a83")
    axes[1, 1].tick_params(axis="y", labelcolor="#762a83")
    open_axis = axes[1, 1].twinx()
    open_axis.bar(
        years,
        [phase.connected_open_space_m2 for phase in phases],
        width=width,
        alpha=0.32,
        color="#1b9e77",
        label="Connected open space",
    )
    open_axis.set_ylabel("open space m2", color="#1b9e77")
    open_axis.tick_params(axis="y", labelcolor="#1b9e77")
    axes[1, 1].set_title("Cumulative cost and delivered open space")

    for axis in axes.flat:
        axis.set_xlabel("completion year")
        axis.grid(axis="y", alpha=0.2)
        axis.set_xticks(years)
    figure.suptitle(
        "Phased mokumitsu joint-renewal feasibility screening\n"
        "Costs are configurable scenario inputs, not a market appraisal",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projects", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--year", type=int, default=2036)
    parser.add_argument("--years-per-phase", type=int, default=3)
    parser.add_argument("--construction-months", type=int, default=18)
    parser.add_argument("--replacement-dwelling-area", type=float, default=55.0)
    parser.add_argument("--residential-efficiency", type=float, default=0.82)
    parser.add_argument("--construction-cost", type=float, default=450_000.0)
    parser.add_argument("--demolition-cost", type=float, default=35_000.0)
    parser.add_argument("--relocation-monthly", type=float, default=150_000.0)
    parser.add_argument("--wind-model", default="residential")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--allow-model-fallback", action="store_true")
    parser.add_argument("--max-wind-increase", type=float, default=0.01)
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument(
        "--out",
        default="outputs/joint_feasibility.json",
    )
    parser.add_argument(
        "--fig",
        default="outputs/joint_feasibility.png",
    )
    parser.add_argument("--include-districts", action="store_true")
    parser.add_argument("--wind-display-res", type=int, default=32)
    args = parser.parse_args()

    district = generate_mokumitsu(seed=args.seed)
    joint_policy = JointRenewalPolicy(
        year=args.year,
        projects=args.projects,
        candidate_limit=args.candidate_limit,
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
        target_building_coverage=0.30,
        minimum_floors=3,
        maximum_floors=6,
        target_road_width_m=6.0,
        ventilation_corridor_width_m=2.0,
        shared_open_space_fraction=0.10,
        placement_grid=7,
        placement_variants=3,
        max_wind_objective_increase=args.max_wind_increase,
    )
    wind_scenario = SummerWindScenario(
        directions=(WindDirection(0.0, 0.7), WindDirection(90.0, 0.3)),
        model_name=args.wind_model,
    )
    model = load_model(
        args.wind_model,
        ckpt_dir=args.checkpoint_dir,
        allow_fallback=args.allow_model_fallback,
    )
    plan = optimize_joint_renewal(
        district,
        policy=joint_policy,
        scenario=wind_scenario,
        model=model,
    )
    result = evaluate_joint_renewal_feasibility(
        district,
        plan,
        JointRenewalFeasibilityPolicy(
            start_year=args.year,
            years_per_phase=args.years_per_phase,
            construction_months=args.construction_months,
            replacement_dwelling_area_m2=args.replacement_dwelling_area,
            residential_efficiency=args.residential_efficiency,
            construction_cost_jpy_per_m2=args.construction_cost,
            demolition_cost_jpy_per_m2=args.demolition_cost,
            relocation_monthly_support_jpy_per_dwelling=args.relocation_monthly,
        ),
    )

    output = PROJECT_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict(include_districts=args.include_districts)
    payload["model"] = model.provenance(include_hash=args.hash_model)
    payload["wind_visualization"] = _wind_visualization(
        result, wind_scenario, model, args.wind_display_res
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure = _plot(result, PROJECT_ROOT / args.fig)
    summary = result.summary
    print(
        f"projects={summary.projects} rights={summary.rights_converted} "
        f"cost={summary.total_cost_jpy / 1e9:.3f}bJPY "
        f"dwellings={summary.initial_dwellings}->{summary.final_dwellings} "
        f"temporary_peak={summary.peak_temporary_relocation_dwellings} "
        f"open_space={summary.connected_open_space_m2:.1f}m2"
    )
    print(f"saved {output}")
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
