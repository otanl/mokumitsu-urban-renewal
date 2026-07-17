"""Command-line entry points for the mokumitsu research workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .district import (
    generate_mokumitsu,
    load_district,
    renewal_priorities,
    save_district,
)
from .feasibility import (
    JointRenewalFeasibilityPolicy,
    evaluate_joint_renewal_feasibility,
)
from .fire import FireScenario
from .joint_renewal import JointRenewalPolicy, optimize_joint_renewal
from .model import load_model
from .pareto import ParetoRenewalPolicy, evaluate_joint_renewal_pareto
from .wind import SummerWindScenario, WindDirection


def _write_json(path: str | Path, value: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _district(args: argparse.Namespace):
    if getattr(args, "district", None):
        return load_district(args.district)
    return generate_mokumitsu(seed=args.seed)


def _wind_rose(value: str) -> tuple[WindDirection, ...]:
    directions = []
    for item in value.split(","):
        try:
            direction, weight = item.split(":", 1)
            directions.append(WindDirection(float(direction), float(weight)))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "wind rose must use direction:weight pairs, for example 0:0.7,90:0.3"
            ) from error
    if not directions:
        raise argparse.ArgumentTypeError("wind rose must not be empty")
    return tuple(directions)


def _run_generate(args: argparse.Namespace) -> int:
    district = generate_mokumitsu(seed=args.seed)
    output = save_district(district, args.out)
    print(f"saved {output}")
    print(json.dumps(asdict(district.summary()), ensure_ascii=False))
    return 0


def _run_prioritize(args: argparse.Namespace) -> int:
    district = _district(args)
    priorities = renewal_priorities(district)[: args.limit]
    payload = {
        "district": {
            "seed": district.seed,
            "generator": district.generator,
            "summary": asdict(district.summary()),
        },
        "priorities": [asdict(item) for item in priorities],
    }
    output = _write_json(args.out, payload)
    print(f"saved {output}")
    return 0


def _run_pareto(args: argparse.Namespace) -> int:
    district = _district(args)
    model = load_model(
        args.wind_model,
        ckpt_dir=args.checkpoint_dir,
        allow_fallback=args.allow_model_fallback,
    )
    result = evaluate_joint_renewal_pareto(
        district,
        joint_policy=JointRenewalPolicy(
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
        ),
        pareto_policy=ParetoRenewalPolicy(
            candidates_per_cluster_size=args.candidates_per_size,
        ),
        wind_scenario=SummerWindScenario(
            directions=args.wind_rose,
            model_name=args.wind_model,
        ),
        fire_scenario=FireScenario(
            runs=args.runs,
            seed=args.seed,
            wind_direction_deg=args.fire_wind_direction,
            wind_speed_m_s=args.fire_wind_speed,
        ),
        model=model,
    )
    payload = result.to_dict(
        include_recommended_district=args.include_recommended_district,
    )
    payload["model"] = model.provenance(include_hash=args.hash_model)
    output = _write_json(args.out, payload)
    print(
        f"saved {output}; alternatives={len(result.alternatives)} "
        f"pareto={len(result.pareto_indices)} recommended={result.recommended_index}"
    )
    return 0


def _run_feasibility(args: argparse.Namespace) -> int:
    district = _district(args)
    model = load_model(
        args.wind_model,
        ckpt_dir=args.checkpoint_dir,
        allow_fallback=args.allow_model_fallback,
    )
    joint_policy = JointRenewalPolicy(
        year=args.year,
        projects=args.projects,
        candidate_limit=args.candidate_limit,
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
        max_wind_objective_increase=args.max_wind_objective_increase,
    )
    plan = optimize_joint_renewal(
        district,
        policy=joint_policy,
        scenario=SummerWindScenario(
            directions=args.wind_rose,
            model_name=args.wind_model,
        ),
        model=model,
    )
    result = evaluate_joint_renewal_feasibility(
        district,
        plan,
        policy=JointRenewalFeasibilityPolicy(
            start_year=args.year,
            years_per_phase=args.years_per_phase,
            construction_months=args.construction_months,
            existing_dwellings_per_residential_building=(args.existing_dwellings_per_building),
            residential_efficiency=args.residential_efficiency,
            replacement_dwelling_area_m2=args.replacement_dwelling_area,
            land_value_weight=args.land_value_weight,
            building_value_weight=args.building_value_weight,
            construction_cost_jpy_per_m2=args.construction_cost,
            demolition_cost_jpy_per_m2=args.demolition_cost,
            road_work_cost_jpy_per_m2=args.road_work_cost,
            open_space_cost_jpy_per_m2=args.open_space_cost,
            relocation_monthly_support_jpy_per_dwelling=args.relocation_monthly,
            moving_cost_jpy_per_dwelling=args.moving_cost,
            rights_coordination_cost_jpy_per_holder=args.rights_coordination_cost,
            design_and_management_rate=args.design_management_rate,
            contingency_rate=args.contingency_rate,
        ),
    )
    payload = result.to_dict(include_districts=args.include_districts)
    payload["model"] = model.provenance(include_hash=args.hash_model)
    output = _write_json(args.out, payload)
    summary = result.summary
    print(
        f"saved {output}; projects={summary.projects} rights={summary.rights_converted} "
        f"cost={summary.total_cost_jpy / 1e9:.3f}bJPY "
        f"dwellings={summary.initial_dwellings}->{summary.final_dwellings} "
        f"temporary_peak={summary.peak_temporary_relocation_dwellings}"
    )
    return 0


def _add_district_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--district", help="existing district JSON; generated when omitted")
    parser.add_argument("--seed", type=int, default=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mokumitsu",
        description="Dense-neighbourhood generation, renewal ranking and Pareto screening.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate an organic synthetic district")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--out", default="outputs/mokumitsu.json")
    generate.set_defaults(handler=_run_generate)

    prioritize = commands.add_parser(
        "prioritize",
        help="rank individual replacement need while reporting access feasibility separately",
    )
    _add_district_source(prioritize)
    prioritize.add_argument("--limit", type=int, default=20)
    prioritize.add_argument("--out", default="outputs/mokumitsu_priorities.json")
    prioritize.set_defaults(handler=_run_prioritize)

    pareto = commands.add_parser(
        "pareto",
        help="screen connected 2-4 parcel joint-renewal alternatives",
    )
    _add_district_source(pareto)
    pareto.add_argument("--year", type=int, default=2036)
    pareto.add_argument("--minimum-parcels", type=int, default=2)
    pareto.add_argument("--maximum-parcels", type=int, default=4)
    pareto.add_argument("--candidates-per-size", type=int, default=4)
    pareto.add_argument("--placement-grid", type=int, default=7)
    pareto.add_argument("--placement-variants", type=int, default=3)
    pareto.add_argument("--target-coverage", type=float, default=0.30)
    pareto.add_argument("--minimum-floors", type=int, default=3)
    pareto.add_argument("--maximum-floors", type=int, default=6)
    pareto.add_argument("--target-road-width", type=float, default=6.0)
    pareto.add_argument("--corridor-width", type=float, default=2.0)
    pareto.add_argument("--open-space-fraction", type=float, default=0.10)
    pareto.add_argument("--wind-rose", type=_wind_rose, default=_wind_rose("0:0.7,90:0.3"))
    pareto.add_argument("--wind-model", default="residential")
    pareto.add_argument("--checkpoint-dir")
    pareto.add_argument(
        "--allow-model-fallback",
        action="store_true",
        help="explicitly permit a different registered checkpoint family",
    )
    pareto.add_argument("--hash-model", action="store_true")
    pareto.add_argument("--fire-wind-direction", type=float, default=0.0)
    pareto.add_argument("--fire-wind-speed", type=float, default=5.0)
    pareto.add_argument("--runs", type=int, default=40)
    pareto.add_argument("--include-recommended-district", action="store_true")
    pareto.add_argument("--out", default="outputs/mokumitsu_pareto.json")
    pareto.set_defaults(handler=_run_pareto)

    feasibility = commands.add_parser(
        "feasibility",
        help="screen phased joint renewal, rights conversion, relocation, dwellings and cost",
    )
    _add_district_source(feasibility)
    feasibility.add_argument("--year", type=int, default=2036)
    feasibility.add_argument("--years-per-phase", type=int, default=3)
    feasibility.add_argument("--projects", type=int, default=3)
    feasibility.add_argument("--candidate-limit", type=int, default=12)
    feasibility.add_argument("--minimum-parcels", type=int, default=2)
    feasibility.add_argument("--maximum-parcels", type=int, default=4)
    feasibility.add_argument("--placement-grid", type=int, default=7)
    feasibility.add_argument("--placement-variants", type=int, default=3)
    feasibility.add_argument("--target-coverage", type=float, default=0.30)
    feasibility.add_argument("--minimum-floors", type=int, default=3)
    feasibility.add_argument("--maximum-floors", type=int, default=6)
    feasibility.add_argument("--target-road-width", type=float, default=6.0)
    feasibility.add_argument("--corridor-width", type=float, default=2.0)
    feasibility.add_argument("--open-space-fraction", type=float, default=0.10)
    feasibility.add_argument("--max-wind-objective-increase", type=float, default=0.01)
    feasibility.add_argument(
        "--wind-rose",
        type=_wind_rose,
        default=_wind_rose("0:0.7,90:0.3"),
    )
    feasibility.add_argument("--wind-model", default="residential")
    feasibility.add_argument("--checkpoint-dir")
    feasibility.add_argument("--allow-model-fallback", action="store_true")
    feasibility.add_argument("--hash-model", action="store_true")
    feasibility.add_argument("--construction-months", type=int, default=18)
    feasibility.add_argument(
        "--existing-dwellings-per-building",
        type=int,
        default=1,
    )
    feasibility.add_argument("--residential-efficiency", type=float, default=0.82)
    feasibility.add_argument("--replacement-dwelling-area", type=float, default=55.0)
    feasibility.add_argument("--land-value-weight", type=float, default=0.70)
    feasibility.add_argument("--building-value-weight", type=float, default=0.30)
    feasibility.add_argument("--construction-cost", type=float, default=450_000.0)
    feasibility.add_argument("--demolition-cost", type=float, default=35_000.0)
    feasibility.add_argument("--road-work-cost", type=float, default=180_000.0)
    feasibility.add_argument("--open-space-cost", type=float, default=90_000.0)
    feasibility.add_argument("--relocation-monthly", type=float, default=150_000.0)
    feasibility.add_argument("--moving-cost", type=float, default=500_000.0)
    feasibility.add_argument(
        "--rights-coordination-cost",
        type=float,
        default=1_500_000.0,
    )
    feasibility.add_argument("--design-management-rate", type=float, default=0.12)
    feasibility.add_argument("--contingency-rate", type=float, default=0.10)
    feasibility.add_argument("--include-districts", action="store_true")
    feasibility.add_argument("--out", default="outputs/mokumitsu_feasibility.json")
    feasibility.set_defaults(handler=_run_feasibility)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the research CLI."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))
