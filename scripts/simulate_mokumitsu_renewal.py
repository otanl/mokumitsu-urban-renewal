"""Run the priority-ordered phased redevelopment baseline.

.venv\\Scripts\\python.exe scripts\\simulate_mokumitsu_renewal.py --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

from mokumitsu import generate_mokumitsu, load_district  # noqa: E402
from mokumitsu.fire import FireScenario  # noqa: E402
from mokumitsu.renewal import (  # noqa: E402
    RenewalPolicy,
    save_renewal_trajectory,
    simulate_renewal_trajectory,
)
from mokumitsu.wind import (  # noqa: E402
    SummerWindScenario,
    WindDirection,
    add_wind_to_trajectory,
)


def _parse_wind_rose(value: str) -> tuple[WindDirection, ...]:
    """Parse toward_degrees[:weight] entries separated by commas."""
    directions = []
    try:
        for entry in value.split(","):
            parts = entry.strip().split(":")
            if len(parts) not in (1, 2) or not parts[0]:
                raise ValueError
            directions.append(
                WindDirection(
                    direction_deg=float(parts[0]),
                    weight=float(parts[1]) if len(parts) == 2 else 1.0,
                )
            )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated toward_degrees[:weight], e.g. 0:0.7,90:0.3"
        ) from exc
    if not directions:
        raise argparse.ArgumentTypeError("summer wind rose cannot be empty")
    return tuple(directions)


def _plot_wind(metrics, weak_ax, strong_ax):
    if metrics[0].wind_dead_fraction is None:
        for ax, title in (
            (weak_ax, "summer wind: weak areas"),
            (strong_ax, "summer wind: strong areas"),
        ):
            ax.text(0.5, 0.5, "not evaluated", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
        return

    years = [m.year for m in metrics]
    weak_ax.plot(
        years,
        [100 * m.wind_dead_fraction for m in metrics],
        "o-",
        label="all outdoor",
        color="#457b9d",
    )
    weak_ax.plot(
        years,
        [100 * m.road_dead_fraction for m in metrics],
        "o-",
        label="roads",
        color="#2a9d8f",
    )
    weak_ax.plot(
        years,
        [100 * m.yard_dead_fraction for m in metrics],
        "o-",
        label="parcel open space",
        color="#e9c46a",
    )
    weak_ax.set_ylabel("weak-wind area (%)")
    weak_ax.set_title("summer ventilation deficit")
    weak_ax.legend(frameon=False, fontsize=8)

    strong_ax.plot(
        years,
        [100 * m.wind_strong_fraction for m in metrics],
        "o-",
        label="all outdoor",
        color="#f4a261",
    )
    strong_ax.plot(
        years,
        [100 * m.building_edge_strong_fraction for m in metrics],
        "o-",
        label="within 3 m of buildings",
        color="#e63946",
    )
    strong_ax.set_ylabel("strong-wind area (%)")
    strong_ax.set_xlabel("year")
    strong_ax.set_title("building-wind screen")
    strong_ax.legend(frameon=False, fontsize=8)


def _plot(trajectory, path):
    metrics = [phase.metrics for phase in trajectory.phases]
    years = [metric.year for metric in metrics]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    axes[0, 0].plot(years, [m.gross_floor_area_ratio for m in metrics], "o-", color="#2878b5")
    axes[0, 0].set_ylabel("gross FAR")
    axes[0, 0].set_title("residential floor-area capacity")

    axes[0, 1].plot(years, [m.expected_burned_buildings for m in metrics], "o-", color="#c82423")
    axes[0, 1].fill_between(
        years,
        [m.expected_burned_buildings for m in metrics],
        [m.p95_burned_buildings for m in metrics],
        color="#c82423",
        alpha=0.16,
        label="mean to p95",
    )
    axes[0, 1].set_ylabel("burned buildings")
    axes[0, 1].set_title("fast fire-spread screen")
    axes[0, 1].legend(frameon=False)
    _plot_wind(metrics, axes[0, 2], axes[1, 2])

    axes[1, 0].plot(years, [m.non_fire_resistant_rate for m in metrics], "o-", color="#7a5195")
    axes[1, 0].set_ylabel("non-fire-resistant rate")
    axes[1, 0].set_xlabel("year")

    axes[1, 1].step(
        years, [m.renewed_total for m in metrics], where="post", label="renewed", color="#2a9d8f"
    )
    axes[1, 1].step(
        years,
        [m.deferred_joint_total for m in metrics],
        where="post",
        label="joint/access deferred",
        color="#e76f51",
    )
    axes[1, 1].set_ylabel("buildings")
    axes[1, 1].set_xlabel("year")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.22)
    fig.suptitle("Dense wooden district: individually rebuildable parcels first")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--district")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phases", type=int, default=6)
    parser.add_argument("--years-per-phase", type=int, default=5)
    parser.add_argument("--rebuild-fraction", type=float, default=0.08)
    parser.add_argument("--fire-weight", type=float, default=0.35)
    parser.add_argument("--runs", type=int, default=160)
    parser.add_argument("--wind-dir", type=float, default=0.0, help="fire wind, toward degrees")
    parser.add_argument("--wind-speed", type=float, default=5.0, help="fire wind speed (m/s)")
    parser.add_argument(
        "--summer-wind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate the FNO summer pedestrian-wind screen for every phase",
    )
    parser.add_argument(
        "--summer-wind-rose",
        type=_parse_wind_rose,
        default=_parse_wind_rose("0:1"),
        metavar="DEG[:WEIGHT],...",
        help="cardinal toward-directions, e.g. 0:0.7,90:0.3",
    )
    parser.add_argument("--wind-model", default="residential")
    parser.add_argument("--out", default="outputs/mokumitsu_renewal.json")
    parser.add_argument("--fig", default="outputs/mokumitsu_renewal.png")
    parser.add_argument("--include-districts", action="store_true")
    args = parser.parse_args()

    district = (
        load_district(ROOT / args.district) if args.district else generate_mokumitsu(args.seed)
    )
    policy = RenewalPolicy(
        start_year=district.reference_year,
        phases=args.phases,
        years_per_phase=args.years_per_phase,
        rebuild_fraction_per_phase=args.rebuild_fraction,
        fire_weight=args.fire_weight,
    )
    scenario = FireScenario(
        runs=args.runs,
        seed=args.seed,
        wind_direction_deg=args.wind_dir,
        wind_speed_m_s=args.wind_speed,
    )
    trajectory = simulate_renewal_trajectory(district, policy, scenario)
    if args.summer_wind:
        trajectory = add_wind_to_trajectory(
            trajectory,
            SummerWindScenario(
                directions=args.summer_wind_rose,
                model_name=args.wind_model,
            ),
        )
    out = save_renewal_trajectory(
        trajectory, ROOT / args.out, include_districts=args.include_districts
    )
    fig = _plot(trajectory, ROOT / args.fig)
    print(f"saved {out}")
    print(f"saved {fig}")
    for phase in trajectory.phases:
        m = phase.metrics
        print(
            f"phase={m.phase} year={m.year} renewed={m.renewed_total} "
            f"deferred={m.deferred_joint_total} gross_cov={m.footprint_coverage:.3f} "
            f"net_cov={m.net_building_coverage:.3f} FAR={m.gross_floor_area_ratio:.3f} "
            f"expected_burned={m.expected_burned_buildings:.2f} p95={m.p95_burned_buildings:.0f}"
            + (
                f" wind_mean={m.wind_mean_index:.3f} weak={m.wind_dead_fraction:.3f} "
                f"road_weak={m.road_dead_fraction:.3f} yard_weak={m.yard_dead_fraction:.3f} "
                f"strong={m.wind_strong_fraction:.3f} edge_strong="
                f"{m.building_edge_strong_fraction:.3f}"
                if m.wind_mean_index is not None
                else ""
            )
        )


if __name__ == "__main__":
    main()
