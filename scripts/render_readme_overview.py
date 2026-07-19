"""Render the factual, non-CFD project overview used by the README files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Polygon as PolygonPatch  # noqa: E402

from mokumitsu import generate_mokumitsu, renewal_priorities  # noqa: E402
from mokumitsu.fire import (  # noqa: E402
    FireScenario,
    fire_aware_renewal_priorities,
    simulate_fire_spread,
)

AGE_COLOURS = {
    "pre_1981": "#b2182b",
    "1981_2000": "#ef8a62",
    "post_2000": "#67a9cf",
}
AGE_LABELS = {
    "pre_1981": "pre-1981",
    "1981_2000": "1981–2000",
    "post_2000": "post-2000",
}
ACTION_COLOURS = {
    "early_individual_rebuild": "#d73027",
    "joint_rebuild_or_access_improvement": "#7b3294",
    "planned_individual_rebuild": "#fdae61",
    "retain_or_group_later": "#91bfdb",
}
ACTION_LABELS = {
    "early_individual_rebuild": "early individual",
    "joint_rebuild_or_access_improvement": "joint / access",
    "planned_individual_rebuild": "planned individual",
    "retain_or_group_later": "retain / later",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the non-CFD Mokumitsu workflow overview for the README."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fire-runs", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/mokumitsu_overview.png"),
    )
    return parser.parse_args()


def _road_polygon(road) -> list[tuple[float, float]]:
    x0, y0 = road.start
    x1, y1 = road.end
    length = max(road.length_m, 1e-12)
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length
    hx, hy = 0.5 * road.width_m * nx, 0.5 * road.width_m * ny
    return [
        (x0 + hx, y0 + hy),
        (x1 + hx, y1 + hy),
        (x1 - hx, y1 - hy),
        (x0 - hx, y0 - hy),
    ]


def _draw_context(ax, district) -> None:
    for road in district.roads:
        if road.legal_class == "article_42_1":
            colour = "#d9d9d9"
        elif road.legal_class == "article_42_2":
            colour = "#f0d58c"
        else:
            colour = "#f7f7f7"
        ax.add_patch(
            PolygonPatch(
                _road_polygon(road),
                closed=True,
                fc=colour,
                ec="#8c8c8c",
                lw=0.35,
            )
        )
    for parcel in district.parcels:
        ax.add_patch(
            PolygonPatch(
                parcel.polygon,
                closed=True,
                fill=False,
                ec="#b5b5b5",
                lw=0.25,
            )
        )
    ax.set_aspect("equal")
    ax.set_xlim(0, district.width_m)
    ax.set_ylim(0, district.height_m)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")


def _draw_buildings(ax, district, colours: dict[str, object]) -> None:
    for building in district.buildings:
        ax.add_patch(
            PolygonPatch(
                building.corners,
                closed=True,
                fc=colours[building.id],
                ec="#282828",
                lw=0.35,
            )
        )


def render(seed: int, fire_runs: int, output: Path) -> Path:
    if fire_runs <= 0:
        raise ValueError("fire_runs must be positive")

    district = generate_mokumitsu(seed=seed)
    priorities = renewal_priorities(district)
    fire = simulate_fire_spread(
        district,
        FireScenario(seed=seed, runs=fire_runs),
    )
    combined = fire_aware_renewal_priorities(district, fire)

    action_by_building = {item.building_id: item.recommended_action for item in priorities}
    fire_by_building = dict(zip(fire.building_ids, fire.burn_probability, strict=True))
    combined_by_building = {item.building_id: item.combined_score for item in combined}
    rank_by_building = {item.building_id: item.rank for item in combined}

    combined_min = min(combined_by_building.values())
    combined_max = max(combined_by_building.values())
    combined_norm = Normalize(vmin=combined_min, vmax=combined_max)
    fire_norm = Normalize(vmin=0.0, vmax=1.0)

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18.2, 5.4),
        sharex=True,
        sharey=True,
    )
    for ax in axes:
        _draw_context(ax, district)

    age_colours = {building.id: AGE_COLOURS[building.age_cohort] for building in district.buildings}
    _draw_buildings(axes[0], district, age_colours)
    axes[0].set_title("1  Procedural district\nconstruction cohort", fontweight="bold")
    axes[0].legend(
        handles=[
            Patch(facecolor=AGE_COLOURS[key], edgecolor="#282828", label=AGE_LABELS[key])
            for key in AGE_COLOURS
        ],
        loc="lower left",
        fontsize=7.5,
        framealpha=0.94,
    )

    action_colours = {
        building.id: ACTION_COLOURS[action_by_building[building.id]]
        for building in district.buildings
    }
    _draw_buildings(axes[1], district, action_colours)
    axes[1].set_title("2  Access + age screen\nrecommended renewal action", fontweight="bold")
    axes[1].legend(
        handles=[
            Patch(
                facecolor=ACTION_COLOURS[key],
                edgecolor="#282828",
                label=ACTION_LABELS[key],
            )
            for key in ACTION_COLOURS
        ],
        loc="lower left",
        fontsize=7.0,
        framealpha=0.94,
    )

    fire_colours = {
        building.id: plt.cm.inferno(fire_norm(fire_by_building[building.id]))
        for building in district.buildings
    }
    _draw_buildings(axes[2], district, fire_colours)
    axes[2].set_title("3  Graph fire screen\nburn probability", fontweight="bold")
    fire_bar = figure.colorbar(
        ScalarMappable(norm=fire_norm, cmap="inferno"),
        ax=axes[2],
        orientation="horizontal",
        fraction=0.05,
        pad=0.035,
    )
    fire_bar.ax.tick_params(labelsize=7)

    priority_colours = {
        building.id: plt.cm.viridis(combined_norm(combined_by_building[building.id]))
        for building in district.buildings
    }
    _draw_buildings(axes[3], district, priority_colours)
    for building in district.buildings:
        rank = rank_by_building[building.id]
        if rank <= 5:
            axes[3].text(
                building.cx,
                building.cy,
                str(rank),
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
                path_effects=[path_effects.withStroke(linewidth=1.5, foreground="#202020")],
            )
    axes[3].set_title(
        "4  Combined decision screen\nrenewal priority (top 5 labelled)", fontweight="bold"
    )
    priority_bar = figure.colorbar(
        ScalarMappable(norm=combined_norm, cmap="viridis"),
        ax=axes[3],
        orientation="horizontal",
        fraction=0.05,
        pad=0.035,
    )
    priority_bar.ax.tick_params(labelsize=7)

    summary = district.summary()
    figure.suptitle(
        "Mokumitsu — dense-neighbourhood renewal screening",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"seed {district.seed} · {summary.buildings} buildings · "
            f"legal access {summary.legal_access_rate:.0%} · "
            f"pre-1981 {summary.pre_1981_rate:.0%} · "
            f"fire-screen expected burn {fire.expected_burned_buildings:.1f} buildings"
        ),
        ha="center",
        fontsize=9.5,
        color="#333333",
    )
    figure.text(
        0.5,
        0.015,
        (
            "Screening-level planning workflow. Wind/CFD rankings are intentionally excluded "
            "until the physical grid-validation gate passes."
        ),
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.025, right=0.99, top=0.86, bottom=0.105, wspace=0.08)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    path = render(args.seed, args.fire_runs, args.out)
    print(path)


if __name__ == "__main__":
    main()
