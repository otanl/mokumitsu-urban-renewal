"""Generate and inspect a synthetic 木密 district before fire/wind optimization.

The JSON is the shared hand-off to Houdini and later renewal/fire simulators. The preview shows
construction cohorts on the left and the first-pass renewal action on the right.

    .venv\\Scripts\\python.exe scripts\\gen_mokumitsu.py --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

ROOT = Path(__file__).resolve().parent.parent

from mokumitsu import (  # noqa: E402
    MokumitsuConfig,
    generate_mokumitsu,
    generate_mokumitsu_grid,
    morphology_summary,
    renewal_priorities,
    save_district,
)


def _road_polygon(road):
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


def _preview(district, path):
    priorities = renewal_priorities(district)
    candidate = {c.building_id: c for c in priorities}
    age_colours = {"pre_1981": "#b2182b", "1981_2000": "#ef8a62", "post_2000": "#67a9cf"}
    action_colours = {
        "early_individual_rebuild": "#d73027",
        "joint_rebuild_or_access_improvement": "#7b3294",
        "planned_individual_rebuild": "#fdae61",
        "retain_or_group_later": "#91bfdb",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, mode in zip(axes, ("age", "renewal"), strict=True):
        for road in district.roads:
            if road.legal_class == "article_42_1":
                colour = "#d9d9d9"
            elif road.legal_class == "article_42_2":
                colour = "#f0d58c"
            else:
                colour = "#f7f7f7"
            ax.add_patch(
                PolygonPatch(_road_polygon(road), closed=True, fc=colour, ec="#888", lw=0.3)
            )
        for parcel in district.parcels:
            ax.add_patch(
                PolygonPatch(parcel.polygon, closed=True, fill=False, ec="#aaaaaa", lw=0.25)
            )
        for building in district.buildings:
            colour = (
                age_colours[building.age_cohort]
                if mode == "age"
                else action_colours[candidate[building.id].recommended_action]
            )
            ax.add_patch(PolygonPatch(building.corners, closed=True, fc=colour, ec="#333", lw=0.25))
        ax.set_aspect("equal")
        ax.set_xlim(0, district.width_m)
        ax.set_ylim(0, district.height_m)
        ax.set_title("construction cohort" if mode == "age" else "renewal action")
        ax.set_xlabel("m")
    axes[0].set_ylabel("m")
    summary = district.summary()
    fig.suptitle(
        f"Dense wooden district seed={district.seed}: {summary.buildings} buildings, "
        f"legal access {summary.legal_access_rate:.0%}, pre-1981 {summary.pre_1981_rate:.0%}"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=float, default=100.0)
    parser.add_argument("--height", type=float, default=100.0)
    parser.add_argument("--historic-routes", type=int, default=2)
    parser.add_argument("--branch-alleys", type=int, default=8)
    parser.add_argument("--net-coverage", type=float, default=0.54)
    parser.add_argument("--grid", action="store_true", help="use the legacy test baseline")
    parser.add_argument("--out", default="data/mokumitsu.json")
    parser.add_argument("--fig", default="outputs/mokumitsu.png")
    args = parser.parse_args()

    config = MokumitsuConfig(
        width_m=args.width,
        height_m=args.height,
        historic_routes=args.historic_routes,
        branch_alleys=args.branch_alleys,
        target_net_building_coverage=args.net_coverage,
    )
    generator = generate_mokumitsu_grid if args.grid else generate_mokumitsu
    district = generator(args.seed, config)
    out = save_district(district, ROOT / args.out)
    fig = _preview(district, ROOT / args.fig)
    summary = district.summary()
    candidates = renewal_priorities(district)
    print(f"saved {out}")
    print(f"saved {fig}")
    print(
        f"parcels={summary.parcels} buildings={summary.buildings} roads={summary.roads} "
        f"coverage={summary.footprint_coverage:.3f} legal_access={summary.legal_access_rate:.3f} "
        f"net_coverage={summary.net_building_coverage:.3f} "
        f"FAR={summary.gross_floor_area_ratio:.3f} "
        f"density={summary.building_density_per_ha:.1f}/ha "
        f"individually_rebuildable={summary.individual_rebuildable_rate:.3f}"
    )
    morphology = morphology_summary(district)
    print(
        f"generator={district.generator} non_axis_roads="
        f"{morphology.non_axis_aligned_road_rate:.3f} "
        f"orientation_entropy={morphology.road_orientation_entropy:.3f} "
        f"dead_ends={morphology.dead_end_road_rate:.3f} "
        f"irregular_parcels={morphology.irregular_parcel_rate:.3f} "
        f"flag_lots={morphology.flag_lot_rate:.3f} back_lots={morphology.back_lot_rate:.3f}"
    )
    for item in candidates[:8]:
        print(
            f"#{item.rank:02d} {item.building_id}: need={item.need_score:.3f} "
            f"feas={item.feasibility_score:.2f} {item.recommended_action} "
            f"[{', '.join(item.reasons)}]"
        )


if __name__ == "__main__":
    main()
