"""Run the fast fire-spread screen on a generated/saved 木密 district.

.venv\\Scripts\\python.exe scripts\\simulate_mokumitsu_fire.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

ROOT = Path(__file__).resolve().parent.parent

from mokumitsu import generate_mokumitsu, load_district, save_district  # noqa: E402
from mokumitsu.fire import (  # noqa: E402
    FireScenario,
    fire_aware_renewal_priorities,
    simulate_fire_spread,
)


def _preview(district, result, path):
    values = (result.burn_probability, result.source_consequence)
    titles = ("probability of burning", "source spread consequence")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, field, title in zip(axes, values, titles, strict=True):
        lookup = dict(zip(result.building_ids, field, strict=True))
        for parcel in district.parcels:
            ax.add_patch(
                PolygonPatch(parcel.polygon, closed=True, fill=False, ec="#cccccc", lw=0.25)
            )
        for building in district.buildings:
            value = lookup[building.id]
            colour = plt.cm.inferno(min(1.0, value))
            ax.add_patch(
                PolygonPatch(
                    building.corners,
                    closed=True,
                    fc=colour,
                    ec="#222222",
                    lw=0.25,
                )
            )
        ax.set_xlim(0, district.width_m)
        ax.set_ylim(0, district.height_m)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("m")
    axes[0].set_ylabel("m")
    fig.suptitle(
        f"Fire screen: wind {result.scenario.wind_direction_deg:.0f} deg / "
        f"{result.scenario.wind_speed_m_s:.1f} m/s; "
        f"expected {result.expected_burned_buildings:.1f} buildings"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default="data/mokumitsu.json")
    parser.add_argument("--seed", type=int, default=0, help="used when --district does not exist")
    parser.add_argument("--wind-dir", type=float, default=0.0, help="flow direction: 0 = +x")
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--horizon", type=float, default=180.0)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--out", default="outputs/mokumitsu_fire.json")
    parser.add_argument("--fig", default="outputs/mokumitsu_fire.png")
    args = parser.parse_args()

    district_path = ROOT / args.district
    if district_path.exists():
        district = load_district(district_path)
    else:
        district = generate_mokumitsu(args.seed)
        save_district(district, district_path)
    scenario = FireScenario(
        wind_direction_deg=args.wind_dir,
        wind_speed_m_s=args.wind_speed,
        horizon_min=args.horizon,
        runs=args.runs,
        seed=args.seed,
    )
    result = simulate_fire_spread(district, scenario)
    priorities = fire_aware_renewal_priorities(district, result)
    payload = result.to_dict()
    payload["renewal_priorities"] = [asdict(item) for item in priorities]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fig = _preview(district, result, ROOT / args.fig)
    print(f"saved {out}")
    print(f"saved {fig}")
    print(
        f"expected={result.expected_burned_buildings:.2f} buildings "
        f"p95={result.p95_burned_buildings:.0f} "
        f"large_fire_p={result.probability_large_fire:.3f} "
        f"expected_area={result.expected_burned_area_m2:.1f} m2"
    )
    for item in priorities[:8]:
        print(
            f"#{item.rank:02d} {item.building_id}: score={item.combined_score:.3f} "
            f"fire={item.fire_exposure:.3f}/{item.source_consequence:.3f} "
            f"feas={item.feasibility_score:.2f} {item.recommended_action}"
        )


if __name__ == "__main__":
    main()
