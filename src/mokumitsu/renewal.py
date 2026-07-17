"""Phased redevelopment baseline for a generated 木密 district.

This module deliberately starts with the legally simple action: individually rebuildable
parcels are renewed in priority order, while parcels needing joint redevelopment or access
improvement are recorded as deferred projects.  Geometry and parcel ownership remain fixed in
this baseline.  That makes the access bottleneck visible instead of silently assuming that an
inaccessible parcel can obtain a permit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .district import DistrictBuilding, MokumitsuDistrict, validate_district
from .fire import (
    FireRiskResult,
    FireScenario,
    fire_aware_renewal_priorities,
    simulate_fire_spread,
)


@dataclass(frozen=True)
class RenewalPolicy:
    """Capacity and replacement assumptions for one scenario."""

    start_year: int = 2026
    phases: int = 6
    years_per_phase: int = 5
    rebuild_fraction_per_phase: float = 0.08
    fire_weight: float = 0.35
    replacement_structure: str = "reinforced_concrete"
    replacement_fire_resistance: str = "fire_resistant"
    minimum_replacement_floors: int = 3
    floor_height_m: float = 3.0


@dataclass(frozen=True)
class RenewalAction:
    phase: int
    year: int
    building_id: str
    parcel_id: str
    action: str
    status: str
    priority_score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RenewalPhaseMetrics:
    phase: int
    year: int
    renewed_this_phase: int
    renewed_total: int
    deferred_joint_total: int
    footprint_coverage: float
    net_building_coverage: float
    gross_floor_area_m2: float
    gross_floor_area_ratio: float
    pre_1981_rate: float
    non_fire_resistant_rate: float
    expected_burned_buildings: float
    p95_burned_buildings: float
    expected_burned_area_m2: float
    wind_mean_index: float | None = None
    wind_dead_fraction: float | None = None
    wind_strong_fraction: float | None = None
    road_dead_fraction: float | None = None
    yard_dead_fraction: float | None = None
    building_edge_strong_fraction: float | None = None


@dataclass(frozen=True)
class RenewalPhase:
    index: int
    year: int
    district: MokumitsuDistrict
    actions: tuple[RenewalAction, ...]
    fire: FireRiskResult
    metrics: RenewalPhaseMetrics


@dataclass(frozen=True)
class RenewalTrajectory:
    policy: RenewalPolicy
    phases: tuple[RenewalPhase, ...]

    def to_dict(self, include_districts: bool = False) -> dict:
        phase_data = []
        for phase in self.phases:
            item = {
                "index": phase.index,
                "year": phase.year,
                "actions": [asdict(action) for action in phase.actions],
                "metrics": asdict(phase.metrics),
                "fire": phase.fire.to_dict(),
            }
            if include_districts:
                item["district"] = phase.district.to_dict()
            phase_data.append(item)
        return {"policy": asdict(self.policy), "phases": phase_data}


def simulate_renewal_trajectory(
    district: MokumitsuDistrict,
    policy: RenewalPolicy | None = None,
    fire_scenario: FireScenario | None = None,
) -> RenewalTrajectory:
    """Apply priority-ordered individual rebuilding and report deferred joint projects."""
    policy = policy or RenewalPolicy(start_year=district.reference_year)
    scenario = fire_scenario or FireScenario(runs=160, seed=district.seed)
    _validate_policy(policy)
    errors = validate_district(district)
    if errors:
        raise ValueError("invalid initial district: " + "; ".join(errors))

    current = replace(district, reference_year=policy.start_year)
    renewed: set[str] = set()
    deferred: set[str] = set()
    initial_fire = simulate_fire_spread(current, scenario)
    phases = [
        RenewalPhase(
            index=0,
            year=policy.start_year,
            district=current,
            actions=(),
            fire=initial_fire,
            metrics=_phase_metrics(current, initial_fire, 0, 0, 0, 0, policy.start_year),
        )
    ]
    fire = initial_fire
    capacity = max(1, math.ceil(len(district.buildings) * policy.rebuild_fraction_per_phase))

    for phase_index in range(1, policy.phases + 1):
        year = policy.start_year + phase_index * policy.years_per_phase
        ranked = fire_aware_renewal_priorities(current, fire, fire_weight=policy.fire_weight)
        buildings = {building.id: building for building in current.buildings}
        replacements: dict[str, DistrictBuilding] = {}
        actions = []

        for candidate in ranked:
            if candidate.building_id in renewed:
                continue
            access = current.access(candidate.parcel_id)
            if access.individual_rebuildable and len(replacements) < capacity:
                building = buildings[candidate.building_id]
                replacements[building.id] = _replacement_building(building, year, policy)
                renewed.add(building.id)
                actions.append(
                    RenewalAction(
                        phase=phase_index,
                        year=year,
                        building_id=building.id,
                        parcel_id=building.parcel_id,
                        action="individual_rebuild",
                        status="completed",
                        priority_score=candidate.combined_score,
                        reasons=candidate.reasons,
                    )
                )
            elif not access.individual_rebuildable and candidate.building_id not in deferred:
                deferred.add(candidate.building_id)
                actions.append(
                    RenewalAction(
                        phase=phase_index,
                        year=year,
                        building_id=candidate.building_id,
                        parcel_id=candidate.parcel_id,
                        action="joint_rebuild_or_access_improvement",
                        status="deferred",
                        priority_score=candidate.combined_score,
                        reasons=candidate.reasons,
                    )
                )
            if len(replacements) >= capacity:
                break

        next_buildings = tuple(replacements.get(b.id, b) for b in current.buildings)
        current = replace(current, reference_year=year, buildings=next_buildings)
        errors = validate_district(current)
        if errors:
            raise RuntimeError("renewal produced invalid district: " + "; ".join(errors))
        fire = simulate_fire_spread(current, scenario)
        metrics = _phase_metrics(
            current,
            fire,
            phase_index,
            len(replacements),
            len(renewed),
            len(deferred),
            year,
        )
        phases.append(
            RenewalPhase(
                index=phase_index,
                year=year,
                district=current,
                actions=tuple(actions),
                fire=fire,
                metrics=metrics,
            )
        )

    return RenewalTrajectory(policy=policy, phases=tuple(phases))


def save_renewal_trajectory(
    trajectory: RenewalTrajectory,
    path: str | Path,
    include_districts: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            trajectory.to_dict(include_districts=include_districts),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _replacement_building(building, year, policy):
    floors = max(building.floors, policy.minimum_replacement_floors)
    return replace(
        building,
        floors=floors,
        height_m=max(building.height_m, floors * policy.floor_height_m + 0.5),
        built_year=year,
        structure=policy.replacement_structure,
        fire_resistance=policy.replacement_fire_resistance,
    )


def _phase_metrics(district, fire, phase, renewed_now, renewed_total, deferred_total, year):
    summary = district.summary()
    return RenewalPhaseMetrics(
        phase=phase,
        year=year,
        renewed_this_phase=renewed_now,
        renewed_total=renewed_total,
        deferred_joint_total=deferred_total,
        footprint_coverage=summary.footprint_coverage,
        net_building_coverage=summary.net_building_coverage,
        gross_floor_area_m2=summary.gross_floor_area_m2,
        gross_floor_area_ratio=summary.gross_floor_area_ratio,
        pre_1981_rate=summary.pre_1981_rate,
        non_fire_resistant_rate=summary.non_fire_resistant_rate,
        expected_burned_buildings=fire.expected_burned_buildings,
        p95_burned_buildings=fire.p95_burned_buildings,
        expected_burned_area_m2=fire.expected_burned_area_m2,
    )


def _validate_policy(policy):
    if policy.phases <= 0 or policy.years_per_phase <= 0:
        raise ValueError("phases and years_per_phase must be positive")
    if not 0 < policy.rebuild_fraction_per_phase <= 1:
        raise ValueError("rebuild_fraction_per_phase must be in (0, 1]")
    if not 0 <= policy.fire_weight <= 1:
        raise ValueError("fire_weight must be in [0, 1]")
    if policy.minimum_replacement_floors <= 0 or policy.floor_height_m <= 0:
        raise ValueError("replacement floor controls must be positive")
