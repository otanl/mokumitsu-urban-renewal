"""Cumulative delivery screening for phased joint renewal.

The geometric and environmental models answer whether a joint project is physically useful.
This module adds a deliberately transparent first-pass delivery model: rights conversion,
temporary relocation, dwelling capacity and configurable project costs.  It is a screening
scenario, not an appraisal.  Every monetary unit rate is explicit in
``JointRenewalFeasibilityPolicy`` so empirical calibration can replace the defaults without
changing the geometry or CFD workflow.

The generated district does not yet contain household or tenure records.  The default therefore
uses one occupied dwelling per existing residential building.  Replacement dwelling capacity is
derived from residentially usable floor area and a target dwelling size.  Both assumptions are
policy parameters and are serialized with every result.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .district import DistrictBuilding, MokumitsuDistrict, Parcel, validate_district
from .joint_renewal import (
    JointOpenSpace,
    JointRenewalPlan,
    JointRenewalPolicy,
    JointRenewalStep,
    apply_joint_redevelopment,
)


@dataclass(frozen=True)
class JointRenewalFeasibilityPolicy:
    """Explicit delivery and cost assumptions for one screening scenario."""

    start_year: int | None = None
    years_per_phase: int = 3
    construction_months: int = 18
    existing_dwellings_per_residential_building: int = 1
    residential_efficiency: float = 0.82
    replacement_dwelling_area_m2: float = 55.0
    land_value_weight: float = 0.70
    building_value_weight: float = 0.30
    construction_cost_jpy_per_m2: float = 450_000.0
    demolition_cost_jpy_per_m2: float = 35_000.0
    road_work_cost_jpy_per_m2: float = 180_000.0
    open_space_cost_jpy_per_m2: float = 90_000.0
    relocation_monthly_support_jpy_per_dwelling: float = 150_000.0
    moving_cost_jpy_per_dwelling: float = 500_000.0
    rights_coordination_cost_jpy_per_holder: float = 1_500_000.0
    design_and_management_rate: float = 0.12
    contingency_rate: float = 0.10


@dataclass(frozen=True)
class JointRightsAllocation:
    """One participating parcel's proportional conversion into the joint building."""

    parcel_id: str
    building_ids: tuple[str, ...]
    land_area_m2: float
    existing_floor_area_m2: float
    existing_dwellings: int
    land_share: float
    building_share: float
    valuation_share: float
    allocated_floor_area_m2: float
    allocated_dwellings: int
    return_dwellings: int
    surplus_dwellings: int
    permanently_displaced_dwellings: int


@dataclass(frozen=True)
class JointProjectCost:
    """Cost components in nominal scenario yen; no subsidy or land acquisition is assumed."""

    demolition_jpy: float
    construction_jpy: float
    road_works_jpy: float
    open_space_works_jpy: float
    relocation_jpy: float
    rights_coordination_jpy: float
    design_and_management_jpy: float
    contingency_jpy: float
    hard_cost_jpy: float
    total_jpy: float


@dataclass(frozen=True)
class JointRenewalFeasibilityPhase:
    """One completed joint project and its cumulative delivery state."""

    index: int
    year: int
    project_id: str
    parcel_ids: tuple[str, ...]
    building_ids: tuple[str, ...]
    joint_building_id: str
    district: MokumitsuDistrict
    open_spaces: tuple[JointOpenSpace, ...]
    rights: tuple[JointRightsAllocation, ...]
    rights_count: int
    existing_dwellings: int
    replacement_dwellings: int
    return_dwellings: int
    surplus_dwellings: int
    permanently_displaced_dwellings: int
    temporary_relocation_dwellings: int
    temporary_relocation_dwelling_months: int
    district_dwellings_after: int
    connected_open_space_m2: float
    road_dedication_m2: float
    wind_objective_before: float
    wind_objective_after: float
    cost: JointProjectCost
    cumulative_cost_jpy: float
    cumulative_rights_converted: int
    cumulative_relocation_dwelling_months: int
    cumulative_net_dwellings: int


@dataclass(frozen=True)
class JointRenewalFeasibilitySummary:
    projects: int
    initial_dwellings: int
    final_dwellings: int
    net_dwellings: int
    rights_converted: int
    total_cost_jpy: float
    peak_temporary_relocation_dwellings: int
    relocation_dwelling_months: int
    return_dwellings: int
    surplus_dwellings: int
    permanently_displaced_dwellings: int
    connected_open_space_m2: float
    road_dedication_m2: float


@dataclass(frozen=True)
class JointRenewalFeasibilityResult:
    policy: JointRenewalFeasibilityPolicy
    joint_policy: JointRenewalPolicy
    initial_district: MokumitsuDistrict
    final_district: MokumitsuDistrict
    phases: tuple[JointRenewalFeasibilityPhase, ...]
    summary: JointRenewalFeasibilitySummary

    def to_dict(self, include_districts: bool = False) -> dict:
        phase_data = []
        for phase in self.phases:
            item = {
                "index": phase.index,
                "year": phase.year,
                "project_id": phase.project_id,
                "parcel_ids": phase.parcel_ids,
                "building_ids": phase.building_ids,
                "joint_building_id": phase.joint_building_id,
                "open_spaces": [asdict(space) for space in phase.open_spaces],
                "rights": [asdict(right) for right in phase.rights],
                "rights_count": phase.rights_count,
                "existing_dwellings": phase.existing_dwellings,
                "replacement_dwellings": phase.replacement_dwellings,
                "return_dwellings": phase.return_dwellings,
                "surplus_dwellings": phase.surplus_dwellings,
                "permanently_displaced_dwellings": phase.permanently_displaced_dwellings,
                "temporary_relocation_dwellings": phase.temporary_relocation_dwellings,
                "temporary_relocation_dwelling_months": (
                    phase.temporary_relocation_dwelling_months
                ),
                "district_dwellings_after": phase.district_dwellings_after,
                "connected_open_space_m2": phase.connected_open_space_m2,
                "road_dedication_m2": phase.road_dedication_m2,
                "wind_objective_before": phase.wind_objective_before,
                "wind_objective_after": phase.wind_objective_after,
                "cost": asdict(phase.cost),
                "cumulative_cost_jpy": phase.cumulative_cost_jpy,
                "cumulative_rights_converted": phase.cumulative_rights_converted,
                "cumulative_relocation_dwelling_months": (
                    phase.cumulative_relocation_dwelling_months
                ),
                "cumulative_net_dwellings": phase.cumulative_net_dwellings,
                "district_summary": asdict(phase.district.summary()),
            }
            if include_districts:
                item["district"] = phase.district.to_dict()
            phase_data.append(item)

        data = {
            "policy": asdict(self.policy),
            "joint_policy": asdict(self.joint_policy),
            "summary": asdict(self.summary),
            "initial_summary": asdict(self.initial_district.summary()),
            "final_summary": asdict(self.final_district.summary()),
            "phases": phase_data,
        }
        if include_districts:
            data["initial_district"] = self.initial_district.to_dict()
            data["final_district"] = self.final_district.to_dict()
        return data


def evaluate_joint_renewal_feasibility(
    district: MokumitsuDistrict,
    plan: JointRenewalPlan,
    policy: JointRenewalFeasibilityPolicy | None = None,
) -> JointRenewalFeasibilityResult:
    """Replay a geometric joint-renewal plan as a phased delivery scenario."""
    policy = policy or JointRenewalFeasibilityPolicy(start_year=plan.policy.year)
    _validate_policy(policy)
    errors = validate_district(district)
    if errors:
        raise ValueError("invalid initial district: " + "; ".join(errors))
    if tuple(step.index for step in plan.steps) != tuple(sorted(step.index for step in plan.steps)):
        raise ValueError("joint renewal steps must be ordered by index")

    start_year = policy.start_year if policy.start_year is not None else plan.policy.year
    current = replace(district, reference_year=start_year)
    initial_dwellings = _existing_dwellings(current.buildings, policy)
    district_dwellings = initial_dwellings
    cumulative_cost = 0.0
    cumulative_rights = 0
    cumulative_relocation_months = 0
    phases = []

    for sequence, step in enumerate(plan.steps, 1):
        year = start_year + (sequence - 1) * policy.years_per_phase
        selected_parcels, selected_buildings = _selected_assets(current, step)
        existing_dwellings = _existing_dwellings(selected_buildings, policy)
        replacement_dwellings = _replacement_dwellings(step, policy)
        rights = _rights_conversion(
            selected_parcels,
            selected_buildings,
            step,
            replacement_dwellings,
            policy,
        )
        return_dwellings = sum(right.return_dwellings for right in rights)
        surplus_dwellings = sum(right.surplus_dwellings for right in rights)
        displaced_dwellings = sum(right.permanently_displaced_dwellings for right in rights)
        relocation_months = existing_dwellings * policy.construction_months
        cost = _project_cost(step, existing_dwellings, policy)

        phase_joint_policy = replace(plan.policy, year=year)
        current = apply_joint_redevelopment(
            current,
            step.candidate,
            step.placement,
            phase_joint_policy,
        )
        current = replace(current, reference_year=year)
        district_dwellings += replacement_dwellings - existing_dwellings
        cumulative_cost += cost.total_jpy
        cumulative_rights += len(rights)
        cumulative_relocation_months += relocation_months
        phases.append(
            JointRenewalFeasibilityPhase(
                index=sequence,
                year=year,
                project_id=_project_id(step),
                parcel_ids=step.candidate.parcel_ids,
                building_ids=step.candidate.building_ids,
                joint_building_id=step.placement.joint_building_id,
                district=current,
                open_spaces=step.placement.open_spaces,
                rights=rights,
                rights_count=len(rights),
                existing_dwellings=existing_dwellings,
                replacement_dwellings=replacement_dwellings,
                return_dwellings=return_dwellings,
                surplus_dwellings=surplus_dwellings,
                permanently_displaced_dwellings=displaced_dwellings,
                temporary_relocation_dwellings=existing_dwellings,
                temporary_relocation_dwelling_months=relocation_months,
                district_dwellings_after=district_dwellings,
                connected_open_space_m2=step.placement.connected_open_space_m2,
                road_dedication_m2=step.placement.road_widening_area_m2,
                wind_objective_before=step.wind_objective_before,
                wind_objective_after=step.wind_objective_after,
                cost=cost,
                cumulative_cost_jpy=cumulative_cost,
                cumulative_rights_converted=cumulative_rights,
                cumulative_relocation_dwelling_months=cumulative_relocation_months,
                cumulative_net_dwellings=district_dwellings - initial_dwellings,
            )
        )

    final = phases[-1].district if phases else current
    summary = JointRenewalFeasibilitySummary(
        projects=len(phases),
        initial_dwellings=initial_dwellings,
        final_dwellings=district_dwellings,
        net_dwellings=district_dwellings - initial_dwellings,
        rights_converted=sum(phase.rights_count for phase in phases),
        total_cost_jpy=sum(phase.cost.total_jpy for phase in phases),
        peak_temporary_relocation_dwellings=max(
            (phase.temporary_relocation_dwellings for phase in phases),
            default=0,
        ),
        relocation_dwelling_months=sum(
            phase.temporary_relocation_dwelling_months for phase in phases
        ),
        return_dwellings=sum(phase.return_dwellings for phase in phases),
        surplus_dwellings=sum(phase.surplus_dwellings for phase in phases),
        permanently_displaced_dwellings=sum(
            phase.permanently_displaced_dwellings for phase in phases
        ),
        connected_open_space_m2=sum(phase.connected_open_space_m2 for phase in phases),
        road_dedication_m2=sum(phase.road_dedication_m2 for phase in phases),
    )
    return JointRenewalFeasibilityResult(
        policy=policy,
        joint_policy=plan.policy,
        initial_district=district,
        final_district=final,
        phases=tuple(phases),
        summary=summary,
    )


def save_joint_renewal_feasibility(
    result: JointRenewalFeasibilityResult,
    path: str | Path,
    include_districts: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            result.to_dict(include_districts=include_districts),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _selected_assets(
    district: MokumitsuDistrict,
    step: JointRenewalStep,
) -> tuple[tuple[Parcel, ...], tuple[DistrictBuilding, ...]]:
    parcel_ids = set(step.candidate.parcel_ids)
    building_ids = set(step.candidate.building_ids)
    parcels = tuple(parcel for parcel in district.parcels if parcel.id in parcel_ids)
    buildings = tuple(building for building in district.buildings if building.id in building_ids)
    if len(parcels) != len(parcel_ids):
        missing = sorted(parcel_ids - {parcel.id for parcel in parcels})
        raise ValueError("joint step references unavailable parcels: " + ", ".join(missing))
    if len(buildings) != len(building_ids):
        missing = sorted(building_ids - {building.id for building in buildings})
        raise ValueError("joint step references unavailable buildings: " + ", ".join(missing))
    if any(building.parcel_id not in parcel_ids for building in buildings):
        raise ValueError("joint step contains a building outside its participating parcels")
    return parcels, buildings


def _existing_dwellings(
    buildings: tuple[DistrictBuilding, ...],
    policy: JointRenewalFeasibilityPolicy,
) -> int:
    return sum(
        policy.existing_dwellings_per_residential_building
        for building in buildings
        if building.use == "residential"
    )


def _replacement_dwellings(
    step: JointRenewalStep,
    policy: JointRenewalFeasibilityPolicy,
) -> int:
    usable = step.placement.floor_area_m2 * policy.residential_efficiency
    return max(0, math.floor(usable / policy.replacement_dwelling_area_m2 + 1e-9))


def _rights_conversion(
    parcels: tuple[Parcel, ...],
    buildings: tuple[DistrictBuilding, ...],
    step: JointRenewalStep,
    replacement_dwellings: int,
    policy: JointRenewalFeasibilityPolicy,
) -> tuple[JointRightsAllocation, ...]:
    buildings_by_parcel = {
        parcel.id: tuple(b for b in buildings if b.parcel_id == parcel.id) for parcel in parcels
    }
    land_areas = [parcel.area_m2 for parcel in parcels]
    floor_areas = [
        sum(building.footprint_area_m2 * building.floors for building in buildings_by_parcel[p.id])
        for p in parcels
    ]
    existing = [_existing_dwellings(buildings_by_parcel[parcel.id], policy) for parcel in parcels]
    land_total = sum(land_areas)
    floor_total = sum(floor_areas)
    land_shares = [area / max(land_total, 1e-9) for area in land_areas]
    building_shares = (
        [area / floor_total for area in floor_areas] if floor_total > 0 else [0.0 for _ in parcels]
    )
    raw_values = [
        policy.land_value_weight * land_share + policy.building_value_weight * building_share
        for land_share, building_share in zip(
            land_shares,
            building_shares,
            strict=True,
        )
    ]
    raw_total = sum(raw_values)
    valuation_shares = [value / raw_total for value in raw_values]
    allocated_dwellings, returned = _allocate_dwellings(
        replacement_dwellings,
        existing,
        valuation_shares,
        [parcel.id for parcel in parcels],
    )

    return tuple(
        JointRightsAllocation(
            parcel_id=parcel.id,
            building_ids=tuple(b.id for b in buildings_by_parcel[parcel.id]),
            land_area_m2=land_area,
            existing_floor_area_m2=floor_area,
            existing_dwellings=existing_dwellings,
            land_share=land_share,
            building_share=building_share,
            valuation_share=valuation_share,
            allocated_floor_area_m2=step.placement.floor_area_m2 * valuation_share,
            allocated_dwellings=allocated,
            return_dwellings=return_count,
            surplus_dwellings=allocated - return_count,
            permanently_displaced_dwellings=existing_dwellings - return_count,
        )
        for (
            parcel,
            land_area,
            floor_area,
            existing_dwellings,
            land_share,
            building_share,
            valuation_share,
            allocated,
            return_count,
        ) in zip(
            parcels,
            land_areas,
            floor_areas,
            existing,
            land_shares,
            building_shares,
            valuation_shares,
            allocated_dwellings,
            returned,
            strict=True,
        )
    )


def _allocate_dwellings(
    capacity: int,
    existing: list[int],
    shares: list[float],
    ids: list[str],
) -> tuple[list[int], list[int]]:
    returned = [0 for _ in existing]
    remaining = capacity
    priority = sorted(range(len(ids)), key=lambda i: (-shares[i], ids[i]))
    while remaining > 0 and any(returned[i] < existing[i] for i in priority):
        for index in priority:
            if remaining <= 0:
                break
            if returned[index] < existing[index]:
                returned[index] += 1
                remaining -= 1
    surplus = _largest_remainder(remaining, shares, ids)
    return [a + b for a, b in zip(returned, surplus, strict=True)], returned


def _largest_remainder(total: int, weights: list[float], ids: list[str]) -> list[int]:
    if total <= 0:
        return [0 for _ in weights]
    weight_sum = sum(weights)
    normalized = (
        [weight / weight_sum for weight in weights]
        if weight_sum > 0
        else [1.0 / len(weights) for _ in weights]
    )
    raw = [total * weight for weight in normalized]
    result = [math.floor(value) for value in raw]
    remainder = total - sum(result)
    order = sorted(
        range(len(ids)),
        key=lambda i: (-(raw[i] - result[i]), ids[i]),
    )
    for index in order[:remainder]:
        result[index] += 1
    return result


def _project_cost(
    step: JointRenewalStep,
    existing_dwellings: int,
    policy: JointRenewalFeasibilityPolicy,
) -> JointProjectCost:
    demolition = step.candidate.old_floor_area_m2 * policy.demolition_cost_jpy_per_m2
    construction = step.placement.floor_area_m2 * policy.construction_cost_jpy_per_m2
    road_works = step.placement.road_widening_area_m2 * policy.road_work_cost_jpy_per_m2
    open_space_area = max(
        0.0,
        step.placement.allocated_open_space_m2 - step.placement.road_widening_area_m2,
    )
    open_space_works = open_space_area * policy.open_space_cost_jpy_per_m2
    relocation = existing_dwellings * (
        policy.relocation_monthly_support_jpy_per_dwelling * policy.construction_months
        + policy.moving_cost_jpy_per_dwelling
    )
    coordination = len(step.candidate.parcel_ids) * policy.rights_coordination_cost_jpy_per_holder
    hard_cost = demolition + construction + road_works + open_space_works
    design = hard_cost * policy.design_and_management_rate
    before_contingency = hard_cost + relocation + coordination + design
    contingency = before_contingency * policy.contingency_rate
    return JointProjectCost(
        demolition_jpy=demolition,
        construction_jpy=construction,
        road_works_jpy=road_works,
        open_space_works_jpy=open_space_works,
        relocation_jpy=relocation,
        rights_coordination_jpy=coordination,
        design_and_management_jpy=design,
        contingency_jpy=contingency,
        hard_cost_jpy=hard_cost,
        total_jpy=before_contingency + contingency,
    )


def _project_id(step: JointRenewalStep) -> str:
    return f"{step.placement.joint_parcel_id}:{step.placement.variant}"


def _validate_policy(policy: JointRenewalFeasibilityPolicy) -> None:
    if policy.start_year is not None and policy.start_year < 1900:
        raise ValueError("start_year must be at least 1900")
    if policy.years_per_phase < 1:
        raise ValueError("years_per_phase must be at least one")
    if policy.construction_months < 1:
        raise ValueError("construction_months must be at least one")
    if policy.existing_dwellings_per_residential_building < 0:
        raise ValueError("existing dwelling assumption must be non-negative")
    if not 0 < policy.residential_efficiency <= 1:
        raise ValueError("residential_efficiency must be in (0, 1]")
    if policy.replacement_dwelling_area_m2 <= 0:
        raise ValueError("replacement_dwelling_area_m2 must be positive")
    if policy.land_value_weight < 0 or policy.building_value_weight < 0:
        raise ValueError("rights-conversion weights must be non-negative")
    if policy.land_value_weight + policy.building_value_weight <= 0:
        raise ValueError("at least one rights-conversion weight must be positive")
    non_negative = (
        policy.construction_cost_jpy_per_m2,
        policy.demolition_cost_jpy_per_m2,
        policy.road_work_cost_jpy_per_m2,
        policy.open_space_cost_jpy_per_m2,
        policy.relocation_monthly_support_jpy_per_dwelling,
        policy.moving_cost_jpy_per_dwelling,
        policy.rights_coordination_cost_jpy_per_holder,
    )
    if any(value < 0 for value in non_negative):
        raise ValueError("cost assumptions must be non-negative")
    if not 0 <= policy.design_and_management_rate <= 1:
        raise ValueError("design_and_management_rate must be in [0, 1]")
    if not 0 <= policy.contingency_rate <= 1:
        raise ValueError("contingency_rate must be in [0, 1]")
