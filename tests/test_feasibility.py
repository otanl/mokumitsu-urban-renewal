from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from mokumitsu import (  # noqa: E402
    JointRenewalFeasibilityPolicy,
    evaluate_joint_renewal_feasibility,
    generate_mokumitsu,
    save_joint_renewal_feasibility,
    validate_district,
)
from mokumitsu.joint_renewal import (  # noqa: E402
    JointRenewalPolicy,
    optimize_joint_renewal,
)


class _FakeModel:
    ny = 32
    nx = 32

    def reference_speed(self):
        return 1.0

    def predict(self, heightmap):
        speed = np.broadcast_to(
            np.linspace(0.15, 1.35, self.nx, dtype=np.float32),
            (self.ny, self.nx),
        ).copy()
        speed[heightmap > 0] = 0.0
        return speed


@pytest.fixture(scope="module")
def district():
    return generate_mokumitsu(seed=0)


@pytest.fixture(scope="module")
def joint_plan(district):
    return optimize_joint_renewal(
        district,
        policy=JointRenewalPolicy(
            projects=2,
            candidate_limit=8,
            placement_grid=5,
            placement_variants=2,
            max_wind_objective_increase=1.0,
        ),
        model=_FakeModel(),
    )


@pytest.fixture(scope="module")
def feasibility(district, joint_plan):
    return evaluate_joint_renewal_feasibility(district, joint_plan)


def test_phases_conserve_rights_floor_area_and_dwelling_accounts(
    district,
    joint_plan,
    feasibility,
):
    assert len(joint_plan.steps) == len(feasibility.phases) == 2
    assert [phase.year for phase in feasibility.phases] == [2036, 2039]
    assert validate_district(feasibility.final_district) == ()
    assert feasibility.final_district.summary().gross_floor_area_m2 == pytest.approx(
        district.summary().gross_floor_area_m2
    )

    seen_parcels = set()
    for step, phase in zip(joint_plan.steps, feasibility.phases, strict=True):
        assert phase.rights_count == len(step.candidate.parcel_ids)
        assert phase.open_spaces == step.placement.open_spaces
        assert not seen_parcels.intersection(step.candidate.parcel_ids)
        seen_parcels.update(step.candidate.parcel_ids)
        assert sum(right.land_share for right in phase.rights) == pytest.approx(1.0)
        assert sum(right.building_share for right in phase.rights) == pytest.approx(1.0)
        assert sum(right.valuation_share for right in phase.rights) == pytest.approx(1.0)
        assert sum(right.allocated_floor_area_m2 for right in phase.rights) == pytest.approx(
            step.placement.floor_area_m2
        )
        assert sum(right.allocated_dwellings for right in phase.rights) == (
            phase.replacement_dwellings
        )
        assert phase.return_dwellings + phase.surplus_dwellings == (phase.replacement_dwellings)
        assert (
            phase.return_dwellings + phase.permanently_displaced_dwellings
            == phase.existing_dwellings
        )
        assert phase.temporary_relocation_dwellings == phase.existing_dwellings
        assert phase.temporary_relocation_dwelling_months == (
            phase.existing_dwellings * feasibility.policy.construction_months
        )
        joint_building = next(
            building
            for building in phase.district.buildings
            if building.id == phase.joint_building_id
        )
        assert joint_building.built_year == phase.year

    initial_dwellings = sum(building.use == "residential" for building in district.buildings)
    assert feasibility.summary.initial_dwellings == initial_dwellings
    assert feasibility.summary.final_dwellings == (
        initial_dwellings + feasibility.summary.net_dwellings
    )
    assert feasibility.summary.rights_converted == sum(
        phase.rights_count for phase in feasibility.phases
    )
    assert feasibility.summary.permanently_displaced_dwellings == 0


def test_cost_components_and_cumulative_metrics_are_consistent(feasibility):
    previous_cost = 0.0
    previous_rights = 0
    previous_months = 0
    for phase in feasibility.phases:
        cost = phase.cost
        assert cost.hard_cost_jpy == pytest.approx(
            cost.demolition_jpy
            + cost.construction_jpy
            + cost.road_works_jpy
            + cost.open_space_works_jpy
        )
        assert cost.total_jpy == pytest.approx(
            cost.hard_cost_jpy
            + cost.relocation_jpy
            + cost.rights_coordination_jpy
            + cost.design_and_management_jpy
            + cost.contingency_jpy
        )
        assert phase.cumulative_cost_jpy == pytest.approx(previous_cost + cost.total_jpy)
        assert phase.cumulative_rights_converted == previous_rights + phase.rights_count
        assert phase.cumulative_relocation_dwelling_months == (
            previous_months + phase.temporary_relocation_dwelling_months
        )
        previous_cost = phase.cumulative_cost_jpy
        previous_rights = phase.cumulative_rights_converted
        previous_months = phase.cumulative_relocation_dwelling_months

    assert feasibility.summary.total_cost_jpy == pytest.approx(previous_cost)
    assert feasibility.summary.relocation_dwelling_months == previous_months
    assert feasibility.summary.peak_temporary_relocation_dwellings == max(
        phase.temporary_relocation_dwellings for phase in feasibility.phases
    )


def test_insufficient_replacement_capacity_reports_permanent_displacement(
    district,
    joint_plan,
):
    result = evaluate_joint_renewal_feasibility(
        district,
        joint_plan,
        JointRenewalFeasibilityPolicy(
            start_year=2040,
            years_per_phase=4,
            replacement_dwelling_area_m2=100_000.0,
        ),
    )
    assert [phase.year for phase in result.phases] == [2040, 2044]
    assert all(phase.replacement_dwellings == 0 for phase in result.phases)
    assert result.summary.permanently_displaced_dwellings == sum(
        phase.existing_dwellings for phase in result.phases
    )
    assert result.summary.final_dwellings == (
        result.summary.initial_dwellings - result.summary.permanently_displaced_dwellings
    )


def test_serialization_is_bounded_and_can_include_districts(
    feasibility,
    tmp_path,
):
    compact = feasibility.to_dict()
    assert "initial_district" not in compact
    assert "district" not in compact["phases"][0]
    assert compact["summary"]["projects"] == 2
    assert compact["phases"][0]["rights"]
    assert compact["phases"][0]["open_spaces"]
    assert compact["phases"][0]["cost"]["total_jpy"] > 0

    expanded = feasibility.to_dict(include_districts=True)
    assert "initial_district" in expanded
    assert "final_district" in expanded
    assert "district" in expanded["phases"][0]

    output = save_joint_renewal_feasibility(
        feasibility,
        tmp_path / "feasibility.json",
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["summary"] == compact["summary"]


@pytest.mark.parametrize(
    "policy",
    [
        JointRenewalFeasibilityPolicy(start_year=1899),
        JointRenewalFeasibilityPolicy(years_per_phase=0),
        JointRenewalFeasibilityPolicy(construction_months=0),
        JointRenewalFeasibilityPolicy(residential_efficiency=0),
        JointRenewalFeasibilityPolicy(replacement_dwelling_area_m2=0),
        JointRenewalFeasibilityPolicy(land_value_weight=0, building_value_weight=0),
        JointRenewalFeasibilityPolicy(construction_cost_jpy_per_m2=-1),
        JointRenewalFeasibilityPolicy(design_and_management_rate=1.1),
        JointRenewalFeasibilityPolicy(contingency_rate=-0.1),
    ],
)
def test_invalid_policy_is_rejected(district, joint_plan, policy):
    with pytest.raises(ValueError):
        evaluate_joint_renewal_feasibility(district, joint_plan, policy)


def test_policy_replacement_does_not_mutate_shared_fixture(feasibility):
    altered = replace(feasibility.policy, construction_months=24)
    assert altered.construction_months == 24
    assert feasibility.policy.construction_months == 18
