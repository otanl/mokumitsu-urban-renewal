from __future__ import annotations

import json

import pytest

from mokumitsu import generate_mokumitsu, validate_district  # noqa: E402
from mokumitsu.fire import FireScenario  # noqa: E402
from mokumitsu.renewal import (  # noqa: E402
    RenewalPolicy,
    save_renewal_trajectory,
    simulate_renewal_trajectory,
)


@pytest.fixture(scope="module")
def trajectory():
    district = generate_mokumitsu(seed=0)
    policy = RenewalPolicy(phases=3, years_per_phase=5, rebuild_fraction_per_phase=0.08)
    scenario = FireScenario(runs=24, horizon_min=120, seed=4)
    return simulate_renewal_trajectory(district, policy, scenario)


def test_renewal_is_phased_valid_and_capacity_limited(trajectory):
    assert len(trajectory.phases) == 4
    completed = []
    previous_total = 0
    for phase in trajectory.phases:
        assert validate_district(phase.district) == ()
        assert phase.metrics.renewed_total >= previous_total
        previous_total = phase.metrics.renewed_total
        completed.extend(a.building_id for a in phase.actions if a.status == "completed")
    assert len(completed) == len(set(completed))
    assert completed
    assert any(a.status == "deferred" for p in trajectory.phases for a in p.actions)


def test_individual_only_baseline_does_not_fake_access_for_landlocked_parcels(trajectory):
    initial = trajectory.phases[0].district
    final = trajectory.phases[-1].district
    initial_buildings = {b.id: b for b in initial.buildings}
    for building in final.buildings:
        if not initial.access(building.parcel_id).individual_rebuildable:
            assert building == initial_buildings[building.id]


def test_floor_area_and_fire_resistance_improve_without_consuming_open_space(trajectory):
    metrics = [phase.metrics for phase in trajectory.phases]
    assert all(
        a.gross_floor_area_m2 <= b.gross_floor_area_m2
        for a, b in zip(metrics, metrics[1:], strict=False)
    )
    assert all(
        a.non_fire_resistant_rate >= b.non_fire_resistant_rate
        for a, b in zip(metrics, metrics[1:], strict=False)
    )
    assert all(
        m.footprint_coverage == pytest.approx(metrics[0].footprint_coverage) for m in metrics
    )
    assert metrics[-1].gross_floor_area_m2 > metrics[0].gross_floor_area_m2
    assert metrics[-1].expected_burned_buildings <= metrics[0].expected_burned_buildings


def test_trajectory_json_defaults_to_metrics_not_duplicated_geometry(trajectory, tmp_path):
    path = save_renewal_trajectory(trajectory, tmp_path / "trajectory.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["phases"]) == 4
    assert "district" not in data["phases"][0]
    assert data["phases"][-1]["metrics"]["renewed_total"] > 0


def test_invalid_policy_is_rejected():
    district = generate_mokumitsu(seed=1)
    with pytest.raises(ValueError):
        simulate_renewal_trajectory(
            district,
            RenewalPolicy(phases=0),
            FireScenario(runs=5, horizon_min=30),
        )
