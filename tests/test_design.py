from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from mokumitsu import generate_mokumitsu, validate_district
from mokumitsu.design import (
    evaluate_joint_design,
    joint_design_cache_key,
    prepare_joint_design_baseline,
)
from mokumitsu.fire import FireScenario
from mokumitsu.joint_renewal import (
    JointRenewalPolicy,
    default_joint_design_parameters,
    joint_renewal_candidates,
    parameterized_joint_redevelopment,
)
from mokumitsu.wind import SummerWindScenario


class _FakeModel:
    ny = 32
    nx = 32

    def reference_speed(self):
        return 1.0

    def predict(self, heightmap):
        speed = np.broadcast_to(
            np.linspace(0.25, 1.20, self.nx, dtype=np.float32),
            (self.ny, self.nx),
        ).copy()
        speed -= 0.12 * np.roll(heightmap, 2, axis=1)
        speed[heightmap > 0] = 0.0
        return np.maximum(speed, 0.0)


@pytest.fixture(scope="module")
def setup():
    district = generate_mokumitsu(seed=0)
    policy = JointRenewalPolicy(
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
        candidate_limit=4,
        placement_grid=5,
        placement_variants=3,
    )
    candidate = joint_renewal_candidates(district, policy)[0]
    parameters = default_joint_design_parameters(district, candidate, policy)
    return district, policy, candidate, parameters


def test_default_parameters_round_trip_to_a_valid_design(setup):
    district, policy, candidate, parameters = setup
    updated, placement = parameterized_joint_redevelopment(
        district,
        candidate,
        parameters,
        policy,
    )

    assert validate_district(updated) == ()
    assert placement.variant == "interactive"
    assert placement.local_building_coverage == pytest.approx(parameters.building_coverage)
    assert placement.floor_area_m2 == pytest.approx(candidate.old_floor_area_m2)
    assert updated.access(placement.joint_parcel_id).individual_rebuildable


def test_explicit_parameters_are_not_silently_repaired(setup):
    district, policy, candidate, parameters = setup
    with pytest.raises(ValueError, match="center_u"):
        parameterized_joint_redevelopment(
            district,
            candidate,
            replace(parameters, center_u=-0.1),
            policy,
        )


def test_design_evaluation_reuses_baseline_and_is_serializable(setup):
    district, policy, candidate, parameters = setup
    model = _FakeModel()
    wind = SummerWindScenario()
    fire = FireScenario(runs=4, horizon_min=45, seed=7)
    baseline = prepare_joint_design_baseline(
        district,
        wind,
        fire,
        model,
        model_fingerprint="fake-v1",
    )
    result = evaluate_joint_design(
        district,
        candidate,
        parameters,
        policy,
        wind,
        fire,
        model,
        baseline=baseline,
        model_fingerprint="fake-v1",
    )

    assert result.baseline is baseline
    assert result.selected_floor_area_retention == pytest.approx(1.0)
    assert result.objectives.rights_count == candidate.parcel_count
    assert result.objectives.access_poor_resolved >= 1
    assert len(result.cache_key) == 64
    payload = result.to_dict(include_district=True)
    assert payload["cache_key"] == result.cache_key
    assert payload["district"]["seed"] == district.seed
    json.dumps(payload, allow_nan=False)


def test_cache_key_changes_with_the_massing_controls(setup):
    district, policy, candidate, parameters = setup
    model = _FakeModel()
    wind = SummerWindScenario()
    fire = FireScenario(runs=4, seed=0)
    first = joint_design_cache_key(
        district,
        candidate,
        parameters,
        policy,
        wind,
        fire,
        model,
        model_fingerprint="fake-v1",
    )
    second = joint_design_cache_key(
        district,
        candidate,
        replace(parameters, building_coverage=parameters.building_coverage * 0.9),
        policy,
        wind,
        fire,
        model,
        model_fingerprint="fake-v1",
    )
    repeated = joint_design_cache_key(
        district,
        candidate,
        parameters,
        policy,
        wind,
        fire,
        model,
        model_fingerprint="fake-v1",
    )

    assert first == repeated
    assert first != second


def test_mismatched_baseline_is_rejected(setup):
    district, policy, candidate, parameters = setup
    model = _FakeModel()
    wind = SummerWindScenario()
    fire = FireScenario(runs=2, seed=1)
    other_baseline = prepare_joint_design_baseline(
        generate_mokumitsu(seed=1),
        wind,
        fire,
        model,
        model_fingerprint="fake-v1",
    )
    with pytest.raises(ValueError, match="baseline does not match"):
        evaluate_joint_design(
            district,
            candidate,
            parameters,
            policy,
            wind,
            fire,
            model,
            baseline=other_baseline,
            model_fingerprint="fake-v1",
        )
