from __future__ import annotations

import numpy as np
import pytest

from mokumitsu import generate_mokumitsu, validate_district  # noqa: E402
from mokumitsu.fire import FireScenario  # noqa: E402
from mokumitsu.joint_renewal import JointRenewalPolicy  # noqa: E402
from mokumitsu.pareto import (  # noqa: E402
    JointRenewalObjectives,
    ParetoRenewalPolicy,
    evaluate_joint_renewal_pareto,
    pareto_front,
)
from mokumitsu.wind import SummerWindScenario  # noqa: E402


class _FakeModel:
    ny = 32
    nx = 32

    def reference_speed(self):
        return 1.0

    def predict(self, heightmap):
        speed = np.broadcast_to(
            np.linspace(0.2, 1.2, self.nx, dtype=np.float32),
            (self.ny, self.nx),
        ).copy()
        speed -= 0.25 * np.roll(heightmap, 2, axis=1)
        speed[heightmap > 0] = 0.0
        return speed


def _objectives(wind, fire, footprint, rights, access, open_space=0.0):
    return JointRenewalObjectives(
        wind,
        fire,
        footprint,
        rights,
        access,
        connected_open_space_m2=open_space,
    )


def test_pareto_front_removes_dominated_alternatives():
    dominated = _objectives(1.0, 100.0, 0.30, 2, 1)
    better = _objectives(0.9, 90.0, 0.29, 2, 1)
    tradeoff = _objectives(0.8, 120.0, 0.28, 3, 2)
    assert pareto_front((dominated, better, tradeoff)) == (1, 2)
    more_connected_space = _objectives(0.9, 90.0, 0.29, 2, 1, open_space=20.0)
    assert pareto_front((better, more_connected_space)) == (1,)


def test_cluster_pareto_evaluation_is_valid_and_serializable():
    district = generate_mokumitsu(seed=0)
    result = evaluate_joint_renewal_pareto(
        district,
        joint_policy=JointRenewalPolicy(
            minimum_cluster_parcels=2,
            maximum_cluster_parcels=4,
            candidate_limit=3,
            placement_grid=5,
            placement_variants=1,
        ),
        pareto_policy=ParetoRenewalPolicy(candidates_per_cluster_size=1),
        wind_scenario=SummerWindScenario(),
        fire_scenario=FireScenario(runs=4, horizon_min=45, seed=3),
        model=_FakeModel(),
    )
    assert result.alternatives
    assert result.pareto_indices
    assert result.recommended_index in result.pareto_indices
    assert [alternative.index for alternative in result.alternatives] == list(
        range(len(result.alternatives))
    )
    for alternative in result.alternatives:
        assert 2 <= alternative.candidate.parcel_count <= 4
        assert validate_district(alternative.district) == ()
        assert alternative.floor_area_retention == pytest.approx(1.0)
        assert np.isfinite(alternative.balanced_score)
        assert alternative.objectives.access_poor_resolved >= 1
        assert alternative.objectives.connected_open_space_m2 >= 0
    assert any(
        alternative.objectives.connected_open_space_m2 > 0 for alternative in result.alternatives
    )

    data = result.to_dict()
    assert data["recommended_index"] in data["pareto_indices"]
    assert all("district" not in alternative for alternative in data["alternatives"])
    assert all(alternative["placement"]["open_spaces"] for alternative in data["alternatives"])
    included = result.to_dict(include_recommended_district=True)["alternatives"]
    assert sum("district" in alternative for alternative in included) == 1


def test_invalid_pareto_weights_are_rejected():
    with pytest.raises(ValueError, match="weights"):
        evaluate_joint_renewal_pareto(
            generate_mokumitsu(seed=0),
            pareto_policy=ParetoRenewalPolicy(
                wind_weight=0,
                fire_weight=0,
                footprint_weight=0,
                open_space_weight=0,
                rights_weight=0,
                access_weight=0,
            ),
            model=_FakeModel(),
        )
