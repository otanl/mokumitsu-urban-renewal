from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mokumitsu import generate_mokumitsu  # noqa: E402
from mokumitsu.fire import FireScenario  # noqa: E402
from mokumitsu.renewal import (  # noqa: E402
    RenewalPolicy,
    simulate_renewal_trajectory,
)
from mokumitsu.wind import (  # noqa: E402
    SummerWindScenario,
    WindDirection,
    add_wind_to_trajectory,
    district_wind_masks,
    evaluate_district_wind,
    predict_directional_wind,
    weighted_world_wind_ratio,
)


class _FakeModel:
    ny = 32
    nx = 32

    def reference_speed(self):
        return 1.0

    def predict(self, heightmap):
        speed = np.broadcast_to(
            np.linspace(0.1, 1.6, self.nx, dtype=np.float32),
            (self.ny, self.nx),
        ).copy()
        speed[heightmap > 0] = -0.2
        return speed


@pytest.fixture(scope="module")
def district():
    return generate_mokumitsu(seed=0)


def test_analysis_masks_cover_distinct_outdoor_zones(district):
    masks = district_wind_masks(district, res=32, building_edge_distance_m=3.0)
    assert set(masks) == {
        "all_outdoor",
        "roads",
        "parcel_open_space",
        "building_edge",
    }
    assert all(mask.shape == (32, 32) and mask.dtype == np.bool_ for mask in masks.values())
    assert all(np.any(mask) for mask in masks.values())
    assert not np.any(masks["roads"] & masks["parcel_open_space"])
    assert np.all(masks["roads"] <= masks["all_outdoor"])
    assert np.all(masks["parcel_open_space"] <= masks["all_outdoor"])
    assert np.all(masks["building_edge"] <= masks["all_outdoor"])


def test_cardinal_wind_rose_reports_weak_and_strong_wind(district):
    scenario = SummerWindScenario(directions=(WindDirection(0.0, 3.0), WindDirection(90.0, 1.0)))
    result = evaluate_district_wind(district, scenario=scenario, model=_FakeModel())
    assert result.resolution == 32
    assert [direction.weight for direction in result.directions] == pytest.approx([0.75, 0.25])
    assert set(result.weighted_zones) == {
        "all_outdoor",
        "roads",
        "parcel_open_space",
        "building_edge",
    }
    for zone in result.weighted_zones.values():
        assert zone.cells > 0
        assert 0 <= zone.dead_fraction <= 1
        assert 0 <= zone.strong_fraction <= 1
        assert (
            zone.dead_fraction + zone.strong_fraction + zone.comfortable_fraction
            == pytest.approx(1.0)
        )
    outdoor = result.weighted_zones["all_outdoor"]
    assert outdoor.dead_fraction > 0
    assert outdoor.strong_fraction > 0


def test_directional_prediction_is_in_evaluation_frame_and_clips_solids(district):
    speed, heightmap, masks, u0 = predict_directional_wind(
        district, direction_deg=270.0, model=_FakeModel()
    )
    assert speed.shape == heightmap.shape == (32, 32)
    assert masks["roads"].shape == (32, 32)
    assert u0 == pytest.approx(1.0)
    assert np.all(speed >= 0)
    assert np.all(speed[heightmap > 0] == 0)


def test_weighted_world_wind_ratio_rotates_each_direction_back(district):
    scenario = SummerWindScenario(directions=(WindDirection(0.0, 3.0), WindDirection(90.0, 1.0)))
    ratio, heightmap, u0 = weighted_world_wind_ratio(
        district, scenario=scenario, model=_FakeModel()
    )
    speed_0, _, _, _ = predict_directional_wind(district, 0.0, _FakeModel())
    speed_90, _, _, _ = predict_directional_wind(district, 90.0, _FakeModel())
    expected = 0.75 * speed_0 + 0.25 * np.rot90(speed_90, k=-1)
    assert ratio.shape == heightmap.shape == (32, 32)
    assert u0 == pytest.approx(1.0)
    assert ratio == pytest.approx(expected)


def test_invalid_direction_and_non_square_site_are_rejected(district):
    with pytest.raises(ValueError, match="cardinal"):
        evaluate_district_wind(
            district,
            SummerWindScenario(directions=(WindDirection(45.0),)),
            _FakeModel(),
        )
    with pytest.raises(ValueError, match="physically square"):
        evaluate_district_wind(replace(district, width_m=120.0), model=_FakeModel())


def test_wind_metrics_attach_to_every_renewal_phase_without_changing_geometry(district):
    trajectory = simulate_renewal_trajectory(
        district,
        RenewalPolicy(phases=1, rebuild_fraction_per_phase=0.05),
        FireScenario(runs=6, horizon_min=45, seed=2),
    )
    updated = add_wind_to_trajectory(trajectory, model=_FakeModel())
    assert updated.policy == trajectory.policy
    assert len(updated.phases) == len(trajectory.phases)
    for before, after in zip(trajectory.phases, updated.phases, strict=True):
        assert after.district is before.district
        assert before.metrics.wind_mean_index is None
        assert after.metrics.wind_mean_index is not None
        assert 0 <= after.metrics.wind_dead_fraction <= 1
        assert 0 <= after.metrics.wind_strong_fraction <= 1
        assert 0 <= after.metrics.road_dead_fraction <= 1
        assert 0 <= after.metrics.yard_dead_fraction <= 1
        assert 0 <= after.metrics.building_edge_strong_fraction <= 1
