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
    district_wind_base_masks,
    district_wind_masks,
    evaluate_district_wind,
    evaluate_district_wind_with_field,
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


def test_cached_road_and_parcel_masks_preserve_wind_metrics(district):
    base = district_wind_base_masks(district, 32)
    direct_masks = district_wind_masks(district, 32)
    cached_masks = district_wind_masks(district, 32, base_masks=base)
    assert all(np.array_equal(direct_masks[name], cached_masks[name]) for name in direct_masks)

    scenario = SummerWindScenario(directions=(WindDirection(0.0, 0.7), WindDirection(90.0, 0.3)))
    direct = evaluate_district_wind(district, scenario, _FakeModel())
    cached = evaluate_district_wind(district, scenario, _FakeModel(), base_masks=base)
    assert cached.to_dict() == direct.to_dict()


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

    metrics, combined, combined_heightmap = evaluate_district_wind_with_field(
        district,
        scenario,
        _FakeModel(),
    )
    assert combined == pytest.approx(ratio)
    assert combined_heightmap == pytest.approx(heightmap)
    assert (
        metrics.to_dict()
        == evaluate_district_wind(
            district,
            scenario,
            _FakeModel(),
        ).to_dict()
    )


def test_invalid_direction_and_non_square_site_are_rejected(district):
    with pytest.raises(ValueError, match="cardinal"):
        evaluate_district_wind(
            district,
            SummerWindScenario(directions=(WindDirection(45.0),)),
            _FakeModel(),
        )
    with pytest.raises(ValueError, match="physically square"):
        evaluate_district_wind(replace(district, width_m=120.0), model=_FakeModel())


def test_real_model_metadata_must_match_the_physical_scenario(district):
    scenario = SummerWindScenario()
    model = _FakeModel()
    model.meta = {
        "physics": {
            "contract_version": 1,
            "height_encoding": "fraction_of_domain_height",
            "domain_length_x_m": district.width_m,
            "domain_length_y_m": district.height_m,
            "domain_height_m": scenario.domain_height_m,
            "reference_height_m": scenario.reference_height_m,
            "pedestrian_height_m": scenario.pedestrian_height_m,
            "backend_signature": "unit-test-backend",
            "output_grid": [model.ny, model.nx],
            "grid_verification": {
                "passed": True,
                "compatible": True,
                "override": False,
            },
        },
        "dataset_sha256": "a" * 64,
        "split": {
            "train_indices": [0],
            "validation_indices": [1],
            "test_indices": [2],
        },
    }
    evaluate_district_wind(district, scenario, model)

    model.meta["physics"]["domain_height_m"] = 40.0
    with pytest.raises(ValueError, match="does not match scenario"):
        evaluate_district_wind(district, scenario, model)
    model.meta = {}
    with pytest.raises(ValueError, match="lacks the physical wind contract"):
        evaluate_district_wind(district, scenario, model)


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
