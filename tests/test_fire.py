from __future__ import annotations

from dataclasses import replace

import pytest

from mokumitsu import (  # noqa: E402
    DistrictBuilding,
    MokumitsuDistrict,
    Parcel,
    generate_mokumitsu,
)
from mokumitsu.fire import (  # noqa: E402
    FireScenario,
    build_fire_graph,
    fire_aware_renewal_priorities,
    simulate_fire_spread,
)


def _two_building_district(target_resistance="none"):
    parcels = (
        Parcel("p0", "b0", ((0, 0), (9, 0), (9, 10), (0, 10))),
        Parcel("p1", "b0", ((10, 0), (19, 0), (19, 10), (10, 10))),
    )
    buildings = (
        DistrictBuilding("a", "p0", 4.5, 5.0, 8.0, 8.0, 5.6, 2, 1970, "timber", "none"),
        DistrictBuilding("b", "p1", 14.5, 5.0, 8.0, 8.0, 5.6, 2, 1970, "timber", target_resistance),
    )
    return MokumitsuDistrict(20.0, 10.0, 0, 2026, (), parcels, buildings)


def test_graph_is_directional_under_wind():
    district = _two_building_district()
    scenario = FireScenario(wind_direction_deg=0, wind_speed_m_s=6, runs=10)
    edges = build_fire_graph(district, scenario)
    downwind = next(e for e in edges if e.source_id == "a")
    upwind = next(e for e in edges if e.source_id == "b")
    assert downwind.mean_delay_min < upwind.mean_delay_min
    assert downwind.gap_m == pytest.approx(2.0)


def test_simulation_is_deterministic_and_bounded():
    district = generate_mokumitsu(seed=0)
    scenario = FireScenario(runs=80, seed=4, horizon_min=120)
    a = simulate_fire_spread(district, scenario)
    b = simulate_fire_spread(district, scenario)
    assert a == b
    assert len(a.burn_probability) == len(district.buildings)
    assert all(0 <= p <= 1 for p in a.burn_probability)
    assert all(0 < c <= 1 for c in a.source_consequence)
    assert 1 <= a.expected_burned_buildings <= len(district.buildings)
    assert 0 <= a.probability_large_fire <= 1


def test_fire_resistant_target_reduces_ignition_probability():
    scenario = FireScenario(runs=1200, seed=2, horizon_min=60, wind_speed_m_s=4)
    ordinary = simulate_fire_spread(_two_building_district("none"), scenario, ("a",))
    resistant = simulate_fire_spread(_two_building_district("fire_resistant"), scenario, ("a",))
    assert resistant.probability_for("b") < ordinary.probability_for("b")


def test_fixed_ignition_and_result_serialization():
    district = generate_mokumitsu(seed=3)
    result = simulate_fire_spread(
        district, FireScenario(runs=30, seed=1, horizon_min=90), ("building_0",)
    )
    data = result.to_dict()
    assert data["burn_probability"]["building_0"] == 1.0
    assert set(data["burn_probability"]) == {b.id for b in district.buildings}
    assert data["scenario"]["wind_direction_deg"] == 0.0


def test_fire_aware_priority_is_sorted_and_preserves_feasibility():
    district = generate_mokumitsu(seed=0)
    result = simulate_fire_spread(district, FireScenario(runs=50, seed=0, horizon_min=120))
    ranked = fire_aware_renewal_priorities(district, result, fire_weight=0.4)
    assert [x.rank for x in ranked] == list(range(1, len(ranked) + 1))
    assert all(
        a.combined_score >= b.combined_score for a, b in zip(ranked, ranked[1:], strict=False)
    )
    assert any(x.feasibility_score < 0.5 for x in ranked[:20])
    assert any("high_fire_exposure" in x.reasons for x in ranked)


def test_invalid_inputs_are_rejected():
    district = generate_mokumitsu(seed=0)
    with pytest.raises(ValueError):
        simulate_fire_spread(district, FireScenario(runs=0))
    with pytest.raises(ValueError):
        simulate_fire_spread(district, FireScenario(runs=1), ("missing",))
    result = simulate_fire_spread(district, FireScenario(runs=2))
    with pytest.raises(ValueError):
        fire_aware_renewal_priorities(district, result, fire_weight=1.2)
    other = replace(district, buildings=district.buildings[:-1])
    with pytest.raises(ValueError):
        fire_aware_renewal_priorities(other, result)
