from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from mokumitsu import generate_mokumitsu, validate_district  # noqa: E402
from mokumitsu.geometry import rect_corners  # noqa: E402
from mokumitsu.joint_renewal import (  # noqa: E402
    JointRenewalPolicy,
    joint_redevelopment_variants,
    joint_renewal_candidates,
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


def test_candidates_pair_access_poor_and_access_capable_contiguous_parcels(district):
    candidates = joint_renewal_candidates(district)
    assert candidates
    for candidate in candidates:
        poor, capable = (district.parcel(parcel_id) for parcel_id in candidate.parcel_ids)
        assert not district.access(poor.id).individual_rebuildable
        assert district.access(capable.id).individual_rebuildable
        assert poor.block_id == capable.block_id
        shared = (
            Polygon(poor.polygon).boundary.intersection(Polygon(capable.polygon).boundary).length
        )
        assert shared == pytest.approx(candidate.shared_boundary_m)
        assert shared >= JointRenewalPolicy().shared_boundary_min_m


def test_joint_variants_preserve_floor_area_reduce_footprint_and_inherit_real_access(district):
    candidate = joint_renewal_candidates(district)[0]
    variants = joint_redevelopment_variants(district, candidate)
    assert 1 <= len(variants) <= JointRenewalPolicy().placement_variants
    initial = district.summary()
    for updated, placement in variants:
        summary = updated.summary()
        assert validate_district(updated) == ()
        assert len(updated.parcels) == len(district.parcels) - 1
        assert len(updated.buildings) == len(district.buildings) - 1
        assert summary.gross_floor_area_m2 == pytest.approx(initial.gross_floor_area_m2)
        assert summary.footprint_coverage < initial.footprint_coverage
        assert placement.floor_area_m2 == pytest.approx(candidate.old_floor_area_m2)
        assert placement.local_building_coverage <= JointRenewalPolicy().target_building_coverage
        assert updated.access(placement.joint_parcel_id).individual_rebuildable
        assert any(space.kind in {"shared_court", "pocket_park"} for space in placement.open_spaces)
        building_shape = Polygon(
            rect_corners(
                placement.cx,
                placement.cy,
                placement.width_m,
                placement.depth_m,
                placement.theta,
            )
        )
        joint_parcel = Polygon(updated.parcel(placement.joint_parcel_id).polygon)
        for space in placement.open_spaces:
            space_shape = Polygon(space.polygon)
            assert space_shape.is_valid
            assert space_shape.area == pytest.approx(space.area_m2)
            assert building_shape.intersection(space_shape).area < 1e-8
            if space.kind != "road_widening":
                assert joint_parcel.buffer(1e-8).covers(space_shape)


def test_land_allocation_widens_a_real_road_and_connects_wind_open_space(district):
    policy = JointRenewalPolicy(
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
        placement_grid=7,
        placement_variants=3,
    )
    for candidate in joint_renewal_candidates(district, policy)[:20]:
        original_shape = unary_union(
            [Polygon(district.parcel(parcel_id).polygon) for parcel_id in candidate.parcel_ids]
        )
        for updated, placement in joint_redevelopment_variants(
            district,
            candidate,
            policy,
            wind_direction_deg=0.0,
        ):
            road_spaces = [
                space for space in placement.open_spaces if space.kind == "road_widening"
            ]
            connected_corridors = [
                space
                for space in placement.open_spaces
                if space.kind == "ventilation_corridor" and space.connected_road_id is not None
            ]
            if not road_spaces or not connected_corridors:
                continue
            road_space = road_spaces[0]
            corridor = connected_corridors[0]
            assert updated.road(road_space.connected_road_id).width_m >= policy.target_road_width_m
            assert updated.road(road_space.connected_road_id).kind == "widened_frontage"
            assert updated.road(road_space.source_road_id).width_m == pytest.approx(
                district.road(road_space.source_road_id).width_m
            )
            assert corridor.orientation_deg == pytest.approx(0.0)
            assert corridor.span_ratio >= policy.minimum_corridor_span_ratio
            assert placement.connected_open_space_m2 >= corridor.area_m2
            joint_shape = Polygon(updated.parcel(placement.joint_parcel_id).polygon)
            assert joint_shape.area == pytest.approx(
                original_shape.area - placement.road_widening_area_m2,
                abs=1e-6,
            )
            return
    pytest.fail("no road-connected wind-corridor alternative was generated")


def test_greedy_plan_is_valid_serializable_and_keeps_capacity(district):
    policy = JointRenewalPolicy(
        projects=1,
        candidate_limit=3,
        placement_grid=5,
        placement_variants=2,
        max_wind_objective_increase=1.0,
    )
    plan = optimize_joint_renewal(district, policy=policy, model=_FakeModel())
    assert len(plan.steps) == 1
    assert validate_district(plan.district) == ()
    assert plan.district.summary().gross_floor_area_m2 == pytest.approx(
        district.summary().gross_floor_area_m2
    )
    assert plan.steps[0].placement.variant in {
        "crosswind_negative",
        "crosswind_positive",
        "central",
    }
    data = plan.to_dict()
    assert "district" not in data
    assert len(data["steps"]) == 1
    assert data["final_summary"]["gross_floor_area_ratio"] == pytest.approx(
        district.summary().gross_floor_area_ratio
    )


def test_invalid_policy_is_rejected(district):
    with pytest.raises(ValueError):
        joint_renewal_candidates(
            district,
            JointRenewalPolicy(target_building_coverage=1.1),
        )


def test_connected_cluster_candidates_cover_two_to_four_real_parcels(district):
    policy = JointRenewalPolicy(
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
    )
    candidates = joint_renewal_candidates(district, policy)
    assert {candidate.parcel_count for candidate in candidates} == {2, 3, 4}

    for candidate in candidates:
        parcels = [district.parcel(parcel_id) for parcel_id in candidate.parcel_ids]
        assert len(parcels) == len(set(candidate.parcel_ids))
        assert len({parcel.block_id for parcel in parcels}) == 1
        assert any(not district.access(parcel.id).individual_rebuildable for parcel in parcels)
        assert any(district.access(parcel.id).individual_rebuildable for parcel in parcels)
        merged = unary_union([Polygon(parcel.polygon) for parcel in parcels])
        assert merged.geom_type == "Polygon"
        assert not merged.interiors

        adjacency = {parcel.id: set() for parcel in parcels}
        for index, first in enumerate(parcels):
            for second in parcels[index + 1 :]:
                shared = (
                    Polygon(first.polygon)
                    .boundary.intersection(Polygon(second.polygon).boundary)
                    .length
                )
                if shared >= policy.shared_boundary_min_m:
                    adjacency[first.id].add(second.id)
                    adjacency[second.id].add(first.id)
        reached = {parcels[0].id}
        frontier = [parcels[0].id]
        while frontier:
            frontier.extend(adjacency[frontier.pop()] - reached)
            reached.update(frontier)
        assert reached == set(candidate.parcel_ids)


def test_four_parcel_variant_keeps_capacity_and_inherits_access(district):
    policy = JointRenewalPolicy(
        minimum_cluster_parcels=4,
        maximum_cluster_parcels=4,
        placement_grid=5,
    )
    candidates = joint_renewal_candidates(district, policy)
    candidate = next(
        candidate
        for candidate in candidates
        if joint_redevelopment_variants(district, candidate, policy)
    )
    updated, placement = joint_redevelopment_variants(district, candidate, policy)[0]
    assert candidate.parcel_count == 4
    assert len(updated.parcels) == len(district.parcels) - 3
    assert len(updated.buildings) == len(district.buildings) - 3
    assert validate_district(updated) == ()
    assert updated.summary().gross_floor_area_m2 == pytest.approx(
        district.summary().gross_floor_area_m2
    )
    assert placement.floor_area_m2 == pytest.approx(candidate.old_floor_area_m2)
    assert updated.access(placement.joint_parcel_id).individual_rebuildable


def test_cluster_policy_rejects_more_than_four_parcels(district):
    with pytest.raises(ValueError, match="cluster parcel limits"):
        joint_renewal_candidates(
            district,
            JointRenewalPolicy(maximum_cluster_parcels=5),
        )


def test_land_allocation_policy_rejects_invalid_geometry_controls(district):
    with pytest.raises(ValueError, match="corridor"):
        joint_renewal_candidates(
            district,
            JointRenewalPolicy(ventilation_corridor_width_m=0),
        )
