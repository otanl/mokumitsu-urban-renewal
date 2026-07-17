"""Joint parcel renewal with floor-area transfer and wind-aware placement.

This is the first discrete intervention beyond parcel-by-parcel rebuilding.  One or more
access-poor parcels are clustered with contiguous, individually rebuildable parcels in the same
block.  Their lots are merged, the existing buildings are replaced by one fire-resistant
mid-rise building, and the original gross floor area is retained while footprint coverage is
reduced.  The default remains the original two-parcel pilot; callers may explicitly search
connected clusters of up to four parcels.  Each placement now carries an explicit land-allocation
plan: a narrow-road dedication where applicable, a wind-aligned open corridor, and a compact
shared court or pocket park.  These polygons are planning reservations rather than decorative
labels; the replacement building is constrained not to occupy them.

The geometry step is deterministic and legally conservative: access is inherited only from
frontages already recorded on the participating parcels.  A small set of valid placements is
searched inside the merged polygon, then the FNO screen selects the option with the best balance
of weak summer wind and strong wind at building edges.  Shortlisted plans still require XLB and
project-specific planning review.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, polylabel, unary_union

from .district import (
    NON_ROAD,
    DistrictBuilding,
    Frontage,
    MokumitsuDistrict,
    Parcel,
    RoadSegment,
    renewal_priorities,
    validate_district,
)
from .model import FnoModel, load_model
from .wind import (
    DistrictWindMetrics,
    SummerWindScenario,
    evaluate_district_wind,
)


@dataclass(frozen=True)
class JointRenewalPolicy:
    """Geometry and search assumptions for a pilot joint-renewal programme."""

    year: int = 2036
    projects: int = 3
    candidate_limit: int = 12
    shared_boundary_min_m: float = 0.35
    minimum_cluster_parcels: int = 2
    maximum_cluster_parcels: int = 2
    target_building_coverage: float = 0.30
    minimum_floors: int = 3
    maximum_floors: int = 6
    floor_height_m: float = 3.0
    parcel_setback_m: float = 0.25
    target_road_width_m: float = 6.0
    ventilation_corridor_width_m: float = 2.0
    minimum_corridor_span_ratio: float = 0.55
    shared_open_space_fraction: float = 0.10
    minimum_shared_open_space_m2: float = 8.0
    placement_grid: int = 9
    placement_variants: int = 3
    max_wind_objective_increase: float = 0.01


@dataclass(frozen=True)
class JointRenewalCandidate:
    parcel_ids: tuple[str, ...]
    building_ids: tuple[str, ...]
    block_id: str
    shared_boundary_m: float
    merged_area_m2: float
    old_footprint_m2: float
    old_floor_area_m2: float
    priority_score: float
    reasons: tuple[str, ...]

    @property
    def parcel_count(self) -> int:
        return len(self.parcel_ids)


@dataclass(frozen=True)
class JointOpenSpace:
    """A polygon reserved by one joint-renewal land-allocation alternative."""

    id: str
    kind: str
    polygon: tuple[tuple[float, float], ...]
    area_m2: float
    connected_road_id: str | None = None
    source_road_id: str | None = None
    orientation_deg: float | None = None
    span_ratio: float | None = None


@dataclass(frozen=True)
class JointBuildingPlacement:
    variant: str
    joint_parcel_id: str
    joint_building_id: str
    cx: float
    cy: float
    width_m: float
    depth_m: float
    theta: float
    floors: int
    height_m: float
    footprint_m2: float
    floor_area_m2: float
    local_building_coverage: float
    open_spaces: tuple[JointOpenSpace, ...] = ()

    @property
    def allocated_open_space_m2(self) -> float:
        return float(sum(space.area_m2 for space in self.open_spaces))

    @property
    def road_widening_area_m2(self) -> float:
        return float(
            sum(space.area_m2 for space in self.open_spaces if space.kind == "road_widening")
        )

    @property
    def connected_open_space_m2(self) -> float:
        return float(
            sum(
                space.area_m2
                for space in self.open_spaces
                if space.kind != "road_widening" and space.connected_road_id is not None
            )
        )


@dataclass(frozen=True)
class JointRenewalStep:
    index: int
    candidate: JointRenewalCandidate
    placement: JointBuildingPlacement
    wind_objective_before: float
    wind_objective_after: float
    outdoor_dead_before: float
    outdoor_dead_after: float
    yard_dead_before: float
    yard_dead_after: float
    edge_strong_before: float
    edge_strong_after: float


@dataclass(frozen=True)
class JointRenewalPlan:
    policy: JointRenewalPolicy
    district: MokumitsuDistrict
    steps: tuple[JointRenewalStep, ...]
    initial_wind: DistrictWindMetrics
    final_wind: DistrictWindMetrics

    def to_dict(self, include_district: bool = False) -> dict:
        data = {
            "policy": asdict(self.policy),
            "steps": [asdict(step) for step in self.steps],
            "initial_wind": self.initial_wind.to_dict(),
            "final_wind": self.final_wind.to_dict(),
            "final_summary": asdict(self.district.summary()),
        }
        if include_district:
            data["district"] = self.district.to_dict()
        return data


def joint_renewal_candidates(
    district: MokumitsuDistrict,
    policy: JointRenewalPolicy | None = None,
) -> tuple[JointRenewalCandidate, ...]:
    """Find connected access-poor/access-capable parcel clusters in priority order."""
    policy = policy or JointRenewalPolicy()
    _validate_policy(policy)
    parcels = [
        parcel
        for parcel in district.parcels
        if parcel.kind != "joint_redevelopment"
        and any(building.parcel_id == parcel.id for building in district.buildings)
    ]
    shapes = {parcel.id: Polygon(parcel.polygon) for parcel in parcels}
    buildings = {building.parcel_id: building for building in district.buildings}
    priorities = {
        candidate.parcel_id: candidate.priority_score for candidate in renewal_priorities(district)
    }
    accesses = {parcel.id: district.access(parcel.id) for parcel in parcels}
    parcels_by_id = {parcel.id: parcel for parcel in parcels}
    adjacency = {parcel.id: set() for parcel in parcels}
    shared_lengths = {}
    for index, first in enumerate(parcels):
        for second in parcels[index + 1 :]:
            if first.block_id != second.block_id:
                continue
            shared = shapes[first.id].boundary.intersection(shapes[second.id].boundary).length
            if shared < policy.shared_boundary_min_m:
                continue
            adjacency[first.id].add(second.id)
            adjacency[second.id].add(first.id)
            shared_lengths[frozenset((first.id, second.id))] = float(shared)

    candidates = []
    for parcel_ids in _connected_parcel_sets(adjacency, policy):
        selected = [parcels_by_id[parcel_id] for parcel_id in parcel_ids]
        if len({parcel.block_id for parcel in selected}) != 1:
            continue
        access_poor_ids = sorted(
            parcel.id for parcel in selected if not accesses[parcel.id].individual_rebuildable
        )
        access_capable_ids = sorted(
            parcel.id for parcel in selected if accesses[parcel.id].individual_rebuildable
        )
        if not access_poor_ids or not access_capable_ids:
            continue
        ordered_ids = tuple(access_poor_ids + access_capable_ids)
        merged = unary_union([shapes[parcel_id] for parcel_id in ordered_ids])
        if merged.geom_type != "Polygon" or merged.interiors:
            continue
        old_buildings = tuple(buildings[parcel_id] for parcel_id in ordered_ids)
        shared = sum(length for edge, length in shared_lengths.items() if edge.issubset(parcel_ids))
        candidates.append(
            JointRenewalCandidate(
                parcel_ids=ordered_ids,
                building_ids=tuple(building.id for building in old_buildings),
                block_id=selected[0].block_id,
                shared_boundary_m=float(shared),
                merged_area_m2=float(merged.area),
                old_footprint_m2=float(
                    sum(building.footprint_area_m2 for building in old_buildings)
                ),
                old_floor_area_m2=float(
                    sum(building.footprint_area_m2 * building.floors for building in old_buildings)
                ),
                priority_score=float(sum(priorities[parcel_id] for parcel_id in ordered_ids)),
                reasons=(
                    "resolves_access_by_joint_parcel",
                    "retains_existing_floor_area",
                    "creates_contiguous_open_space",
                    f"connected_{len(ordered_ids)}_parcel_cluster",
                ),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.priority_score / candidate.parcel_count,
            candidate.shared_boundary_m / candidate.parcel_count,
        ),
        reverse=True,
    )
    return tuple(candidates)


def joint_redevelopment_variants(
    district: MokumitsuDistrict,
    candidate: JointRenewalCandidate,
    policy: JointRenewalPolicy | None = None,
    wind_direction_deg: float = 0.0,
) -> tuple[tuple[MokumitsuDistrict, JointBuildingPlacement], ...]:
    """Create valid central and crosswind-edge placements for one connected parcel cluster."""
    policy = policy or JointRenewalPolicy()
    _validate_policy(policy)
    parcels = {parcel.id: parcel for parcel in district.parcels}
    try:
        selected = tuple(parcels[parcel_id] for parcel_id in candidate.parcel_ids)
    except KeyError as exc:
        raise ValueError(f"candidate references missing parcel {exc.args[0]}") from exc
    shape = unary_union([Polygon(parcel.polygon) for parcel in selected])
    if shape.geom_type != "Polygon" or shape.interiors:
        raise ValueError("joint parcels must form one simply connected polygon")

    access_road, road_space, developable = _road_widening_allocation(
        district,
        selected,
        shape,
        candidate,
        policy,
    )
    placements = _placement_options(
        developable,
        candidate,
        policy,
        math.radians(wind_direction_deg),
        access_road=access_road,
        base_spaces=(road_space,) if road_space is not None else (),
    )
    return tuple(
        (
            apply_joint_redevelopment(district, candidate, placement, policy),
            placement,
        )
        for placement in placements
    )


def apply_joint_redevelopment(
    district: MokumitsuDistrict,
    candidate: JointRenewalCandidate,
    placement: JointBuildingPlacement,
    policy: JointRenewalPolicy | None = None,
) -> MokumitsuDistrict:
    """Merge the selected parcels and replace their buildings with one joint building."""
    policy = policy or JointRenewalPolicy()
    _validate_policy(policy)
    parcel_ids = set(candidate.parcel_ids)
    building_ids = set(candidate.building_ids)
    selected_parcels = [parcel for parcel in district.parcels if parcel.id in parcel_ids]
    if len(selected_parcels) != len(parcel_ids) or len(selected_parcels) < 2:
        raise ValueError("joint redevelopment requires all referenced parcels and at least two")
    merged = unary_union([Polygon(parcel.polygon) for parcel in selected_parcels])
    if merged.geom_type != "Polygon" or merged.interiors:
        raise ValueError("joint parcels must form one simply connected polygon")

    road_spaces = [
        Polygon(space.polygon) for space in placement.open_spaces if space.kind == "road_widening"
    ]
    developable = merged.difference(unary_union(road_spaces)).buffer(0) if road_spaces else merged
    if developable.geom_type != "Polygon" or developable.interiors:
        raise RuntimeError("road dedication split or perforated the joint parcel")

    building_shape = _oriented_rectangle(
        placement.cx,
        placement.cy,
        placement.width_m,
        placement.depth_m,
        placement.theta,
    )
    for space in placement.open_spaces:
        space_shape = Polygon(space.polygon)
        if (
            not space_shape.is_valid
            or space_shape.area <= 0
            or not merged.buffer(1e-8).covers(space_shape)
        ):
            raise RuntimeError(f"invalid joint open-space allocation {space.id}")
        if building_shape.intersection(space_shape).area > 1e-8:
            raise RuntimeError(f"joint building overlaps allocated open space {space.id}")

    joint_frontages = list(_merged_frontages(selected_parcels))
    local_widened_roads = []
    for space in placement.open_spaces:
        if space.kind != "road_widening" or space.source_road_id is None:
            continue
        local_road = _local_widened_road(
            district.road(space.source_road_id),
            Polygon(space.polygon),
            placement.joint_parcel_id,
            policy.target_road_width_m,
        )
        if local_road is None or local_road.id != space.connected_road_id:
            raise RuntimeError(f"cannot reconstruct local road widening {space.id}")
        local_widened_roads.append(local_road)
        joint_frontages.append(
            Frontage(
                road_id=local_road.id,
                length_m=local_road.length_m,
                side="street_edge",
            )
        )

    joint_parcel = Parcel(
        id=placement.joint_parcel_id,
        block_id=candidate.block_id,
        polygon=tuple((float(x), float(y)) for x, y in list(developable.exterior.coords)[:-1]),
        frontages=tuple(joint_frontages),
        kind="joint_redevelopment",
    )
    joint_building = DistrictBuilding(
        id=placement.joint_building_id,
        parcel_id=joint_parcel.id,
        cx=placement.cx,
        cy=placement.cy,
        width_m=placement.width_m,
        depth_m=placement.depth_m,
        height_m=placement.height_m,
        floors=placement.floors,
        built_year=policy.year,
        structure="reinforced_concrete",
        fire_resistance="fire_resistant",
        use="residential",
        theta=placement.theta,
    )
    updated = replace(
        district,
        reference_year=policy.year,
        roads=district.roads + tuple(local_widened_roads),
        parcels=tuple(parcel for parcel in district.parcels if parcel.id not in parcel_ids)
        + (joint_parcel,),
        buildings=tuple(
            building for building in district.buildings if building.id not in building_ids
        )
        + (joint_building,),
        generator=(
            district.generator
            if "joint_renewal" in district.generator
            else district.generator + "+joint_renewal"
        ),
    )
    errors = validate_district(updated)
    if errors:
        raise RuntimeError("joint redevelopment produced invalid district: " + "; ".join(errors))
    if not updated.access(joint_parcel.id).individual_rebuildable:
        raise RuntimeError("joint redevelopment failed to inherit legal access")
    return updated


def optimize_joint_renewal(
    district: MokumitsuDistrict,
    policy: JointRenewalPolicy | None = None,
    scenario: SummerWindScenario | None = None,
    model: FnoModel | None = None,
) -> JointRenewalPlan:
    """Greedily select discrete joint projects and FNO-screen their valid placements."""
    policy = policy or JointRenewalPolicy()
    scenario = scenario or SummerWindScenario()
    _validate_policy(policy)
    model = model or load_model(scenario.model_name)
    reference_speed = float(model.reference_speed())
    current = district
    current_wind = evaluate_district_wind(
        current,
        scenario=scenario,
        model=model,
        reference_speed=reference_speed,
    )
    initial_wind = current_wind
    steps = []

    for index in range(1, policy.projects + 1):
        candidates = joint_renewal_candidates(current, policy)[: policy.candidate_limit]
        options = []
        for candidate in candidates:
            variants = joint_redevelopment_variants(
                current,
                candidate,
                policy,
                wind_direction_deg=scenario.directions[0].direction_deg,
            )
            for variant_district, placement in variants:
                wind = evaluate_district_wind(
                    variant_district,
                    scenario=scenario,
                    model=model,
                    reference_speed=reference_speed,
                )
                options.append(
                    (
                        joint_wind_objective(wind),
                        -candidate.priority_score,
                        placement.variant,
                        candidate,
                        placement,
                        variant_district,
                        wind,
                    )
                )
        if not options:
            break
        best = min(options, key=lambda item: item[:3])
        objective_before = joint_wind_objective(current_wind)
        objective_after, _, _, candidate, placement, next_district, next_wind = best
        if objective_after > objective_before + policy.max_wind_objective_increase:
            break
        before_zones = current_wind.weighted_zones
        after_zones = next_wind.weighted_zones
        steps.append(
            JointRenewalStep(
                index=index,
                candidate=candidate,
                placement=placement,
                wind_objective_before=objective_before,
                wind_objective_after=objective_after,
                outdoor_dead_before=before_zones["all_outdoor"].dead_fraction,
                outdoor_dead_after=after_zones["all_outdoor"].dead_fraction,
                yard_dead_before=before_zones["parcel_open_space"].dead_fraction,
                yard_dead_after=after_zones["parcel_open_space"].dead_fraction,
                edge_strong_before=before_zones["building_edge"].strong_fraction,
                edge_strong_after=after_zones["building_edge"].strong_fraction,
            )
        )
        current = next_district
        current_wind = next_wind

    return JointRenewalPlan(
        policy=policy,
        district=current,
        steps=tuple(steps),
        initial_wind=initial_wind,
        final_wind=current_wind,
    )


def _connected_parcel_sets(
    adjacency: dict[str, set[str]],
    policy: JointRenewalPolicy,
) -> tuple[frozenset[str], ...]:
    """Enumerate unique connected induced subsets without assuming an orthogonal lot grid."""
    stack = [frozenset((parcel_id,)) for parcel_id in sorted(adjacency)]
    seen = set(stack)
    clusters = set()
    while stack:
        current = stack.pop()
        if len(current) >= policy.minimum_cluster_parcels:
            clusters.add(current)
        if len(current) >= policy.maximum_cluster_parcels:
            continue
        frontier = set().union(*(adjacency[parcel_id] for parcel_id in current)) - set(current)
        for parcel_id in sorted(frontier, reverse=True):
            expanded = current | {parcel_id}
            if expanded in seen:
                continue
            seen.add(expanded)
            stack.append(expanded)
    return tuple(sorted(clusters, key=lambda ids: (len(ids), tuple(sorted(ids)))))


def _placement_options(
    shape: Polygon,
    candidate: JointRenewalCandidate,
    policy: JointRenewalPolicy,
    wind_theta: float,
    *,
    access_road: RoadSegment | None = None,
    base_spaces: tuple[JointOpenSpace, ...] = (),
) -> tuple[JointBuildingPlacement, ...]:
    usable = shape.buffer(-policy.parcel_setback_m)
    if usable.is_empty:
        usable = shape
    if usable.geom_type == "MultiPolygon":
        usable = max(usable.geoms, key=lambda polygon: polygon.area)
    if usable.geom_type != "Polygon":
        return ()

    first_floors = max(
        policy.minimum_floors,
        math.ceil(
            candidate.old_floor_area_m2 / max(policy.target_building_coverage * shape.area, 1e-9)
        ),
    )
    if first_floors > policy.maximum_floors:
        return ()
    shape_axis = _long_axis_angle(usable)
    angles = _unique_angles(
        (
            wind_theta,
            wind_theta - math.radians(15),
            wind_theta + math.radians(15),
            shape_axis,
        )
    )
    centres = _candidate_centres(usable, policy.placement_grid)
    crosswind = np.asarray((-math.sin(wind_theta), math.cos(wind_theta)))
    origin = np.asarray((shape.centroid.x, shape.centroid.y))
    raw = []

    for floors in range(first_floors, policy.maximum_floors + 1):
        footprint = candidate.old_floor_area_m2 / floors
        for aspect in (2.0, 1.5, 1.0):
            width = math.sqrt(footprint * aspect)
            depth = footprint / width
            for theta in angles:
                for centre in centres:
                    rectangle = _oriented_rectangle(
                        centre.x,
                        centre.y,
                        width,
                        depth,
                        theta,
                    )
                    if not usable.buffer(1e-8).covers(rectangle):
                        continue
                    projection = float(np.dot(np.asarray((centre.x, centre.y)) - origin, crosswind))
                    raw.append(
                        (
                            projection,
                            centre.distance(shape.centroid),
                            centre,
                            width,
                            depth,
                            theta,
                            floors,
                        )
                    )
        if raw:
            break
    if not raw:
        return ()

    selected = [
        ("crosswind_negative", min(raw, key=lambda item: item[0])),
        ("crosswind_positive", max(raw, key=lambda item: item[0])),
        ("central", min(raw, key=lambda item: item[1])),
    ]
    joint_parcel_id, joint_building_id = _joint_ids(candidate.parcel_ids)
    placements = []
    seen = set()
    for variant, item in selected:
        _, _, centre, width, depth, theta, floors = item
        key = (
            round(centre.x, 4),
            round(centre.y, 4),
            round(width, 4),
            round(depth, 4),
            round(theta, 4),
            floors,
        )
        if key in seen:
            continue
        seen.add(key)
        footprint = width * depth
        building_shape = _oriented_rectangle(
            centre.x,
            centre.y,
            width,
            depth,
            theta,
        )
        open_spaces = list(base_spaces)
        corridor = _wind_corridor_space(
            shape,
            building_shape,
            access_road,
            joint_parcel_id,
            policy,
            wind_theta,
        )
        if corridor is not None:
            open_spaces.append(corridor)
        shared_space = _shared_open_space(
            shape,
            building_shape,
            tuple(open_spaces),
            candidate,
            joint_parcel_id,
            policy,
        )
        if shared_space is not None:
            open_spaces.append(shared_space)
        placements.append(
            JointBuildingPlacement(
                variant=variant,
                joint_parcel_id=joint_parcel_id,
                joint_building_id=joint_building_id,
                cx=float(centre.x),
                cy=float(centre.y),
                width_m=float(width),
                depth_m=float(depth),
                theta=float(theta % math.pi),
                floors=int(floors),
                height_m=float(floors * policy.floor_height_m + 0.5),
                footprint_m2=float(footprint),
                floor_area_m2=float(footprint * floors),
                local_building_coverage=float(footprint / shape.area),
                open_spaces=tuple(open_spaces),
            )
        )
        if len(placements) >= policy.placement_variants:
            break
    return tuple(placements)


def _candidate_centres(polygon: Polygon, grid: int) -> tuple[Point, ...]:
    min_x, min_y, max_x, max_y = polygon.bounds
    points = [
        polylabel(polygon, tolerance=0.08),
        polygon.centroid,
        polygon.representative_point(),
    ]
    for y in np.linspace(min_y, max_y, grid):
        for x in np.linspace(min_x, max_x, grid):
            point = Point(float(x), float(y))
            if polygon.covers(point):
                points.append(point)
    unique = {}
    for point in points:
        unique[(round(point.x, 5), round(point.y, 5))] = point
    return tuple(unique.values())


def _primary_access_road(
    district: MokumitsuDistrict,
    parcels: tuple[Parcel, ...],
) -> RoadSegment | None:
    frontage_by_road: dict[str, float] = {}
    for parcel in parcels:
        for frontage in parcel.frontages:
            road = district.road(frontage.road_id)
            if road.legal_class == NON_ROAD:
                continue
            frontage_by_road[road.id] = frontage_by_road.get(road.id, 0.0) + frontage.length_m
    if not frontage_by_road:
        return None
    return max(
        (district.road(road_id) for road_id in frontage_by_road),
        key=lambda road: (frontage_by_road[road.id], road.width_m, road.id),
    )


def _road_widening_allocation(
    district: MokumitsuDistrict,
    parcels: tuple[Parcel, ...],
    shape: Polygon,
    candidate: JointRenewalCandidate,
    policy: JointRenewalPolicy,
) -> tuple[RoadSegment | None, JointOpenSpace | None, Polygon]:
    road = _primary_access_road(district, parcels)
    if road is None or road.width_m >= policy.target_road_width_m - 1e-9:
        return road, None, shape

    centreline = LineString((road.start, road.end))
    existing = centreline.buffer(road.width_m / 2, cap_style=2, join_style=2)
    widened = centreline.buffer(policy.target_road_width_m / 2, cap_style=2, join_style=2)
    reserve = _largest_simple_polygon(shape.intersection(widened.difference(existing)))
    if reserve is None or reserve.area < 0.25:
        return road, None, shape
    developable = shape.difference(reserve).buffer(0)
    if (
        developable.geom_type != "Polygon"
        or developable.interiors
        or developable.area < 0.60 * shape.area
    ):
        return road, None, shape

    joint_parcel_id, _ = _joint_ids(candidate.parcel_ids)
    angle = math.degrees(math.atan2(road.end[1] - road.start[1], road.end[0] - road.start[0])) % 180
    local_road = _local_widened_road(
        road,
        reserve,
        joint_parcel_id,
        policy.target_road_width_m,
    )
    if local_road is None:
        return road, None, shape
    space = _open_space_from_polygon(
        reserve,
        id=f"{joint_parcel_id}_road_widening",
        kind="road_widening",
        connected_road_id=local_road.id,
        source_road_id=road.id,
        orientation_deg=angle,
    )
    return local_road, space, developable


def _local_widened_road(
    source: RoadSegment,
    reserve: Polygon,
    joint_parcel_id: str,
    target_width_m: float,
) -> RoadSegment | None:
    centreline = LineString((source.start, source.end))
    projected = [centreline.project(Point(float(x), float(y))) for x, y in reserve.exterior.coords]
    start_distance = max(0.0, min(projected))
    end_distance = min(source.length_m, max(projected))
    if end_distance - start_distance < 0.25:
        return None
    start = centreline.interpolate(start_distance)
    end = centreline.interpolate(end_distance)
    return RoadSegment(
        id=f"{source.id}__{joint_parcel_id}_widened",
        start=(float(start.x), float(start.y)),
        end=(float(end.x), float(end.y)),
        width_m=float(target_width_m),
        legal_class=source.legal_class,
        kind="widened_frontage",
        dead_end=False,
    )


def _wind_corridor_space(
    shape: Polygon,
    building_shape: Polygon,
    access_road: RoadSegment | None,
    joint_parcel_id: str,
    policy: JointRenewalPolicy,
    wind_theta: float,
) -> JointOpenSpace | None:
    width = policy.ventilation_corridor_width_m
    axis = np.asarray((math.cos(wind_theta), math.sin(wind_theta)), dtype=float)
    crosswind = np.asarray((-axis[1], axis[0]), dtype=float)
    coordinates = np.asarray(shape.exterior.coords, dtype=float)
    along = coordinates @ axis
    across = coordinates @ crosswind
    full_span = float(np.ptp(along))
    cross_span = float(np.ptp(across))
    if full_span <= 1e-9 or cross_span <= width:
        return None

    road_envelope = None
    if access_road is not None:
        road_envelope = LineString((access_road.start, access_road.end)).buffer(
            max(access_road.width_m, policy.target_road_width_m) / 2,
            cap_style=2,
            join_style=2,
        )
    centre = np.asarray((shape.centroid.x, shape.centroid.y), dtype=float)
    centre_along = float(centre @ axis)
    centre_across = float(centre @ crosswind)
    half_extent = max(math.hypot(*(np.ptp(coordinates, axis=0))) * 1.5, full_span)
    offsets = np.linspace(
        float(across.min() + width / 2),
        float(across.max() - width / 2),
        max(2 * policy.placement_grid + 1, 9),
    )
    blocked = building_shape.buffer(policy.parcel_setback_m)
    candidates = []
    for offset in offsets:
        point = axis * centre_along + crosswind * float(offset)
        centreline = LineString((point - axis * half_extent, point + axis * half_extent))
        corridor = shape.intersection(centreline.buffer(width / 2, cap_style=2))
        for part in _polygon_parts(corridor):
            if part.interiors or part.area < width * width:
                continue
            if part.intersection(blocked).area > 1e-8:
                continue
            part_coordinates = np.asarray(part.exterior.coords, dtype=float)
            span_ratio = float(np.ptp(part_coordinates @ axis) / full_span)
            if span_ratio + 1e-9 < policy.minimum_corridor_span_ratio:
                continue
            connected = road_envelope is not None and part.distance(road_envelope) <= 0.15
            candidates.append(
                (
                    bool(connected),
                    span_ratio,
                    float(part.area),
                    -abs(float(offset) - centre_across),
                    part,
                )
            )
    if not candidates:
        return None
    connected, span_ratio, _, _, corridor = max(candidates, key=lambda item: item[:4])
    return _open_space_from_polygon(
        corridor,
        id=f"{joint_parcel_id}_wind_corridor",
        kind="ventilation_corridor",
        connected_road_id=access_road.id if connected and access_road is not None else None,
        orientation_deg=math.degrees(wind_theta) % 180,
        span_ratio=span_ratio,
    )


def _shared_open_space(
    shape: Polygon,
    building_shape: Polygon,
    existing_spaces: tuple[JointOpenSpace, ...],
    candidate: JointRenewalCandidate,
    joint_parcel_id: str,
    policy: JointRenewalPolicy,
) -> JointOpenSpace | None:
    reserved = [
        Polygon(space.polygon) for space in existing_spaces if space.kind != "road_widening"
    ]
    unavailable = [building_shape.buffer(policy.parcel_setback_m), *reserved]
    available = shape.difference(unary_union(unavailable)).buffer(0)
    target_area = max(
        policy.minimum_shared_open_space_m2,
        policy.shared_open_space_fraction * shape.area,
    )
    connectors = [
        (space, Polygon(space.polygon))
        for space in existing_spaces
        if space.connected_road_id is not None
    ]
    options = []
    for part in _polygon_parts(available):
        if part.area < policy.minimum_shared_open_space_m2:
            continue
        anchors = [polylabel(part, tolerance=0.08), part.centroid]
        for _, connector in connectors:
            if part.distance(connector) <= 0.15:
                anchors.append(nearest_points(connector, part)[1])
        for anchor in anchors:
            reserve = _compact_reserve(part, anchor, min(target_area, part.area))
            if reserve is None or reserve.area < policy.minimum_shared_open_space_m2:
                continue
            connected_space = next(
                (space for space, connector in connectors if reserve.distance(connector) <= 0.15),
                None,
            )
            options.append(
                (
                    connected_space is not None,
                    min(float(reserve.area / target_area), 1.0),
                    float(reserve.area),
                    -float(reserve.length),
                    reserve,
                    connected_space,
                )
            )
    if not options:
        return None
    _, _, _, _, reserve, connected_space = max(options, key=lambda item: item[:4])
    kind = "pocket_park" if candidate.parcel_count >= 3 else "shared_court"
    return _open_space_from_polygon(
        reserve,
        id=f"{joint_parcel_id}_{kind}",
        kind=kind,
        connected_road_id=(
            connected_space.connected_road_id if connected_space is not None else None
        ),
    )


def _compact_reserve(
    available: Polygon,
    anchor: Point,
    target_area: float,
) -> Polygon | None:
    min_x, min_y, max_x, max_y = available.bounds
    high = max(math.hypot(max_x - min_x, max_y - min_y), 0.1)
    low = 0.0
    best = None
    for _ in range(24):
        radius = (low + high) / 2
        candidate = _largest_simple_polygon(
            available.intersection(anchor.buffer(radius, quad_segs=12))
        )
        area = candidate.area if candidate is not None else 0.0
        if candidate is not None:
            best = candidate
        if area < target_area:
            low = radius
        else:
            high = radius
    return best


def _polygon_parts(geometry) -> tuple[Polygon, ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    return tuple(
        part
        for part in getattr(geometry, "geoms", ())
        if part.geom_type == "Polygon" and not part.is_empty
    )


def _largest_simple_polygon(geometry) -> Polygon | None:
    simple = [part for part in _polygon_parts(geometry) if not part.interiors and part.area > 1e-9]
    return max(simple, key=lambda part: part.area) if simple else None


def _open_space_from_polygon(
    polygon: Polygon,
    *,
    id: str,
    kind: str,
    connected_road_id: str | None = None,
    source_road_id: str | None = None,
    orientation_deg: float | None = None,
    span_ratio: float | None = None,
) -> JointOpenSpace:
    return JointOpenSpace(
        id=id,
        kind=kind,
        polygon=tuple((float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]),
        area_m2=float(polygon.area),
        connected_road_id=connected_road_id,
        source_road_id=source_road_id,
        orientation_deg=orientation_deg,
        span_ratio=span_ratio,
    )


def _merged_frontages(parcels: list[Parcel]) -> tuple[Frontage, ...]:
    return tuple(
        sorted(
            (frontage for parcel in parcels for frontage in parcel.frontages),
            key=lambda frontage: (frontage.road_id, frontage.side, frontage.length_m),
        )
    )


def _joint_ids(parcel_ids: tuple[str, ...]) -> tuple[str, str]:
    suffix = "__".join(parcel_id.removeprefix("parcel_") for parcel_id in sorted(parcel_ids))
    return f"joint_parcel_{suffix}", f"joint_building_{suffix}"


def _oriented_rectangle(
    cx: float,
    cy: float,
    width: float,
    depth: float,
    theta: float,
) -> Polygon:
    ct, st = math.cos(theta), math.sin(theta)
    corners = []
    for x, y in (
        (-width / 2, -depth / 2),
        (width / 2, -depth / 2),
        (width / 2, depth / 2),
        (-width / 2, depth / 2),
    ):
        corners.append((cx + ct * x - st * y, cy + st * x + ct * y))
    return Polygon(corners)


def _long_axis_angle(polygon: Polygon) -> float:
    rectangle = polygon.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)[:4]
    edges = [
        (
            math.dist(coordinates[index], coordinates[(index + 1) % 4]),
            coordinates[index],
            coordinates[(index + 1) % 4],
        )
        for index in range(4)
    ]
    _, start, end = max(edges, key=lambda edge: edge[0])
    return math.atan2(end[1] - start[1], end[0] - start[0]) % math.pi


def _unique_angles(angles) -> tuple[float, ...]:
    unique = {}
    for angle in angles:
        normalized = angle % math.pi
        unique[round(normalized, 8)] = normalized
    return tuple(unique.values())


def joint_wind_objective(wind: DistrictWindMetrics) -> float:
    """Scalar screening penalty shared by greedy and Pareto renewal workflows."""
    zones = wind.weighted_zones
    return float(
        0.55 * zones["parcel_open_space"].dead_fraction
        + 0.25 * zones["all_outdoor"].dead_fraction
        + 0.10 * zones["roads"].dead_fraction
        + 0.35 * zones["building_edge"].strong_fraction
        + 0.15 * zones["all_outdoor"].strong_fraction
    )


def _validate_policy(policy: JointRenewalPolicy) -> None:
    if policy.projects < 0 or policy.candidate_limit <= 0:
        raise ValueError("projects must be non-negative and candidate_limit positive")
    if policy.shared_boundary_min_m <= 0 or policy.parcel_setback_m < 0:
        raise ValueError("shared-boundary threshold must be positive and setback non-negative")
    if not (2 <= policy.minimum_cluster_parcels <= policy.maximum_cluster_parcels <= 4):
        raise ValueError("cluster parcel limits must satisfy 2 <= minimum <= maximum <= 4")
    if not 0 < policy.target_building_coverage < 1:
        raise ValueError("target_building_coverage must lie between zero and one")
    if not 1 <= policy.minimum_floors <= policy.maximum_floors:
        raise ValueError("floor limits are inconsistent")
    if policy.floor_height_m <= 0:
        raise ValueError("floor_height_m must be positive")
    if policy.target_road_width_m <= 0 or policy.ventilation_corridor_width_m <= 0:
        raise ValueError("road and ventilation-corridor widths must be positive")
    if not 0 < policy.minimum_corridor_span_ratio <= 1:
        raise ValueError("minimum_corridor_span_ratio must lie in (0, 1]")
    if not 0 < policy.shared_open_space_fraction < 1:
        raise ValueError("shared_open_space_fraction must lie between zero and one")
    if policy.minimum_shared_open_space_m2 <= 0:
        raise ValueError("minimum_shared_open_space_m2 must be positive")
    if policy.placement_grid < 3 or policy.placement_variants <= 0:
        raise ValueError("placement search resolution and variants must be positive")
    if policy.max_wind_objective_increase < 0:
        raise ValueError("max_wind_objective_increase must be non-negative")
