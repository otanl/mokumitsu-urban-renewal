"""Growth-based, non-orthogonal morphology for synthetic 木密 districts.

The generator deliberately does not start from a rectangular street grid.  It builds
kinked historical routes, grows secondary alleys from them, buffers the resulting
planar graph, polygonises the residual land, and only then subdivides each irregular
block into parcels.  Some landlocked parcels receive either a narrow flag-lot stem or
a shared non-road access path.  The result is synthetic, but its topology is explicit
and measurable instead of being a jittered Cartesian lattice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, polylabel, split, unary_union

from .district import (
    ARTICLE_42_1,
    ARTICLE_42_2,
    NON_ROAD,
    DistrictBuilding,
    Frontage,
    MokumitsuConfig,
    MokumitsuDistrict,
    Parcel,
    RoadSegment,
    _building_for_parcel,
    _validate_config,
    validate_district,
)


@dataclass(frozen=True)
class _RoadFeature:
    line: LineString
    width_m: float
    legal_class: str
    kind: str
    dead_end: bool = False


def generate_organic_mokumitsu(
    seed: int = 0, config: MokumitsuConfig | None = None
) -> MokumitsuDistrict:
    """Generate one deterministic non-grid district and its correlated attributes."""
    cfg = config or MokumitsuConfig()
    _validate_config(cfg)
    rng = np.random.default_rng(seed)
    site = box(0.0, 0.0, cfg.width_m, cfg.height_m)

    features = _perimeter_features(cfg)
    historic = _historic_features(cfg, rng)
    features.extend(historic)
    features.extend(_branch_features(cfg, rng, historic, features, site))
    roads = list(_node_features(features))

    road_surface = unary_union(
        [
            feature.line.buffer(feature.width_m / 2, cap_style="flat", join_style="mitre")
            for feature in features
        ]
    )
    residual = site.difference(road_surface)
    blocks = [
        polygon.buffer(0)
        for polygon in _polygons(residual)
        if polygon.area >= cfg.minimum_lot_area_m2 * 1.25
    ]
    blocks.sort(key=lambda p: (round(p.centroid.x, 6), round(p.centroid.y, 6)))
    if len(blocks) < 3:
        raise RuntimeError("organic road growth did not produce enough blocks")

    parcel_records: list[tuple[Parcel, Polygon]] = []
    block_legacy: dict[str, float] = {}
    for block_number, block in enumerate(blocks):
        block_id = f"block_{block_number:03d}"
        block_legacy[block_id] = float(rng.beta(2.0, 2.0))
        lots = _subdivide_block(block, cfg, rng)
        lots, kinds, forced_frontages = _add_flag_stems(lots, block, roads, cfg, rng)

        for lot_number, (lot, kind) in enumerate(zip(lots, kinds, strict=True)):
            if lot.area < cfg.minimum_lot_area_m2 * 0.45:
                continue
            frontages = _street_frontages(lot, roads)
            forced = forced_frontages.get(lot_number)
            if (
                kind == "flag_lot"
                and forced is not None
                and not any(frontage.road_id == forced[0] for frontage in frontages)
            ):
                frontages.append(Frontage(forced[0], forced[1], "flag_stem"))
            if not frontages:
                road, parcel_point, road_point = _nearest_legal_road(lot, roads)
                path_width = float(rng.uniform(*cfg.access_path_width_m))
                path = RoadSegment(
                    id=f"access_{block_number:03d}_{lot_number:03d}",
                    start=_point_tuple(parcel_point),
                    end=_point_tuple(road_point),
                    width_m=path_width,
                    legal_class=NON_ROAD,
                    kind="shared_access",
                    dead_end=True,
                )
                roads.append(path)
                frontages.append(Frontage(path.id, path_width, "shared_path"))
                if kind == "regular":
                    kind = "back_lot"

            parcel = Parcel(
                id=f"parcel_{block_number:03d}_{lot_number:03d}",
                block_id=block_id,
                polygon=_polygon_tuple(lot),
                frontages=tuple(frontages),
                kind=kind,
            )
            parcel_records.append((parcel, lot))

    road_lookup = {road.id: road for road in roads}
    buildings = tuple(
        _building_for_irregular_parcel(
            parcel,
            polygon,
            number,
            block_legacy[parcel.block_id],
            road_lookup,
            cfg,
            rng,
        )
        for number, (parcel, polygon) in enumerate(parcel_records)
    )
    district = MokumitsuDistrict(
        width_m=cfg.width_m,
        height_m=cfg.height_m,
        seed=seed,
        reference_year=cfg.reference_year,
        roads=tuple(roads),
        parcels=tuple(record[0] for record in parcel_records),
        buildings=buildings,
        generator="organic",
    )
    errors = validate_district(district)
    if errors:
        raise RuntimeError("invalid generated district: " + "; ".join(errors))
    return district


def _perimeter_features(cfg: MokumitsuConfig) -> list[_RoadFeature]:
    w, h = cfg.width_m, cfg.height_m
    width = cfg.perimeter_road_width_m
    return [
        _RoadFeature(LineString(((0, 0), (w, 0))), width, ARTICLE_42_1, "perimeter"),
        _RoadFeature(LineString(((w, 0), (w, h))), width, ARTICLE_42_1, "perimeter"),
        _RoadFeature(LineString(((w, h), (0, h))), width, ARTICLE_42_1, "perimeter"),
        _RoadFeature(LineString(((0, h), (0, 0))), width, ARTICLE_42_1, "perimeter"),
    ]


def _historic_features(cfg: MokumitsuConfig, rng) -> list[_RoadFeature]:
    features = []
    for route in range(cfg.historic_routes):
        horizontal = route % 2 == 0
        fractions = np.linspace(0.0, 1.0, 6)
        if horizontal:
            start_y = rng.uniform(0.28, 0.72) * cfg.height_m
            end_y = rng.uniform(0.28, 0.72) * cfg.height_m
            baseline = start_y + fractions * (end_y - start_y)
            jitter = rng.normal(0.0, 0.055 * cfg.height_m, fractions.size)
            jitter[[0, -1]] = 0.0
            coords = tuple(
                (float(t * cfg.width_m), float(np.clip(y + j, 8.0, cfg.height_m - 8.0)))
                for t, y, j in zip(fractions, baseline, jitter, strict=True)
            )
        else:
            start_x = rng.uniform(0.25, 0.75) * cfg.width_m
            end_x = rng.uniform(0.25, 0.75) * cfg.width_m
            baseline = start_x + fractions * (end_x - start_x)
            jitter = rng.normal(0.0, 0.055 * cfg.width_m, fractions.size)
            jitter[[0, -1]] = 0.0
            coords = tuple(
                (float(np.clip(x + j, 8.0, cfg.width_m - 8.0)), float(t * cfg.height_m))
                for t, x, j in zip(fractions, baseline, jitter, strict=True)
            )
        width = float(rng.uniform(*cfg.legal_street_width_m))
        features.append(_RoadFeature(LineString(coords), width, ARTICLE_42_1, "historic_route"))
    return features


def _branch_features(cfg, rng, historic, existing, site) -> list[_RoadFeature]:
    branches: list[_RoadFeature] = []
    inset = site.buffer(-1.0)
    for branch_number in range(cfg.branch_alleys):
        parent = historic[branch_number % len(historic)].line
        t = float(rng.uniform(0.10, 0.90))
        start = parent.interpolate(t, normalized=True)
        before = parent.interpolate(max(0.0, t - 0.015), normalized=True)
        after = parent.interpolate(min(1.0, t + 0.015), normalized=True)
        tangent = math.atan2(after.y - before.y, after.x - before.x)
        direction = tangent + rng.choice((-1.0, 1.0)) * math.pi / 2 + rng.normal(0.0, 0.25)
        coords = [(start.x, start.y)]
        segments = int(rng.integers(2, 5))
        for _ in range(segments):
            direction += float(rng.normal(0.0, 0.22))
            length = float(rng.uniform(8.0, 17.0))
            candidate = Point(
                coords[-1][0] + length * math.cos(direction),
                coords[-1][1] + length * math.sin(direction),
            )
            if not inset.covers(candidate):
                break
            coords.append((candidate.x, candidate.y))
        if len(coords) < 2:
            continue

        dead_end = bool(rng.random() < cfg.dead_end_probability)
        if not dead_end and rng.random() < cfg.loop_probability:
            targets = [
                feature.line for feature in existing + branches if feature.line is not parent
            ]
            if targets:
                endpoint = Point(coords[-1])
                _, target = nearest_points(endpoint, unary_union(targets))
                if 4.0 < endpoint.distance(target) < 28.0:
                    coords.append((target.x, target.y))
        width = float(rng.uniform(*cfg.narrow_street_width_m))
        legal_class = ARTICLE_42_2 if rng.random() > 0.16 else NON_ROAD
        branches.append(
            _RoadFeature(LineString(coords), width, legal_class, "grown_alley", dead_end=dead_end)
        )
    return branches


def _node_features(features: list[_RoadFeature]) -> tuple[RoadSegment, ...]:
    """Node crossings so the road list is a real planar graph, not just crossing strokes."""
    noded = unary_union([feature.line for feature in features])
    road_segments = []
    for line in _lines(noded):
        coords = list(line.coords)
        for start, end in zip(coords, coords[1:], strict=False):
            if math.dist(start, end) < 1e-5:
                continue
            midpoint = Point((0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])))
            source = min(features, key=lambda feature: feature.line.distance(midpoint))
            road_segments.append((start, end, source))
    road_segments.sort(
        key=lambda item: (
            round(min(item[0][0], item[1][0]), 6),
            round(min(item[0][1], item[1][1]), 6),
            round(max(item[0][0], item[1][0]), 6),
            round(max(item[0][1], item[1][1]), 6),
        )
    )
    return tuple(
        RoadSegment(
            id=f"road_{number:04d}",
            start=_point_tuple(start),
            end=_point_tuple(end),
            width_m=source.width_m,
            legal_class=source.legal_class,
            kind=source.kind,
            dead_end=source.dead_end,
        )
        for number, (start, end, source) in enumerate(road_segments)
    )


def _subdivide_block(block: Polygon, cfg: MokumitsuConfig, rng) -> list[Polygon]:
    target = cfg.target_lot_area_m2 * float(rng.uniform(0.82, 1.20))
    lots = [block]
    unsplittable: set[int] = set()
    for _ in range(256):
        candidates = [
            (polygon.area, index)
            for index, polygon in enumerate(lots)
            if index not in unsplittable and polygon.area > target * 1.45
        ]
        if not candidates:
            break
        _, index = max(candidates)
        pieces = _split_lot(lots[index], cfg.minimum_lot_area_m2, rng)
        if pieces is None:
            unsplittable.add(index)
            continue
        lots[index : index + 1] = pieces
        unsplittable = {i if i < index else i + len(pieces) - 1 for i in unsplittable}
    return [polygon.buffer(0) for polygon in lots if polygon.area >= cfg.minimum_lot_area_m2]


def _split_lot(polygon: Polygon, minimum_area: float, rng) -> list[Polygon] | None:
    theta, span = _long_axis(polygon)
    diagonal = math.hypot(
        polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]
    )
    for _ in range(10):
        cut_angle = theta + math.pi / 2 + float(rng.normal(0.0, 0.17))
        along = float(rng.uniform(-0.13, 0.13) * span)
        centre = polygon.centroid
        cx = centre.x + along * math.cos(theta)
        cy = centre.y + along * math.sin(theta)
        dx, dy = math.cos(cut_angle) * diagonal * 2, math.sin(cut_angle) * diagonal * 2
        result = split(polygon, LineString(((cx - dx, cy - dy), (cx + dx, cy + dy))))
        pieces = [part.buffer(0) for part in _polygons(result)]
        if len(pieces) >= 2 and min(part.area for part in pieces) >= minimum_area:
            return sorted(pieces, key=lambda p: (p.centroid.x, p.centroid.y))
    return None


def _add_flag_stems(lots, block, roads, cfg, rng):
    kinds = ["regular"] * len(lots)
    forced: dict[int, tuple[str, float]] = {}
    interior = [index for index, lot in enumerate(lots) if not _street_frontages(lot, roads)]
    rng.shuffle(interior)
    desired = min(len(interior), max(1, int(round(0.30 * len(interior))))) if interior else 0
    made = 0
    for index in interior:
        if made >= desired:
            break
        if made > 0 and rng.random() > cfg.flag_lot_probability:
            continue
        road, parcel_point, road_point = _nearest_legal_road(lots[index], roads)
        width = float(rng.uniform(1.45, 2.55))
        stem = LineString((parcel_point, road_point)).buffer(width / 2, cap_style="flat")
        stem = stem.intersection(block)
        merged = lots[index].union(stem).buffer(0)
        merged_parts = _polygons(merged)
        if not merged_parts:
            continue
        merged = max(merged_parts, key=lambda p: p.area)
        trial = list(lots)
        trial[index] = merged
        valid = True
        for other_index, other in enumerate(lots):
            if other_index == index:
                continue
            difference = other.difference(stem).buffer(0)
            parts = _polygons(difference)
            if not parts:
                valid = False
                break
            largest = max(parts, key=lambda p: p.area)
            if largest.area < cfg.minimum_lot_area_m2 * 0.45:
                valid = False
                break
            trial[other_index] = largest
        if not valid:
            continue
        lots = trial
        kinds[index] = "flag_lot"
        forced[index] = (road.id, width)
        made += 1
    return lots, kinds, forced


def _street_frontages(polygon: Polygon, roads: list[RoadSegment]) -> list[Frontage]:
    frontages = []
    for road in roads:
        if road.legal_class == NON_ROAD:
            continue
        line = LineString((road.start, road.end))
        contact = polygon.boundary.intersection(
            line.buffer(road.width_m / 2 + 0.22, cap_style="flat")
        ).length
        if contact > 0.35:
            frontages.append(Frontage(road.id, float(contact), "street_edge"))
    return frontages


def _nearest_legal_road(polygon, roads):
    legal = [road for road in roads if road.legal_class != NON_ROAD]
    if not legal:
        raise RuntimeError("organic district needs at least one legal road")
    road = min(
        legal,
        key=lambda item: polygon.distance(LineString((item.start, item.end))),
    )
    parcel_point, road_point = nearest_points(polygon, LineString((road.start, road.end)))
    return road, parcel_point, road_point


def _building_for_irregular_parcel(parcel, polygon, number, legacy, roads, cfg, rng):
    base: DistrictBuilding = _building_for_parcel(parcel, number, legacy, roads, cfg, rng)
    setback = float(rng.uniform(*cfg.building_setback_m))
    inner = polygon.buffer(-setback, join_style="mitre")
    parts = _polygons(inner)
    if not parts:
        inner = polygon.buffer(-0.15)
        parts = _polygons(inner)
    usable = max(parts, key=lambda p: p.area) if parts else polygon
    theta, _ = _long_axis(usable)
    theta += float(rng.normal(0.0, 0.035))
    centre, width, depth, theta = _largest_inscribed_rectangle(usable, theta)

    age_bonus = 0.04 if base.built_year <= 1980 else 0.015 if base.built_year <= 2000 else -0.015
    target_coverage = float(
        np.clip(
            cfg.target_net_building_coverage
            + age_bonus
            + rng.normal(0.0, cfg.building_coverage_jitter),
            0.34,
            0.68,
        )
    )
    target_area = min(parcel.area_m2 * target_coverage, width * depth)
    scale = math.sqrt(target_area / max(width * depth, 1e-9))
    width *= scale
    depth *= scale
    footprint = _oriented_box(centre.x, centre.y, width, depth, theta)
    if not polygon.buffer(1e-7).covers(footprint):
        raise RuntimeError(f"could not fit building in {parcel.id}")
    return replace(
        base,
        cx=float(centre.x),
        cy=float(centre.y),
        width_m=float(width),
        depth_m=float(depth),
        theta=float(_normalise_half_turn(theta)),
    )


def _largest_inscribed_rectangle(polygon: Polygon, theta: float):
    """Fast deterministic search for a large parcel-aligned inscribed rectangle."""
    rectangle = polygon.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)[:4]
    lengths = [math.dist(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    long_side, short_side = max(lengths), min(lengths)
    centres = [
        polylabel(polygon, tolerance=0.06),
        polygon.centroid,
        rectangle.centroid,
        polygon.representative_point(),
    ]
    unique_centres = []
    seen = set()
    for centre in centres:
        key = round(centre.x, 5), round(centre.y, 5)
        if key not in seen and polygon.covers(centre):
            seen.add(key)
            unique_centres.append(centre)

    best = None
    prepared = polygon.buffer(1e-8)
    for centre in unique_centres:
        for angle in (theta - 0.055, theta, theta + 0.055):
            low, high = 0.0, 1.0
            for _ in range(22):
                scale = (low + high) / 2
                candidate = _oriented_box(
                    centre.x, centre.y, long_side * scale, short_side * scale, angle
                )
                if prepared.covers(candidate):
                    low = scale
                else:
                    high = scale
            area = long_side * short_side * low * low
            if best is None or area > best[0]:
                best = (area, centre, long_side * low, short_side * low, angle)
    if best is None or best[0] <= 0.25:
        centre = polylabel(polygon, tolerance=0.04)
        radius = max(0.25, centre.distance(polygon.boundary) * 0.90)
        return centre, math.sqrt(2) * radius, math.sqrt(2) * radius, 0.0
    return best[1], best[2], best[3], best[4]


def _oriented_box(cx, cy, width, depth, theta):
    rectangle = box(cx - width / 2, cy - depth / 2, cx + width / 2, cy + depth / 2)
    return affinity.rotate(rectangle, math.degrees(theta), origin=(cx, cy))


def _long_axis(polygon: Polygon) -> tuple[float, float]:
    rectangle = polygon.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)[:4]
    edges = [
        (math.dist(coords[i], coords[(i + 1) % 4]), coords[i], coords[(i + 1) % 4])
        for i in range(4)
    ]
    length, start, end = max(edges, key=lambda edge: edge[0])
    return math.atan2(end[1] - start[1], end[0] - start[0]), length


def _normalise_half_turn(theta):
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def _polygons(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for geom in geometry.geoms for part in _polygons(geom)]
    return []


def _lines(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for geom in geometry.geoms for part in _lines(geom)]
    return []


def _point_tuple(point) -> tuple[float, float]:
    if hasattr(point, "x"):
        return round(float(point.x), 9), round(float(point.y), 9)
    return round(float(point[0]), 9), round(float(point[1]), 9)


def _polygon_tuple(polygon: Polygon) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in list(polygon.exterior.coords)[:-1])
