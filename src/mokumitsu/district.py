"""Synthetic Japanese dense-wooden-neighbourhood (木密) data and generation.

The wind-design experiments need more than a height-map: redevelopment sequencing also needs
roads, parcels, legal-access proxies, construction cohorts and fire-vulnerability attributes.
This module supplies that shared, Houdini-independent representation.  Coordinates are metres;
``MokumitsuDistrict.heightmap`` converts them to the normalized field used by the FNO/XLB path.

The access assessment is a planning proxy, not a building-permit decision.  It models the common
Article 43 rule (at least 2 m of frontage to a legal road), Article 42(2)-style narrow-road setback,
and non-road paths, while deliberately leaving local exceptions to the relevant authority.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon

from . import geometry

Point = tuple[float, float]
Polygon = tuple[Point, ...]

ARTICLE_42_1 = "article_42_1"
ARTICLE_42_2 = "article_42_2"
NON_ROAD = "non_road"
_LEGAL_ROADS = frozenset((ARTICLE_42_1, ARTICLE_42_2))


@dataclass(frozen=True)
class RoadSegment:
    """A road-graph edge represented by its centre line and physical width."""

    id: str
    start: Point
    end: Point
    width_m: float
    legal_class: str
    kind: str = "street"
    dead_end: bool = False

    @property
    def length_m(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def is_legal_road(self) -> bool:
        return self.legal_class in _LEGAL_ROADS


@dataclass(frozen=True)
class Frontage:
    """Length of one parcel side facing a road/path."""

    road_id: str
    length_m: float
    side: str


@dataclass(frozen=True)
class Parcel:
    id: str
    block_id: str
    polygon: Polygon
    frontages: tuple[Frontage, ...] = ()
    kind: str = "regular"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def width_m(self) -> float:
        x0, _, x1, _ = self.bounds
        return x1 - x0

    @property
    def depth_m(self) -> float:
        _, y0, _, y1 = self.bounds
        return y1 - y0

    @property
    def area_m2(self) -> float:
        return (
            abs(
                sum(
                    x0 * y1 - x1 * y0
                    for (x0, y0), (x1, y1) in zip(
                        self.polygon, self.polygon[1:] + self.polygon[:1], strict=True
                    )
                )
            )
            / 2
        )


@dataclass(frozen=True)
class DistrictBuilding:
    """Existing building and the attributes needed for renewal/fire-risk screening."""

    id: str
    parcel_id: str
    cx: float
    cy: float
    width_m: float
    depth_m: float
    height_m: float
    floors: int
    built_year: int
    structure: str
    fire_resistance: str
    use: str = "residential"
    theta: float = 0.0

    @property
    def footprint_area_m2(self) -> float:
        return self.width_m * self.depth_m

    @property
    def age_cohort(self) -> str:
        if self.built_year <= 1980:
            return "pre_1981"
        if self.built_year <= 2000:
            return "1981_2000"
        return "post_2000"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.corners]
        ys = [point[1] for point in self.corners]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def corners(self) -> Polygon:
        """Counter-clockwise footprint corners, including ``theta`` rotation."""
        ct, st = math.cos(self.theta), math.sin(self.theta)
        corners = []
        for x, y in (
            (-self.width_m / 2, -self.depth_m / 2),
            (self.width_m / 2, -self.depth_m / 2),
            (self.width_m / 2, self.depth_m / 2),
            (-self.width_m / 2, self.depth_m / 2),
        ):
            corners.append((self.cx + ct * x - st * y, self.cy + st * x + ct * y))
        return tuple(corners)


@dataclass(frozen=True)
class AccessAssessment:
    parcel_id: str
    legal_frontage_m: float
    max_continuous_legal_frontage_m: float
    path_frontage_m: float
    best_road_width_m: float
    individual_rebuildable: bool
    setback_required_m: float
    access_risk: float
    feasibility: float


@dataclass(frozen=True)
class PriorityWeights:
    """Weights for renewal *need*; feasibility remains a separately reported quantity."""

    age: float = 0.25
    structure: float = 0.18
    fire_resistance: float = 0.17
    access: float = 0.15
    spacing: float = 0.15
    parcel_coverage: float = 0.10
    feasibility_adjustment: float = 0.15


@dataclass(frozen=True)
class RenewalCandidate:
    rank: int
    building_id: str
    parcel_id: str
    need_score: float
    feasibility_score: float
    priority_score: float
    individual_rebuildable: bool
    recommended_action: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DistrictSummary:
    parcels: int
    buildings: int
    roads: int
    footprint_coverage: float
    net_building_coverage: float
    gross_floor_area_m2: float
    gross_floor_area_ratio: float
    building_density_per_ha: float
    legal_access_rate: float
    individual_rebuildable_rate: float
    pre_1981_rate: float
    timber_rate: float
    non_fire_resistant_rate: float


@dataclass(frozen=True)
class MorphologySummary:
    """Topology/geometry diagnostics used to reject grid-like synthetic districts."""

    non_axis_aligned_road_rate: float
    road_orientation_entropy: float
    dead_end_road_rate: float
    irregular_parcel_rate: float
    parcel_area_cv: float
    flag_lot_rate: float
    back_lot_rate: float


@dataclass(frozen=True)
class MokumitsuConfig:
    """Generation controls in metres, calibrated for a compact 100 m-class study area."""

    width_m: float = 100.0
    height_m: float = 100.0
    through_roads_x: int = 2
    through_roads_y: int = 2
    perimeter_road_width_m: float = 6.0
    legal_street_width_m: tuple[float, float] = (4.2, 5.5)
    narrow_street_width_m: tuple[float, float] = (2.2, 3.8)
    narrow_street_probability: float = 0.58
    target_lot_frontage_m: float = 7.2
    target_lot_depth_m: float = 9.0
    minimum_lot_width_m: float = 4.5
    minimum_lot_depth_m: float = 5.5
    lot_size_jitter: float = 0.10
    access_path_probability: float = 0.65
    access_path_width_m: tuple[float, float] = (1.3, 2.2)
    building_setback_m: tuple[float, float] = (0.25, 0.55)
    target_net_building_coverage: float = 0.54
    building_coverage_jitter: float = 0.055
    historic_routes: int = 2
    branch_alleys: int = 8
    dead_end_probability: float = 0.68
    loop_probability: float = 0.20
    target_lot_area_m2: float = 78.0
    minimum_lot_area_m2: float = 32.0
    flag_lot_probability: float = 0.52
    reference_year: int = 2026


@dataclass(frozen=True)
class MokumitsuDistrict:
    width_m: float
    height_m: float
    seed: int
    reference_year: int
    roads: tuple[RoadSegment, ...]
    parcels: tuple[Parcel, ...]
    buildings: tuple[DistrictBuilding, ...]
    schema_version: int = 2
    generator: str = "organic"

    def road(self, road_id: str) -> RoadSegment:
        return next(r for r in self.roads if r.id == road_id)

    def parcel(self, parcel_id: str) -> Parcel:
        return next(p for p in self.parcels if p.id == parcel_id)

    def access(self, parcel_id: str) -> AccessAssessment:
        return assess_access(self, self.parcel(parcel_id))

    def normalized_rects(
        self,
        domain_height_m: float = 60.0,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """Buildings in the normalized (cx, cy, w, d, h, theta) wind contract.

        Horizontal coordinates are normalized by the district plan dimensions;
        building heights are normalized by the explicitly physical CFD domain
        height. Mixing those scales silently flattened the former wind geometry.
        """

        if not np.isfinite(domain_height_m) or domain_height_m <= 0:
            raise ValueError("domain_height_m must be finite and positive")
        tallest = max((building.height_m for building in self.buildings), default=0.0)
        if tallest >= domain_height_m:
            raise ValueError("domain_height_m must exceed the tallest building")
        return [
            (
                b.cx / self.width_m,
                b.cy / self.height_m,
                b.width_m / self.width_m,
                b.depth_m / self.height_m,
                b.height_m / domain_height_m,
                b.theta,
            )
            for b in self.buildings
        ]

    def heightmap(self, res: int = 128, domain_height_m: float = 60.0) -> np.ndarray:
        return geometry.rasterize(
            self.normalized_rects(domain_height_m),
            res,
            res,
            L=1.0,
        )

    def summary(self) -> DistrictSummary:
        return district_summary(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> MokumitsuDistrict:
        roads = tuple(
            RoadSegment(
                id=r["id"],
                start=tuple(r["start"]),
                end=tuple(r["end"]),
                width_m=r["width_m"],
                legal_class=r["legal_class"],
                kind=r.get("kind", "street"),
                dead_end=r.get("dead_end", False),
            )
            for r in data["roads"]
        )
        parcels = tuple(
            Parcel(
                id=p["id"],
                block_id=p["block_id"],
                polygon=tuple(tuple(q) for q in p["polygon"]),
                frontages=tuple(Frontage(**f) for f in p.get("frontages", ())),
                kind=p.get("kind", "regular"),
            )
            for p in data["parcels"]
        )
        buildings = tuple(DistrictBuilding(**b) for b in data["buildings"])
        return cls(
            width_m=data["width_m"],
            height_m=data["height_m"],
            seed=data["seed"],
            reference_year=data["reference_year"],
            roads=roads,
            parcels=parcels,
            buildings=buildings,
            schema_version=data.get("schema_version", 1),
            generator=data.get("generator", "grid"),
        )


def save_district(district: MokumitsuDistrict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(district.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_district(path: str | Path) -> MokumitsuDistrict:
    return MokumitsuDistrict.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def assess_access(district: MokumitsuDistrict, parcel: Parcel) -> AccessAssessment:
    roads = {r.id: r for r in district.roads}
    legal = [f for f in parcel.frontages if roads[f.road_id].is_legal_road]
    paths = [f for f in parcel.frontages if not roads[f.road_id].is_legal_road]
    legal_length = sum(f.length_m for f in legal)
    max_legal_length = max((f.length_m for f in legal), default=0.0)
    path_length = sum(f.length_m for f in paths)
    best_width = max((roads[f.road_id].width_m for f in legal), default=0.0)
    individual = max_legal_length >= 2.0
    setbacks = [
        max(0.0, (4.0 - roads[f.road_id].width_m) / 2)
        for f in legal
        if roads[f.road_id].legal_class == ARTICLE_42_2
    ]
    if not legal:
        risk = 1.0
    elif max_legal_length < 2.0:
        risk = 0.75
    elif all(roads[f.road_id].legal_class == ARTICLE_42_2 for f in legal):
        risk = 0.35
    else:
        risk = 0.0
    if individual:
        feasibility = 1.0
    elif legal_length > 0:
        feasibility = 0.45
    elif path_length >= 2.0:
        feasibility = 0.25
    else:
        feasibility = 0.10
    return AccessAssessment(
        parcel_id=parcel.id,
        legal_frontage_m=legal_length,
        max_continuous_legal_frontage_m=max_legal_length,
        path_frontage_m=path_length,
        best_road_width_m=best_width,
        individual_rebuildable=individual,
        setback_required_m=max(setbacks, default=0.0),
        access_risk=risk,
        feasibility=feasibility,
    )


def road_adjacency(district: MokumitsuDistrict) -> dict[str, tuple[str, ...]]:
    """Road-edge adjacency at crossings/endpoints (the topology part of the street graph)."""
    out: dict[str, set[str]] = {r.id: set() for r in district.roads}
    for i, a in enumerate(district.roads):
        for b in district.roads[i + 1 :]:
            if _segments_intersect(a.start, a.end, b.start, b.end):
                out[a.id].add(b.id)
                out[b.id].add(a.id)
    return {key: tuple(sorted(value)) for key, value in out.items()}


def renewal_priorities(
    district: MokumitsuDistrict, weights: PriorityWeights | None = None
) -> list[RenewalCandidate]:
    """Rank buildings while keeping renewal need and implementation feasibility explicit."""
    weights = weights or PriorityWeights()
    parcels = {p.id: p for p in district.parcels}
    accesses = {p.id: assess_access(district, p) for p in district.parcels}
    raw = []
    weight_total = (
        weights.age
        + weights.structure
        + weights.fire_resistance
        + weights.access
        + weights.spacing
        + weights.parcel_coverage
    )
    if weight_total <= 0:
        raise ValueError("renewal need weights must sum to a positive value")

    for building in district.buildings:
        parcel = parcels[building.parcel_id]
        access = accesses[parcel.id]
        age_score = _age_vulnerability(building.built_year)
        structure_score = {"timber": 1.0, "steel": 0.45, "reinforced_concrete": 0.15}.get(
            building.structure, 0.60
        )
        fire_score = {
            "none": 1.0,
            "quasi_fire_resistant": 0.45,
            "fire_resistant": 0.10,
        }.get(building.fire_resistance, 0.65)
        gap = _nearest_building_gap(building, district.buildings)
        spacing_score = float(np.clip((3.0 - gap) / 3.0, 0.0, 1.0))
        coverage = building.footprint_area_m2 / max(parcel.area_m2, 1e-9)
        coverage_score = float(np.clip((coverage - 0.35) / 0.45, 0.0, 1.0))
        need = (
            weights.age * age_score
            + weights.structure * structure_score
            + weights.fire_resistance * fire_score
            + weights.access * access.access_risk
            + weights.spacing * spacing_score
            + weights.parcel_coverage * coverage_score
        ) / weight_total
        priority = need * (
            1.0
            - weights.feasibility_adjustment
            + weights.feasibility_adjustment * access.feasibility
        )
        reasons = []
        if building.built_year <= 1980:
            reasons.append("pre_1981")
        elif building.built_year <= 2000:
            reasons.append("pre_2000_timber_detail_period")
        if building.structure == "timber":
            reasons.append("timber")
        if building.fire_resistance == "none":
            reasons.append("non_fire_resistant")
        if access.legal_frontage_m == 0:
            reasons.append("no_legal_frontage")
        elif access.max_continuous_legal_frontage_m < 2.0:
            reasons.append("frontage_under_2m")
        elif access.setback_required_m > 0:
            reasons.append("article_42_2_setback")
        if gap < 1.5:
            reasons.append("tight_building_gap")

        if need >= 0.58 and access.individual_rebuildable:
            action = "early_individual_rebuild"
        elif need >= 0.58:
            action = "joint_rebuild_or_access_improvement"
        elif access.individual_rebuildable:
            action = "planned_individual_rebuild"
        else:
            action = "retain_or_group_later"
        raw.append((priority, need, access, building, action, tuple(reasons)))

    raw.sort(key=lambda item: (-item[0], item[3].id))
    return [
        RenewalCandidate(
            rank=rank,
            building_id=building.id,
            parcel_id=building.parcel_id,
            need_score=float(need),
            feasibility_score=float(access.feasibility),
            priority_score=float(priority),
            individual_rebuildable=access.individual_rebuildable,
            recommended_action=action,
            reasons=reasons,
        )
        for rank, (priority, need, access, building, action, reasons) in enumerate(raw, 1)
    ]


def district_summary(district: MokumitsuDistrict) -> DistrictSummary:
    area = district.width_m * district.height_m
    access = [assess_access(district, p) for p in district.parcels]
    buildings = district.buildings
    n = max(len(buildings), 1)
    footprint_area = sum(b.footprint_area_m2 for b in buildings)
    parcel_area = sum(parcel.area_m2 for parcel in district.parcels)
    floor_area = sum(b.footprint_area_m2 * b.floors for b in buildings)
    return DistrictSummary(
        parcels=len(district.parcels),
        buildings=len(buildings),
        roads=len(district.roads),
        footprint_coverage=footprint_area / area,
        net_building_coverage=footprint_area / max(parcel_area, 1e-9),
        gross_floor_area_m2=floor_area,
        gross_floor_area_ratio=floor_area / area,
        building_density_per_ha=len(buildings) / area * 10_000.0,
        legal_access_rate=sum(a.legal_frontage_m > 0 for a in access) / max(len(access), 1),
        individual_rebuildable_rate=sum(a.individual_rebuildable for a in access)
        / max(len(access), 1),
        pre_1981_rate=sum(b.built_year <= 1980 for b in buildings) / n,
        timber_rate=sum(b.structure == "timber" for b in buildings) / n,
        non_fire_resistant_rate=sum(b.fire_resistance == "none" for b in buildings) / n,
    )


def morphology_summary(district: MokumitsuDistrict) -> MorphologySummary:
    """Measure whether a generated district has genuinely non-grid morphology."""
    angles = np.asarray(
        [math.atan2(r.end[1] - r.start[1], r.end[0] - r.start[0]) % math.pi for r in district.roads]
    )
    if angles.size:
        distance_to_axis = np.minimum.reduce(
            (np.abs(angles), np.abs(angles - math.pi / 2), np.abs(angles - math.pi))
        )
        non_axis = float(np.mean(distance_to_axis > math.radians(5.0)))
        hist, _ = np.histogram(angles, bins=12, range=(0.0, math.pi))
        probabilities = hist[hist > 0] / hist.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)) / math.log(12.0))
    else:
        non_axis = entropy = 0.0

    irregular = []
    for parcel in district.parcels:
        edge_angles = [
            math.atan2(y1 - y0, x1 - x0) % math.pi
            for (x0, y0), (x1, y1) in zip(
                parcel.polygon, parcel.polygon[1:] + parcel.polygon[:1], strict=True
            )
        ]
        axis_aligned = all(
            min(abs(a), abs(a - math.pi / 2), abs(a - math.pi)) <= math.radians(2.0)
            for a in edge_angles
        )
        irregular.append(len(parcel.polygon) != 4 or not axis_aligned)
    areas = np.asarray([parcel.area_m2 for parcel in district.parcels], dtype=float)
    area_cv = float(areas.std() / areas.mean()) if areas.size and areas.mean() else 0.0
    n_parcels = max(len(district.parcels), 1)
    return MorphologySummary(
        non_axis_aligned_road_rate=non_axis,
        road_orientation_entropy=entropy,
        dead_end_road_rate=sum(road.dead_end for road in district.roads)
        / max(len(district.roads), 1),
        irregular_parcel_rate=sum(irregular) / n_parcels,
        parcel_area_cv=area_cv,
        flag_lot_rate=sum(parcel.kind == "flag_lot" for parcel in district.parcels) / n_parcels,
        back_lot_rate=sum(parcel.kind == "back_lot" for parcel in district.parcels) / n_parcels,
    )


def validate_district(district: MokumitsuDistrict) -> tuple[str, ...]:
    """Return structural validation errors; an empty tuple means the district is consistent."""
    errors = []
    road_ids = [r.id for r in district.roads]
    parcel_ids = [p.id for p in district.parcels]
    building_ids = [b.id for b in district.buildings]
    for name, ids in (("road", road_ids), ("parcel", parcel_ids), ("building", building_ids)):
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate {name} id")
    roads = set(road_ids)
    parcels = {p.id: p for p in district.parcels}
    for parcel in district.parcels:
        shape = ShapelyPolygon(parcel.polygon)
        if parcel.area_m2 <= 0 or not shape.is_valid:
            errors.append(f"{parcel.id}: non-positive area")
        for frontage in parcel.frontages:
            if frontage.road_id not in roads:
                errors.append(f"{parcel.id}: unknown road {frontage.road_id}")
    for building in district.buildings:
        parcel = parcels.get(building.parcel_id)
        if parcel is None:
            errors.append(f"{building.id}: unknown parcel {building.parcel_id}")
            continue
        parcel_shape = ShapelyPolygon(parcel.polygon)
        footprint = ShapelyPolygon(building.corners)
        if not parcel_shape.buffer(1e-7).covers(footprint):
            errors.append(f"{building.id}: footprint outside parcel")
    return tuple(errors)


def generate_mokumitsu(seed: int = 0, config: MokumitsuConfig | None = None) -> MokumitsuDistrict:
    """Generate the research-default, non-orthogonal synthetic 木密 district."""
    from .organic import generate_organic_mokumitsu

    return generate_organic_mokumitsu(seed, config)


def generate_mokumitsu_grid(
    seed: int = 0, config: MokumitsuConfig | None = None
) -> MokumitsuDistrict:
    """Legacy rectilinear baseline retained only for tests and controlled comparisons."""
    cfg = config or MokumitsuConfig()
    _validate_config(cfg)
    rng = np.random.default_rng(seed)
    vertical = _axis_roads("v", cfg.width_m, cfg.height_m, cfg.through_roads_x, cfg, rng)
    horizontal = _axis_roads("h", cfg.height_m, cfg.width_m, cfg.through_roads_y, cfg, rng)
    roads = [r for _, r in vertical] + [r for _, r in horizontal]
    parcels: list[Parcel] = []
    block_legacy: dict[str, float] = {}

    for ix, ((xl, left), (xr, right)) in enumerate(zip(vertical, vertical[1:], strict=False)):
        for iy, ((yb, bottom), (yt, top)) in enumerate(
            zip(horizontal, horizontal[1:], strict=False)
        ):
            x0, x1 = xl + left.width_m / 2, xr - right.width_m / 2
            y0, y1 = yb + bottom.width_m / 2, yt - top.width_m / 2
            if x1 - x0 < cfg.minimum_lot_width_m or y1 - y0 < cfg.minimum_lot_depth_m:
                continue
            block_id = f"block_{ix}_{iy}"
            block_legacy[block_id] = float(rng.beta(2.0, 2.0))
            cells = _parcel_cells(x0, x1, y0, y1, cfg, rng)
            path = _make_access_path(block_id, cells, bottom, top, left, right, cfg, rng)
            if path is not None:
                roads.append(path)
            for cell_id, cell in enumerate(cells):
                cx0, cy0, cx1, cy1, col, row, nx, ny = cell
                frontages = []
                if col == 0:
                    frontages.append(Frontage(left.id, cy1 - cy0, "west"))
                if col == nx - 1:
                    frontages.append(Frontage(right.id, cy1 - cy0, "east"))
                if row == 0:
                    frontages.append(Frontage(bottom.id, cx1 - cx0, "south"))
                if row == ny - 1:
                    frontages.append(Frontage(top.id, cx1 - cx0, "north"))
                if path is not None:
                    frontages.extend(_path_frontages(path, cell))
                parcels.append(
                    Parcel(
                        id=f"parcel_{ix}_{iy}_{cell_id}",
                        block_id=block_id,
                        polygon=((cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)),
                        frontages=tuple(frontages),
                    )
                )

    road_lookup = {r.id: r for r in roads}
    buildings = tuple(
        _building_for_parcel(
            parcel,
            number,
            block_legacy[parcel.block_id],
            road_lookup,
            cfg,
            rng,
        )
        for number, parcel in enumerate(parcels)
    )
    district = MokumitsuDistrict(
        width_m=cfg.width_m,
        height_m=cfg.height_m,
        seed=seed,
        reference_year=cfg.reference_year,
        roads=tuple(roads),
        parcels=tuple(parcels),
        buildings=buildings,
        generator="grid",
    )
    errors = validate_district(district)
    if errors:
        raise RuntimeError("invalid generated district: " + "; ".join(errors))
    return district


def _validate_config(cfg: MokumitsuConfig) -> None:
    if cfg.width_m <= 0 or cfg.height_m <= 0:
        raise ValueError("district dimensions must be positive")
    if cfg.through_roads_x < 0 or cfg.through_roads_y < 0:
        raise ValueError("road counts cannot be negative")
    if not 0 <= cfg.narrow_street_probability <= 1:
        raise ValueError("narrow_street_probability must be in [0, 1]")
    if cfg.historic_routes < 1 or cfg.branch_alleys < 0:
        raise ValueError("organic road counts are invalid")
    for name, value in (
        ("dead_end_probability", cfg.dead_end_probability),
        ("loop_probability", cfg.loop_probability),
        ("flag_lot_probability", cfg.flag_lot_probability),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if cfg.minimum_lot_area_m2 <= 0 or cfg.target_lot_area_m2 <= cfg.minimum_lot_area_m2:
        raise ValueError("lot areas must satisfy 0 < minimum < target")
    if not 0 < cfg.target_net_building_coverage < 0.85:
        raise ValueError("target_net_building_coverage must be in (0, 0.85)")
    if cfg.building_coverage_jitter < 0:
        raise ValueError("building_coverage_jitter cannot be negative")


def _axis_roads(axis, length, cross_length, count, cfg, rng):
    spacing = length / (count + 1)
    positions = [
        (i + 1) * spacing + float(rng.uniform(-0.16, 0.16)) * spacing for i in range(count)
    ]
    positions.sort()
    out = []
    all_positions = [0.0, *positions, length]
    for i, position in enumerate(all_positions):
        perimeter = i in (0, len(all_positions) - 1)
        if perimeter:
            width = cfg.perimeter_road_width_m
            legal_class, kind = ARTICLE_42_1, "perimeter"
        elif rng.random() < cfg.narrow_street_probability:
            width = float(rng.uniform(*cfg.narrow_street_width_m))
            legal_class, kind = ARTICLE_42_2, "narrow_street"
        else:
            width = float(rng.uniform(*cfg.legal_street_width_m))
            legal_class, kind = ARTICLE_42_1, "street"
        start = (position, 0.0) if axis == "v" else (0.0, position)
        end = (position, cross_length) if axis == "v" else (cross_length, position)
        out.append(
            (
                position,
                RoadSegment(
                    id=f"road_{axis}_{i}",
                    start=start,
                    end=end,
                    width_m=width,
                    legal_class=legal_class,
                    kind=kind,
                ),
            )
        )
    return out


def _partition(lo, hi, target, minimum, jitter, rng):
    length = hi - lo
    count = max(1, round(length / target))
    count = min(count, max(1, int(length / (minimum * 1.15))))
    weights = rng.uniform(1.0 - jitter, 1.0 + jitter, count)
    widths = length * weights / weights.sum()
    edges = [lo]
    for width in widths:
        edges.append(edges[-1] + float(width))
    edges[-1] = hi
    return edges


def _parcel_cells(x0, x1, y0, y1, cfg, rng):
    xs = _partition(
        x0,
        x1,
        cfg.target_lot_frontage_m,
        cfg.minimum_lot_width_m,
        cfg.lot_size_jitter,
        rng,
    )
    ys = _partition(
        y0,
        y1,
        cfg.target_lot_depth_m,
        cfg.minimum_lot_depth_m,
        cfg.lot_size_jitter,
        rng,
    )
    cells = []
    nx, ny = len(xs) - 1, len(ys) - 1
    for row in range(ny):
        for col in range(nx):
            cells.append((xs[col], ys[row], xs[col + 1], ys[row + 1], col, row, nx, ny))
    return cells


def _make_access_path(block_id, cells, bottom, top, left, right, cfg, rng):
    if not cells or rng.random() >= cfg.access_path_probability:
        return None
    nx, ny = cells[0][6], cells[0][7]
    if nx < 3 and ny < 3:
        return None
    x0 = min(c[0] for c in cells)
    y0 = min(c[1] for c in cells)
    x1 = max(c[2] for c in cells)
    y1 = max(c[3] for c in cells)
    width = float(rng.uniform(*cfg.access_path_width_m))
    if nx >= 3 and (ny < 3 or rng.random() < 0.5):
        split = sorted({c[2] for c in cells if c[4] < nx - 1})[int(rng.integers(0, nx - 1))]
        from_bottom = bool(rng.integers(0, 2))
        reach = float(rng.uniform(0.55, 0.82)) * (y1 - y0)
        start = (split, bottom.start[1] if from_bottom else top.start[1])
        end = (split, y0 + reach if from_bottom else y1 - reach)
    else:
        split = sorted({c[3] for c in cells if c[5] < ny - 1})[int(rng.integers(0, ny - 1))]
        from_left = bool(rng.integers(0, 2))
        reach = float(rng.uniform(0.55, 0.82)) * (x1 - x0)
        start = (left.start[0] if from_left else right.start[0], split)
        end = (x0 + reach if from_left else x1 - reach, split)
    return RoadSegment(
        id=f"path_{block_id}",
        start=start,
        end=end,
        width_m=width,
        legal_class=NON_ROAD,
        kind="access_path",
        dead_end=True,
    )


def _path_frontages(path, cell):
    x0, y0, x1, y1, *_ = cell
    eps = 1e-8
    out = []
    if abs(path.start[0] - path.end[0]) < eps:
        x = path.start[0]
        lo, hi = sorted((path.start[1], path.end[1]))
        overlap = max(0.0, min(y1, hi) - max(y0, lo))
        if overlap > 0 and abs(x1 - x) < eps:
            out.append(Frontage(path.id, overlap, "east"))
        if overlap > 0 and abs(x0 - x) < eps:
            out.append(Frontage(path.id, overlap, "west"))
    else:
        y = path.start[1]
        lo, hi = sorted((path.start[0], path.end[0]))
        overlap = max(0.0, min(x1, hi) - max(x0, lo))
        if overlap > 0 and abs(y1 - y) < eps:
            out.append(Frontage(path.id, overlap, "north"))
        if overlap > 0 and abs(y0 - y) < eps:
            out.append(Frontage(path.id, overlap, "south"))
    return out


def _building_for_parcel(parcel, number, legacy, roads, cfg, rng):
    legal = [f for f in parcel.frontages if roads[f.road_id].is_legal_road]
    if not legal:
        access_risk = 1.0
    elif max(f.length_m for f in legal) < 2.0:
        access_risk = 0.75
    elif all(roads[f.road_id].legal_class == ARTICLE_42_2 for f in legal):
        access_risk = 0.35
    else:
        access_risk = 0.0
    smallness = float(np.clip(1.0 - parcel.area_m2 / 120.0, 0.0, 1.0))
    oldness = float(
        np.clip(
            0.45 * legacy + 0.25 * access_risk + 0.15 * smallness + 0.15 * rng.random(),
            0.0,
            1.0,
        )
    )
    if oldness >= 0.62:
        built_year = int(rng.integers(1948, 1981))
        timber_p = 0.92
    elif oldness >= 0.34:
        built_year = int(rng.integers(1981, 2001))
        timber_p = 0.82
    else:
        built_year = int(rng.integers(2001, cfg.reference_year + 1))
        timber_p = 0.62
    timber_p = min(0.98, timber_p + 0.06 * access_risk)
    if rng.random() < timber_p:
        structure = "timber"
    else:
        structure = "steel" if rng.random() < 0.70 else "reinforced_concrete"

    if structure == "timber":
        if built_year <= 1980:
            fire_resistance = str(rng.choice(["none", "quasi_fire_resistant"], p=[0.86, 0.14]))
        elif built_year <= 2000:
            fire_resistance = str(rng.choice(["none", "quasi_fire_resistant"], p=[0.58, 0.42]))
        else:
            fire_resistance = str(
                rng.choice(["none", "quasi_fire_resistant", "fire_resistant"], p=[0.22, 0.68, 0.10])
            )
    else:
        fire_resistance = str(
            rng.choice(["quasi_fire_resistant", "fire_resistant"], p=[0.48, 0.52])
        )

    setback = float(rng.uniform(*cfg.building_setback_m))
    usable_w = max(2.5, parcel.width_m - 2 * setback)
    usable_d = max(3.0, parcel.depth_m - 2 * setback)
    target_coverage = float(np.clip(0.48 + 0.28 * oldness + rng.uniform(-0.04, 0.04), 0.42, 0.78))
    target_area = parcel.area_m2 * target_coverage
    aspect = float(rng.uniform(0.78, 1.28))
    width = math.sqrt(target_area * aspect)
    depth = math.sqrt(target_area / aspect)
    fit = min(1.0, usable_w / width, usable_d / depth)
    width, depth = width * fit, depth * fit
    px0, py0, px1, py1 = parcel.bounds
    slack_x = max(0.0, usable_w - width)
    slack_y = max(0.0, usable_d - depth)
    cx = (px0 + px1) / 2 + float(rng.uniform(-0.25, 0.25)) * slack_x
    cy = (py0 + py1) / 2 + float(rng.uniform(-0.25, 0.25)) * slack_y
    if built_year <= 1980:
        floors = int(rng.choice([1, 2], p=[0.25, 0.75]))
    else:
        floors = int(rng.choice([2, 3], p=[0.82, 0.18]))
    return DistrictBuilding(
        id=f"building_{number}",
        parcel_id=parcel.id,
        cx=cx,
        cy=cy,
        width_m=width,
        depth_m=depth,
        height_m=2.8 * floors,
        floors=floors,
        built_year=built_year,
        structure=structure,
        fire_resistance=fire_resistance,
    )


def _age_vulnerability(year):
    if year <= 1980:
        return 1.0
    if year <= 2000:
        return 0.65
    return 0.25


def _nearest_building_gap(building, buildings):
    best = math.inf
    footprint = ShapelyPolygon(building.corners)
    for other in buildings:
        if other.id == building.id:
            continue
        best = min(best, footprint.distance(ShapelyPolygon(other.corners)))
    return best if math.isfinite(best) else 999.0


def _segments_intersect(a, b, c, d):
    return LineString((a, b)).distance(LineString((c, d))) <= 1e-7
