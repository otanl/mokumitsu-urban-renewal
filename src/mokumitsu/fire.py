"""Fast district-fire screening on the shared 木密 building/road graph.

This is deliberately a planning model, not combustion CFD and not a code-compliance tool. It
represents fire spread as stochastic first-passage times on directed building-to-building edges.
Edge times respond to separation, construction/fire resistance, intervening road width and wind
alignment. The result is cheap enough for redevelopment sequencing; selected plans should later be
checked with an established urban-fire model.

Wind direction follows the rest of winddesign: degrees in the direction the air moves, with
``0`` meaning flow toward ``+x`` (not the meteorological "from" convention).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree

from .district import (
    DistrictBuilding,
    MokumitsuDistrict,
    renewal_priorities,
)


@dataclass(frozen=True)
class FireScenario:
    wind_direction_deg: float = 0.0
    wind_speed_m_s: float = 5.0
    horizon_min: float = 180.0
    ignitions: int = 1
    runs: int = 500
    seed: int = 0
    max_gap_m: float = 12.0
    base_delay_min: float = 20.0
    distance_decay_m: float = 4.0
    barrier_decay_m: float = 4.0
    wind_gain: float = 0.12
    uncertainty_sigma: float = 0.45
    large_fire_fraction: float = 0.25


@dataclass(frozen=True)
class SpreadEdge:
    source_id: str
    target_id: str
    gap_m: float
    direction_deg: float
    barrier_width_m: float
    mean_delay_min: float


@dataclass(frozen=True)
class FireRiskResult:
    scenario: FireScenario
    building_ids: tuple[str, ...]
    burn_probability: tuple[float, ...]
    source_consequence: tuple[float, ...]
    expected_burned_buildings: float
    p95_burned_buildings: float
    probability_large_fire: float
    expected_burned_area_m2: float

    def probability_for(self, building_id: str) -> float:
        return self.burn_probability[self.building_ids.index(building_id)]

    def consequence_for(self, building_id: str) -> float:
        return self.source_consequence[self.building_ids.index(building_id)]

    def to_dict(self) -> dict:
        return {
            "scenario": asdict(self.scenario),
            "building_ids": list(self.building_ids),
            "burn_probability": dict(zip(self.building_ids, self.burn_probability, strict=True)),
            "source_consequence": dict(
                zip(self.building_ids, self.source_consequence, strict=True)
            ),
            "expected_burned_buildings": self.expected_burned_buildings,
            "p95_burned_buildings": self.p95_burned_buildings,
            "probability_large_fire": self.probability_large_fire,
            "expected_burned_area_m2": self.expected_burned_area_m2,
        }


@dataclass(frozen=True)
class FireAwareRenewalCandidate:
    rank: int
    building_id: str
    parcel_id: str
    base_priority_score: float
    fire_exposure: float
    source_consequence: float
    combined_score: float
    feasibility_score: float
    recommended_action: str
    reasons: tuple[str, ...]


def build_fire_graph(
    district: MokumitsuDistrict, scenario: FireScenario | None = None
) -> tuple[SpreadEdge, ...]:
    """Build directed spread edges for all building pairs within the scenario cutoff."""
    scenario = scenario or FireScenario()
    _validate_scenario(scenario, len(district.buildings))
    edges = []
    buildings = district.buildings
    footprints = tuple(ShapelyPolygon(building.corners) for building in buildings)
    tree = STRtree(footprints)
    for i, source in enumerate(buildings):
        target_indices = sorted(
            int(index)
            for index in tree.query(
                footprints[i],
                predicate="dwithin",
                distance=scenario.max_gap_m,
            )
            if int(index) > i
        )
        for target_index in target_indices:
            target = buildings[target_index]
            gap = float(footprints[i].distance(footprints[target_index]))
            barrier = _barrier_width(district, source, target)
            edges.append(_directed_edge(source, target, gap, barrier, scenario))
            edges.append(_directed_edge(target, source, gap, barrier, scenario))
    return tuple(edges)


def simulate_fire_spread(
    district: MokumitsuDistrict,
    scenario: FireScenario | None = None,
    ignition_ids: tuple[str, ...] | None = None,
) -> FireRiskResult:
    """Monte-Carlo first-passage spread plus deterministic source-consequence screening."""
    scenario = scenario or FireScenario()
    count = len(district.buildings)
    _validate_scenario(scenario, count)
    ids = tuple(b.id for b in district.buildings)
    index = {building_id: i for i, building_id in enumerate(ids)}
    if ignition_ids is not None:
        unknown = set(ignition_ids) - set(ids)
        if unknown:
            raise ValueError(f"unknown ignition buildings: {sorted(unknown)}")
        if not ignition_ids:
            raise ValueError("ignition_ids cannot be empty")
    edges = build_fire_graph(district, scenario)
    adjacency = _adjacency(edges, index)

    consequence = []
    for source in range(count):
        arrival = _arrival_times(adjacency, (source,), scenario.horizon_min)
        consequence.append(sum(t <= scenario.horizon_min for t in arrival) / count)

    rng = np.random.default_rng(scenario.seed)
    burn_hits = np.zeros(count, dtype=np.int64)
    burned_counts = np.empty(scenario.runs, dtype=np.int64)
    burned_areas = np.empty(scenario.runs, dtype=np.float64)
    areas = np.array([b.footprint_area_m2 for b in district.buildings], dtype=np.float64)
    fixed_sources = tuple(index[x] for x in ignition_ids) if ignition_ids is not None else None
    for run in range(scenario.runs):
        if fixed_sources is None:
            sources = tuple(
                int(x)
                for x in rng.choice(count, size=min(scenario.ignitions, count), replace=False)
            )
        else:
            sources = fixed_sources
        arrival = _arrival_times(
            adjacency,
            sources,
            scenario.horizon_min,
            rng=rng,
            sigma=scenario.uncertainty_sigma,
        )
        burned = np.asarray(arrival) <= scenario.horizon_min
        burn_hits += burned
        burned_counts[run] = int(burned.sum())
        burned_areas[run] = float(areas[burned].sum())

    threshold = max(1, math.ceil(scenario.large_fire_fraction * count))
    return FireRiskResult(
        scenario=scenario,
        building_ids=ids,
        burn_probability=tuple(float(x) for x in burn_hits / scenario.runs),
        source_consequence=tuple(float(x) for x in consequence),
        expected_burned_buildings=float(burned_counts.mean()),
        p95_burned_buildings=float(np.quantile(burned_counts, 0.95)),
        probability_large_fire=float((burned_counts >= threshold).mean()),
        expected_burned_area_m2=float(burned_areas.mean()),
    )


def fire_aware_renewal_priorities(
    district: MokumitsuDistrict,
    result: FireRiskResult,
    fire_weight: float = 0.35,
) -> list[FireAwareRenewalCandidate]:
    """Blend fire exposure/consequence with the existing need-first renewal ranking."""
    if not 0 <= fire_weight <= 1:
        raise ValueError("fire_weight must be in [0, 1]")
    if set(result.building_ids) != {b.id for b in district.buildings}:
        raise ValueError("fire result does not match district buildings")
    base = renewal_priorities(district)
    exposure_threshold = max(float(np.quantile(result.burn_probability, 0.80)), 1e-9)
    consequence_threshold = max(float(np.quantile(result.source_consequence, 0.80)), 1e-9)
    raw = []
    for candidate in base:
        exposure = result.probability_for(candidate.building_id)
        consequence = result.consequence_for(candidate.building_id)
        fire_score = 0.5 * (exposure + consequence)
        combined = (1.0 - fire_weight) * candidate.priority_score + fire_weight * fire_score
        reasons = list(candidate.reasons)
        if exposure >= exposure_threshold:
            reasons.append("high_fire_exposure")
        if consequence >= consequence_threshold:
            reasons.append("high_spread_consequence")
        raw.append((combined, candidate, exposure, consequence, tuple(reasons)))
    raw.sort(key=lambda item: (-item[0], item[1].building_id))
    return [
        FireAwareRenewalCandidate(
            rank=rank,
            building_id=candidate.building_id,
            parcel_id=candidate.parcel_id,
            base_priority_score=candidate.priority_score,
            fire_exposure=float(exposure),
            source_consequence=float(consequence),
            combined_score=float(combined),
            feasibility_score=candidate.feasibility_score,
            recommended_action=candidate.recommended_action,
            reasons=reasons,
        )
        for rank, (combined, candidate, exposure, consequence, reasons) in enumerate(raw, 1)
    ]


def _validate_scenario(scenario, building_count):
    if building_count <= 0:
        raise ValueError("district needs at least one building")
    if scenario.runs <= 0 or scenario.horizon_min <= 0:
        raise ValueError("runs and horizon_min must be positive")
    if scenario.ignitions <= 0:
        raise ValueError("ignitions must be positive")
    if scenario.max_gap_m <= 0 or scenario.distance_decay_m <= 0:
        raise ValueError("spread distances must be positive")
    if scenario.base_delay_min <= 0 or scenario.barrier_decay_m <= 0:
        raise ValueError("delay/barrier scales must be positive")
    if not 0 < scenario.large_fire_fraction <= 1:
        raise ValueError("large_fire_fraction must be in (0, 1]")


def _directed_edge(source, target, gap, barrier, scenario):
    dx, dy = target.cx - source.cx, target.cy - source.cy
    direction = math.atan2(dy, dx)
    alignment = math.cos(direction - math.radians(scenario.wind_direction_deg))
    wind_factor = math.exp(scenario.wind_gain * scenario.wind_speed_m_s * alignment)
    distance_factor = math.exp(-gap / scenario.distance_decay_m)
    material = math.sqrt(_combustibility(source) * _susceptibility(target))
    barrier_factor = math.exp(-barrier / scenario.barrier_decay_m)
    hazard = max(0.015, distance_factor * material * wind_factor * barrier_factor)
    return SpreadEdge(
        source_id=source.id,
        target_id=target.id,
        gap_m=gap,
        direction_deg=math.degrees(direction) % 360.0,
        barrier_width_m=barrier,
        mean_delay_min=scenario.base_delay_min / hazard,
    )


def _combustibility(building):
    structure = {"timber": 1.0, "steel": 0.38, "reinforced_concrete": 0.16}.get(
        building.structure, 0.60
    )
    resistance = {
        "none": 1.0,
        "quasi_fire_resistant": 0.52,
        "fire_resistant": 0.18,
    }.get(building.fire_resistance, 0.65)
    return structure * resistance


def _susceptibility(building):
    return math.sqrt(_combustibility(building))


def _building_gap(a: DistrictBuilding, b: DistrictBuilding) -> float:
    return float(ShapelyPolygon(a.corners).distance(ShapelyPolygon(b.corners)))


def _barrier_width(district, a, b):
    width = 0.0
    start, end = (a.cx, a.cy), (b.cx, b.cy)
    for road in district.roads:
        if _segments_intersect(start, end, road.start, road.end):
            width = max(width, road.width_m)
    return width


def _segments_intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (abs(o1) < 1e-9 and on_segment(a, c, b))
        or (abs(o2) < 1e-9 and on_segment(a, d, b))
        or (abs(o3) < 1e-9 and on_segment(c, a, d))
        or (abs(o4) < 1e-9 and on_segment(c, b, d))
    )


def _adjacency(edges, index):
    out = [[] for _ in index]
    for edge in edges:
        out[index[edge.source_id]].append((index[edge.target_id], edge.mean_delay_min))
    return out


def _arrival_times(adjacency, sources, horizon, rng=None, sigma=0.0):
    distances = [math.inf] * len(adjacency)
    queue = []
    for source in sources:
        distances[source] = 0.0
        heapq.heappush(queue, (0.0, source))
    log_shift = -0.5 * sigma * sigma
    while queue:
        current, node = heapq.heappop(queue)
        if current != distances[node] or current > horizon:
            continue
        for target, mean_delay in adjacency[node]:
            if rng is None or sigma <= 0:
                delay = mean_delay
            else:
                delay = mean_delay * math.exp(float(rng.normal(log_shift, sigma)))
            arrival = current + delay
            if arrival < distances[target] and arrival <= horizon:
                distances[target] = arrival
                heapq.heappush(queue, (arrival, target))
    return distances
