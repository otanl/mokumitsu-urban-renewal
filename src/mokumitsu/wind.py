"""Summer pedestrian-wind screening for a ``MokumitsuDistrict``.

The fast FNO reports normalized pedestrian-level speed.  This module separates stagnant wind
(summer ventilation deficit) from strong wind and evaluates three spatially distinct outdoor
zones: streets, parcel open space, and the public-facing band around buildings.  Directional
evaluation is currently exact for cardinal directions by rotating the square height-map; arbitrary
angles belong in the later padded wind-rose/XLB verification path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .comfort import Thresholds
from .district import MokumitsuDistrict
from .model import FnoModel, load_model
from .renewal import RenewalTrajectory


@dataclass(frozen=True)
class WindDirection:
    direction_deg: float
    weight: float = 1.0


@dataclass(frozen=True)
class SummerWindScenario:
    """A site-specific cardinal wind rose; direction is where the air moves toward."""

    directions: tuple[WindDirection, ...] = (WindDirection(0.0, 1.0),)
    model_name: str = "residential"
    thresholds: Thresholds = Thresholds()
    building_edge_distance_m: float = 3.0


@dataclass(frozen=True)
class WindZoneMetrics:
    cells: int
    mean_index: float
    dead_fraction: float
    strong_fraction: float
    comfortable_fraction: float


@dataclass(frozen=True)
class DirectionalWindMetrics:
    direction_deg: float
    weight: float
    zones: dict[str, WindZoneMetrics]


@dataclass(frozen=True)
class DistrictWindMetrics:
    model_name: str
    resolution: int
    reference_speed: float
    directions: tuple[DirectionalWindMetrics, ...]
    weighted_zones: dict[str, WindZoneMetrics]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_district_wind(
    district: MokumitsuDistrict,
    scenario: SummerWindScenario | None = None,
    model: FnoModel | None = None,
    reference_speed: float | None = None,
) -> DistrictWindMetrics:
    """Evaluate weak/strong-wind fractions for all configured cardinal directions."""
    scenario = scenario or SummerWindScenario()
    _validate_scenario(scenario)
    model = model or load_model(scenario.model_name)
    res = int(model.ny)
    _validate_square_grid(district, model)
    u0 = float(reference_speed if reference_speed is not None else model.reference_speed())
    if not math.isfinite(u0) or u0 <= 0:
        raise ValueError("surrogate reference speed must be positive")

    heightmap = district.heightmap(res)
    masks = district_wind_masks(
        district, res, building_edge_distance_m=scenario.building_edge_distance_m
    )
    total_weight = sum(direction.weight for direction in scenario.directions)
    directional = []
    for direction in scenario.directions:
        quarter_turns = int(round(direction.direction_deg / 90.0)) % 4
        # With row index representing +y, np.rot90(+k) is the physical -k*90° layout rotation
        # that maps a wind vector toward direction_deg back onto the FNO's native +x inflow.
        hm = np.rot90(heightmap, k=quarter_turns).copy()
        rotated_masks = {
            name: np.rot90(mask, k=quarter_turns).copy() for name, mask in masks.items()
        }
        speed = _predict_speed(model, hm)
        zones = {
            name: _zone_metrics(speed, hm, u0, mask, scenario.thresholds)
            for name, mask in rotated_masks.items()
        }
        directional.append(
            DirectionalWindMetrics(
                direction_deg=float(direction.direction_deg % 360.0),
                weight=float(direction.weight / total_weight),
                zones=zones,
            )
        )

    weighted = {name: _weighted_zone(tuple(directional), name) for name in masks}
    return DistrictWindMetrics(
        model_name=scenario.model_name,
        resolution=res,
        reference_speed=u0,
        directions=tuple(directional),
        weighted_zones=weighted,
    )


def predict_directional_wind(
    district: MokumitsuDistrict,
    direction_deg: float = 0.0,
    model: FnoModel | None = None,
    model_name: str = "residential",
    reference_speed: float | None = None,
    building_edge_distance_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    """Return ``(speed, heightmap, masks, u0)`` in the rotated evaluation frame."""
    _quarter_turn(direction_deg)
    model = model or load_model(model_name)
    _validate_square_grid(district, model)
    res = int(model.ny)
    k = int(round(direction_deg / 90.0)) % 4
    heightmap = np.rot90(district.heightmap(res), k=k).copy()
    masks = {
        name: np.rot90(mask, k=k).copy()
        for name, mask in district_wind_masks(
            district, res, building_edge_distance_m=building_edge_distance_m
        ).items()
    }
    u0 = float(reference_speed if reference_speed is not None else model.reference_speed())
    if not math.isfinite(u0) or u0 <= 0:
        raise ValueError("surrogate reference speed must be positive")
    speed = _predict_speed(model, heightmap)
    return speed, heightmap, masks, u0


def weighted_world_wind_ratio(
    district: MokumitsuDistrict,
    scenario: SummerWindScenario | None = None,
    model: FnoModel | None = None,
    reference_speed: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return wind-rose-weighted U/U0 in the district's world orientation.

    predict_directional_wind works in the FNO's +x evaluation frame. This helper
    rotates every directional result back to world coordinates before weighting it,
    which makes the returned scalar field suitable for plan-view visualization.
    """
    scenario = scenario or SummerWindScenario()
    _validate_scenario(scenario)
    model = model or load_model(scenario.model_name)
    _validate_square_grid(district, model)
    u0 = float(reference_speed if reference_speed is not None else model.reference_speed())
    if not math.isfinite(u0) or u0 <= 0:
        raise ValueError("surrogate reference speed must be positive")

    total_weight = sum(direction.weight for direction in scenario.directions)
    ratio = np.zeros((int(model.ny), int(model.nx)), dtype=np.float32)
    for direction in scenario.directions:
        speed, _, _, _ = predict_directional_wind(
            district,
            direction_deg=direction.direction_deg,
            model=model,
            reference_speed=u0,
            building_edge_distance_m=scenario.building_edge_distance_m,
        )
        k = _quarter_turn(direction.direction_deg)
        speed_world = np.rot90(speed, k=-k).copy()
        ratio += np.float32(direction.weight / total_weight) * (speed_world / u0)
    return ratio, district.heightmap(int(model.ny)), u0


def district_wind_masks(
    district: MokumitsuDistrict,
    res: int,
    building_edge_distance_m: float = 3.0,
) -> dict[str, np.ndarray]:
    """Rasterize outdoor analysis zones on the same square grid as the surrogate."""
    if res <= 0 or building_edge_distance_m <= 0:
        raise ValueError("resolution and building-edge distance must be positive")
    xs = (np.arange(res, dtype=float) + 0.5) / res * district.width_m
    ys = (np.arange(res, dtype=float) + 0.5) / res * district.height_m
    gx, gy = np.meshgrid(xs, ys)
    road = np.zeros((res, res), dtype=bool)
    for segment in district.roads:
        x0, y0 = segment.start
        x1, y1 = segment.end
        vx, vy = x1 - x0, y1 - y0
        length2 = max(vx * vx + vy * vy, 1e-12)
        t = np.clip(((gx - x0) * vx + (gy - y0) * vy) / length2, 0.0, 1.0)
        distance2 = (gx - (x0 + t * vx)) ** 2 + (gy - (y0 + t * vy)) ** 2
        road |= distance2 <= (segment.width_m / 2) ** 2

    parcel_union = unary_union([Polygon(parcel.polygon) for parcel in district.parcels])
    parcel = np.asarray(contains_xy(parcel_union, gx, gy), dtype=bool)
    heightmap = district.heightmap(res)
    building = heightmap > 1e-7
    open_cells = ~building
    edge = (
        _dilate_physical(
            building,
            building_edge_distance_m,
            district.width_m / res,
            district.height_m / res,
        )
        & open_cells
    )
    return {
        "all_outdoor": open_cells,
        "roads": road & open_cells,
        "parcel_open_space": parcel & ~road & open_cells,
        "building_edge": edge,
    }


def add_wind_to_trajectory(
    trajectory: RenewalTrajectory,
    scenario: SummerWindScenario | None = None,
    model: FnoModel | None = None,
) -> RenewalTrajectory:
    """Return the same renewal states with summer-wind metrics added phase by phase."""
    scenario = scenario or SummerWindScenario()
    model = model or load_model(scenario.model_name)
    u0 = float(model.reference_speed())
    phases = []
    for phase in trajectory.phases:
        wind = evaluate_district_wind(
            phase.district, scenario=scenario, model=model, reference_speed=u0
        )
        all_outdoor = wind.weighted_zones["all_outdoor"]
        roads = wind.weighted_zones["roads"]
        yards = wind.weighted_zones["parcel_open_space"]
        edge = wind.weighted_zones["building_edge"]
        metrics = replace(
            phase.metrics,
            wind_mean_index=all_outdoor.mean_index,
            wind_dead_fraction=all_outdoor.dead_fraction,
            wind_strong_fraction=all_outdoor.strong_fraction,
            road_dead_fraction=roads.dead_fraction,
            yard_dead_fraction=yards.dead_fraction,
            building_edge_strong_fraction=edge.strong_fraction,
        )
        phases.append(replace(phase, metrics=metrics))
    return replace(trajectory, phases=tuple(phases))


def _zone_metrics(
    speed: np.ndarray,
    heightmap: np.ndarray,
    u0: float,
    region: np.ndarray,
    thresholds: Thresholds,
) -> WindZoneMetrics:
    mask = np.asarray(region, dtype=bool) & (heightmap <= 1e-7)
    values = speed[mask]
    if values.size == 0:
        return WindZoneMetrics(0, 0.0, 0.0, 0.0, 0.0)
    low = thresholds.dead * u0
    high = thresholds.strong * u0
    return WindZoneMetrics(
        cells=int(values.size),
        mean_index=float(values.mean() / u0),
        dead_fraction=float(np.mean(values < low)),
        strong_fraction=float(np.mean(values > high)),
        comfortable_fraction=float(np.mean((values >= low) & (values <= high))),
    )


def _weighted_zone(directions: tuple[DirectionalWindMetrics, ...], name: str) -> WindZoneMetrics:
    zones = [direction.zones[name] for direction in directions]
    weights = [direction.weight for direction in directions]
    return WindZoneMetrics(
        cells=int(round(sum(w * zone.cells for w, zone in zip(weights, zones, strict=True)))),
        mean_index=float(sum(w * zone.mean_index for w, zone in zip(weights, zones, strict=True))),
        dead_fraction=float(
            sum(w * zone.dead_fraction for w, zone in zip(weights, zones, strict=True))
        ),
        strong_fraction=float(
            sum(w * zone.strong_fraction for w, zone in zip(weights, zones, strict=True))
        ),
        comfortable_fraction=float(
            sum(w * zone.comfortable_fraction for w, zone in zip(weights, zones, strict=True))
        ),
    )


def _dilate_physical(
    mask: np.ndarray,
    radius_m: float,
    cell_width_m: float,
    cell_height_m: float,
) -> np.ndarray:
    # The source mask represents finite raster cells, not zero-area points.  Measure from each
    # target cell centre to the source cell rectangle so a sub-cell-width analysis band still
    # produces the physically adjacent outdoor ring on coarse surrogate grids.
    radius_x = int(math.ceil(radius_m / cell_width_m + 0.5))
    radius_y = int(math.ceil(radius_m / cell_height_m + 0.5))
    out = np.zeros_like(mask)
    ny, nx = mask.shape
    for dy in range(-radius_y, radius_y + 1):
        for dx in range(-radius_x, radius_x + 1):
            distance_x = max(abs(dx) * cell_width_m - cell_width_m / 2, 0.0)
            distance_y = max(abs(dy) * cell_height_m - cell_height_m / 2, 0.0)
            if distance_x**2 + distance_y**2 > radius_m**2:
                continue
            sy0, sy1 = max(0, -dy), min(ny, ny - dy)
            sx0, sx1 = max(0, -dx), min(nx, nx - dx)
            dy0, dy1 = sy0 + dy, sy1 + dy
            dx0, dx1 = sx0 + dx, sx1 + dx
            out[dy0:dy1, dx0:dx1] |= mask[sy0:sy1, sx0:sx1]
    return out


def _predict_speed(model: FnoModel, heightmap: np.ndarray) -> np.ndarray:
    speed = np.asarray(model.predict(heightmap), dtype=np.float32)
    if speed.shape != heightmap.shape:
        raise ValueError(f"surrogate returned shape {speed.shape}; expected {heightmap.shape}")
    if not np.isfinite(speed).all():
        raise ValueError("surrogate returned non-finite wind speed")
    return np.maximum(speed, 0.0)


def _validate_square_grid(district: MokumitsuDistrict, model: FnoModel) -> None:
    if int(model.nx) != int(model.ny):
        raise ValueError("district wind evaluation requires a square surrogate grid")
    if not math.isclose(district.width_m, district.height_m, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("cardinal rotation currently requires a physically square district")


def _quarter_turn(direction_deg: float) -> int:
    if not math.isfinite(direction_deg):
        raise ValueError("wind direction must be finite")
    quarter = direction_deg / 90.0
    if abs(quarter - round(quarter)) > 1e-8:
        raise ValueError("only cardinal wind directions (multiples of 90 degrees) are supported")
    return int(round(quarter)) % 4


def _validate_scenario(scenario: SummerWindScenario) -> None:
    if not scenario.directions:
        raise ValueError("summer wind scenario needs at least one direction")
    if any(
        not math.isfinite(direction.weight) or direction.weight <= 0
        for direction in scenario.directions
    ):
        raise ValueError("wind-direction weights must be finite and positive")
    for direction in scenario.directions:
        _quarter_turn(direction.direction_deg)
    if (
        not math.isfinite(scenario.thresholds.dead)
        or not math.isfinite(scenario.thresholds.strong)
        or not 0 <= scenario.thresholds.dead < scenario.thresholds.strong
    ):
        raise ValueError("wind thresholds must satisfy 0 <= dead < strong")
    if (
        not math.isfinite(scenario.building_edge_distance_m)
        or scenario.building_edge_distance_m <= 0
    ):
        raise ValueError("building_edge_distance_m must be positive")
