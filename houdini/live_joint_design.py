"""Live Houdini SOP for one editable two-to-four-parcel renewal project."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import hou
import numpy as np

from mokumitsu import (
    FireScenario,
    JointBuildingPlacement,
    JointDesignParameters,
    JointOpenSpace,
    JointRenewalPolicy,
    MokumitsuDistrict,
    SummerWindScenario,
    WindDirection,
    default_joint_design_parameters,
    evaluate_joint_design,
    generate_mokumitsu,
    joint_design_cache_key,
    joint_renewal_candidates,
    load_model,
    parameterized_joint_redevelopment,
    prepare_joint_design_baseline,
)

_MODELS: dict[tuple[str, str], object] = {}
_BASELINES: dict[tuple[object, ...], object] = {}
_DISTRICTS: dict[int, object] = {}
_CANDIDATES: dict[tuple[object, ...], tuple] = {}
_HOUDINI_CACHE_SCHEMA = "bgeo-v2"


class _WorkerClient:
    def __init__(self, python: Path, script: Path):
        environment = os.environ.copy()
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV"):
            environment.pop(name, None)
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["VIRTUAL_ENV"] = str(python.parent.parent)
        environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
        worker_tmp = script.parents[1] / "outputs" / "worker_tmp"
        worker_tmp.mkdir(parents=True, exist_ok=True)
        environment["TMP"] = str(worker_tmp)
        environment["TEMP"] = str(worker_tmp)
        environment.setdefault("WANDB_MODE", "disabled")
        environment.setdefault("WANDB_SILENT", "true")
        self.process = subprocess.Popen(
            [str(python), "-u", str(script)],
            cwd=str(script.parents[1]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def request(self, payload: dict) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(f"joint-design worker exited with code {self.process.returncode}")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("joint-design worker pipes are unavailable")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            self.process.poll()
            details = ""
            if self.process.stderr is not None and self.process.returncode is not None:
                details = self.process.stderr.read()[-2000:].strip()
            raise RuntimeError(
                "joint-design worker closed its response stream"
                + (f": {details}" if details else "")
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "joint-design worker failed"))
        return response

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.write('{"command":"shutdown"}\n')
                self.process.stdin.flush()
            self.process.wait(timeout=2.0)
        except Exception:
            self.process.terminate()


def _worker_registry() -> dict[tuple[str, str], _WorkerClient]:
    registry = getattr(hou.session, "_mokumitsu_joint_design_workers", None)
    if registry is None:
        registry = {}
        hou.session._mokumitsu_joint_design_workers = registry
    return registry


def restart_worker() -> None:
    """Close persistent preview workers; the next cook starts a clean process."""
    registry = _worker_registry()
    for worker in registry.values():
        worker.close()
    registry.clear()


def reset_parameters(node: hou.Node) -> None:
    """Reset massing controls to a searched, known-feasible central placement."""
    district, policy, candidate = _selection(node)
    primary = float(node.evalParm("wind_primary_deg"))
    parameters = default_joint_design_parameters(
        district,
        candidate,
        policy,
        wind_direction_deg=primary,
    )
    node.parm("center_u").set(parameters.center_u)
    node.parm("center_v").set(parameters.center_v)
    node.parm("building_coverage").set(parameters.building_coverage)
    node.parm("aspect_ratio").set(parameters.aspect_ratio)
    node.parm("rotation_deg").set(parameters.rotation_deg)
    node.parm("floors").set(parameters.floors)


def cook(node: hou.Node) -> None:
    """Cook full geometry, using a deterministic per-design bgeo cache when possible."""
    start = time.perf_counter()
    geo = node.geometry()
    geo.clear()
    _attributes(geo)
    try:
        district, policy, candidate = _selection(node)
        parameters = _parameters(node)
        primary = float(node.evalParm("wind_primary_deg"))
        updated, placement = parameterized_joint_redevelopment(
            district,
            candidate,
            parameters,
            policy,
            wind_direction_deg=primary,
        )
    except Exception as exc:
        district = _district(int(node.evalParm("seed")))
        _render_district(geo, district, (), None, None)
        _set_detail(geo, "status", "invalid_geometry")
        _set_detail(geo, "cache_hit", 0)
        _set_detail(geo, "evaluation_ms", 1000.0 * (time.perf_counter() - start))
        _set_detail(geo, "hud_text", f"INFEASIBLE MASSING\n{type(exc).__name__}: {exc}")
        return

    wind_scenario = _wind_scenario(node)
    fire_scenario = _fire_scenario(node, district.seed)
    worker_error = ""
    if bool(node.evalParm("use_worker")):
        try:
            _cook_from_worker(
                node,
                geo,
                start,
                district,
                policy,
                candidate,
                parameters,
                wind_scenario,
                fire_scenario,
            )
            return
        except Exception as exc:
            worker_error = f"{type(exc).__name__}: {exc}"
    try:
        model = _model(node, wind_scenario.model_name)
    except Exception as exc:
        _render_district(geo, updated, candidate.parcel_ids, placement, None)
        _design_details(
            geo,
            candidate,
            placement,
            None,
            status="geometry_only",
            elapsed_ms=1000.0 * (time.perf_counter() - start),
        )
        _set_detail(
            geo,
            "hud_text",
            _geometry_hud(candidate, placement)
            + f"\nWIND/FIRE UNAVAILABLE: {type(exc).__name__}: {exc}"
            + (f"\nWORKER ERROR: {worker_error}" if worker_error else ""),
        )
        return

    key = joint_design_cache_key(
        district,
        candidate,
        parameters,
        policy,
        wind_scenario,
        fire_scenario,
        model,
    )
    display_resolution = int(node.evalParm("display_resolution"))
    cache_path = _cache_path(key, display_resolution)
    if bool(node.evalParm("disk_cache")) and cache_path.exists():
        geo.loadFromFile(str(cache_path))
        _set_detail(geo, "cache_hit", 1)
        _set_detail(geo, "evaluation_ms", 1000.0 * (time.perf_counter() - start))
        _set_detail(geo, "cache_path", str(cache_path))
        return

    baseline_slot = (
        district.seed,
        repr(wind_scenario),
        repr(fire_scenario),
        getattr(model, "resolved_stem", wind_scenario.model_name),
        getattr(model, "_sha256", None),
    )
    baseline = _BASELINES.get(baseline_slot)
    if baseline is None:
        baseline = prepare_joint_design_baseline(
            district,
            wind_scenario,
            fire_scenario,
            model,
        )
        _BASELINES[baseline_slot] = baseline
    result = evaluate_joint_design(
        district,
        candidate,
        parameters,
        policy,
        wind_scenario,
        fire_scenario,
        model,
        baseline=baseline,
        include_wind_field=True,
    )
    if result.wind_ratio is None or result.heightmap is None:
        raise RuntimeError("interactive evaluation did not return a wind field")
    ratio, heightmap = result.wind_ratio, result.heightmap
    _render_district(
        geo,
        result.district,
        candidate.parcel_ids,
        placement,
        (ratio, heightmap, display_resolution),
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    _design_details(
        geo,
        candidate,
        placement,
        result,
        status="screened",
        elapsed_ms=elapsed_ms,
    )
    _set_detail(geo, "cache_key", key)
    _set_detail(geo, "cache_path", str(cache_path))
    _set_detail(geo, "cache_hit", 0)
    if worker_error:
        _set_detail(geo, "worker_error", worker_error)
    if bool(node.evalParm("disk_cache")):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            geo.saveToFile(str(cache_path))
        except Exception as exc:
            _set_detail(geo, "cache_write_error", str(exc))


def _district(seed: int):
    district = _DISTRICTS.get(seed)
    if district is None:
        district = generate_mokumitsu(seed=seed)
        _DISTRICTS[seed] = district
    return district


def _policy(node: hou.Node) -> JointRenewalPolicy:
    return JointRenewalPolicy(
        year=int(node.evalParm("renewal_year")),
        candidate_limit=24,
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
        target_building_coverage=0.30,
        minimum_floors=int(node.evalParm("minimum_floors")),
        maximum_floors=int(node.evalParm("maximum_floors")),
        target_road_width_m=float(node.evalParm("target_road_width")),
        ventilation_corridor_width_m=float(node.evalParm("corridor_width")),
        shared_open_space_fraction=float(node.evalParm("open_space_fraction")),
        placement_grid=7,
        placement_variants=3,
    )


def _selection(node: hou.Node):
    district = _district(int(node.evalParm("seed")))
    policy = _policy(node)
    parcel_count = int(node.evalParm("cluster_size"))
    candidate_slot = (district.seed, policy, parcel_count)
    candidates = _CANDIDATES.get(candidate_slot)
    if candidates is None:
        candidates = tuple(
            candidate
            for candidate in joint_renewal_candidates(district, policy)
            if candidate.parcel_count == parcel_count
        )
        _CANDIDATES[candidate_slot] = candidates
    if not candidates:
        raise ValueError(f"no {parcel_count}-parcel candidate for seed {district.seed}")
    rank = max(0, min(int(node.evalParm("candidate_rank")), len(candidates) - 1))
    return district, policy, candidates[rank]


def _parameters(node: hou.Node) -> JointDesignParameters:
    return JointDesignParameters(
        center_u=float(node.evalParm("center_u")),
        center_v=float(node.evalParm("center_v")),
        building_coverage=float(node.evalParm("building_coverage")),
        aspect_ratio=float(node.evalParm("aspect_ratio")),
        rotation_deg=float(node.evalParm("rotation_deg")),
        floors=int(node.evalParm("floors")),
    )


def _wind_scenario(node: hou.Node) -> SummerWindScenario:
    primary_weight = float(node.evalParm("wind_primary_weight"))
    return SummerWindScenario(
        directions=(
            WindDirection(float(node.evalParm("wind_primary_deg")), primary_weight),
            WindDirection(float(node.evalParm("wind_secondary_deg")), 1.0 - primary_weight),
        ),
        model_name=str(node.evalParm("wind_model")),
    )


def _fire_scenario(node: hou.Node, seed: int) -> FireScenario:
    return FireScenario(
        wind_direction_deg=float(node.evalParm("fire_wind_deg")),
        wind_speed_m_s=float(node.evalParm("fire_wind_speed")),
        runs=int(node.evalParm("fire_runs")),
        seed=seed,
    )


def _cook_from_worker(
    node,
    geo,
    start,
    district,
    policy,
    candidate,
    parameters,
    wind_scenario,
    fire_scenario,
) -> None:
    checkpoint_dir = _checkpoint_dir(node)
    python = _worker_python(node)
    project_root = Path(hou.expandString("$HIP")).resolve().parent
    script = project_root / "scripts" / "joint_design_worker.py"
    if not script.exists():
        raise FileNotFoundError(f"joint-design worker script is missing: {script}")
    worker_key = (str(python), str(script))
    workers = _worker_registry()
    worker = workers.get(worker_key)
    if worker is None or worker.process.poll() is not None:
        worker = _WorkerClient(python, script)
        workers[worker_key] = worker
    request = {
        "seed": district.seed,
        "policy": asdict(policy),
        "cluster_size": candidate.parcel_count,
        "candidate_rank": int(node.evalParm("candidate_rank")),
        "parameters": asdict(parameters),
        "wind_model": wind_scenario.model_name,
        "wind_directions": [asdict(direction) for direction in wind_scenario.directions],
        "fire_scenario": asdict(fire_scenario),
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(node.evalParm("worker_device")),
        "display_resolution": int(node.evalParm("display_resolution")),
    }
    response = worker.request(request)
    display_resolution = int(response["display_resolution"])
    cache_path = _cache_path(response["cache_key"], display_resolution)
    if bool(node.evalParm("disk_cache")) and cache_path.exists():
        geo.loadFromFile(str(cache_path))
        _set_detail(geo, "cache_hit", 1)
        _set_detail(geo, "worker_cache_hit", int(response["worker_cache_hit"]))
        _set_detail(geo, "worker_ms", float(response["worker_ms"]))
        _set_detail(geo, "evaluation_ms", 1000.0 * (time.perf_counter() - start))
        _set_detail(geo, "cache_path", str(cache_path))
        return

    updated = MokumitsuDistrict.from_dict(response["district"])
    placement_data = dict(response["placement"])
    placement_data["open_spaces"] = tuple(
        JointOpenSpace(**space) for space in placement_data.get("open_spaces", ())
    )
    placement = JointBuildingPlacement(**placement_data)
    ratio = np.asarray(response["wind_ratio"], dtype=float)
    solid = np.asarray(response["solid"], dtype=float)
    _render_district(
        geo,
        updated,
        candidate.parcel_ids,
        placement,
        (ratio, solid, display_resolution),
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    _worker_design_details(
        geo,
        candidate,
        placement,
        response,
        elapsed_ms=elapsed_ms,
    )
    _set_detail(geo, "cache_key", response["cache_key"])
    _set_detail(geo, "cache_path", str(cache_path))
    _set_detail(geo, "cache_hit", 0)
    if bool(node.evalParm("disk_cache")):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            geo.saveToFile(str(cache_path))
        except Exception as exc:
            _set_detail(geo, "cache_write_error", str(exc))


def _worker_python(node: hou.Node) -> Path:
    raw = hou.expandString(str(node.evalParm("worker_python"))).strip()
    if not raw or "$" in raw:
        raw = os.environ.get("MOKUMITSU_PYTHON", "").strip()
    if not raw:
        project_root = Path(hou.expandString("$HIP")).resolve().parent
        raw = str(project_root / ".venv" / "Scripts" / "python.exe")
    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError("set MOKUMITSU_PYTHON or Worker Python to a project environment")
    return path


def _checkpoint_dir(node: hou.Node) -> Path:
    raw = hou.expandString(str(node.evalParm("checkpoint_dir"))).strip()
    if not raw or "$" in raw:
        raw = os.environ.get("MOKUMITSU_CHECKPOINT_DIR", "").strip()
    if not raw:
        raise FileNotFoundError("set MOKUMITSU_CHECKPOINT_DIR or Checkpoint directory")
    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    return path


def _model(node: hou.Node, model_name: str):
    key = (model_name, str(_checkpoint_dir(node)))
    model = _MODELS.get(key)
    if model is None:
        model = load_model(model_name, key[1])
        _MODELS[key] = model
    return model


def _cache_path(key: str, display_resolution: int) -> Path:
    hip_dir = Path(hou.expandString("$HIP")).resolve()
    filename = f"{_HOUDINI_CACHE_SCHEMA}.{key}.r{display_resolution}.bgeo.sc"
    return hip_dir / "cache" / "joint_design" / filename


def _attributes(geo: hou.Geometry) -> None:
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
    geo.addAttrib(hou.attribType.Point, "Alpha", 1.0)
    for name in ("kind", "status", "building_id", "parcel_id", "space_kind"):
        geo.addAttrib(hou.attribType.Prim, name, "")


def _point(geo: hou.Geometry, x, y, z, colour, alpha=1.0):
    point = geo.createPoint()
    point.setPosition(hou.Vector3(float(x), float(y), float(z)))
    point.setAttribValue("Cd", tuple(float(value) for value in colour))
    point.setAttribValue("Alpha", float(alpha))
    return point


def _polygon(
    geo: hou.Geometry,
    coordinates,
    colour,
    kind: str,
    *,
    closed: bool = True,
    alpha: float = 1.0,
    status: str = "",
    parcel_id: str = "",
    space_kind: str = "",
):
    primitive = geo.createPolygon()
    primitive.setIsClosed(closed)
    for x, y, z in coordinates:
        primitive.addVertex(_point(geo, x, y, z, colour, alpha))
    primitive.setAttribValue("kind", kind)
    primitive.setAttribValue("status", status)
    primitive.setAttribValue("parcel_id", parcel_id)
    primitive.setAttribValue("space_kind", space_kind)
    return primitive


def _road_polygon(road, z=-0.02):
    x0, y0 = road.start
    x1, y1 = road.end
    dx, dy = x1 - x0, y1 - y0
    length = max(math.hypot(dx, dy), 1e-9)
    nx, ny = -dy / length, dx / length
    hx = 0.5 * road.width_m * nx
    hy = 0.5 * road.width_m * ny
    return (
        (x0 + hx, y0 + hy, z),
        (x1 + hx, y1 + hy, z),
        (x1 - hx, y1 - hy, z),
        (x0 - hx, y0 - hy, z),
    )


def _building_box(geo, building, colour, status):
    corners = building.corners
    bottom = [_point(geo, x, y, 0.08, colour) for x, y in corners]
    top = [_point(geo, x, y, 0.08 + building.height_m, colour) for x, y in corners]
    points = bottom + top
    for face in (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ):
        primitive = geo.createPolygon()
        primitive.setIsClosed(True)
        for index in face:
            primitive.addVertex(points[index])
        primitive.setAttribValue("kind", "building")
        primitive.setAttribValue("status", status)
        primitive.setAttribValue("building_id", building.id)
        primitive.setAttribValue("parcel_id", building.parcel_id)


def _render_district(geo, district, source_parcel_ids, placement, wind):
    selected = set(source_parcel_ids)
    if wind is not None:
        display_resolution = int(wind[2]) if len(wind) > 2 else 32
        _wind_cells(geo, district, wind[0], wind[1], display_resolution)
    for road in district.roads:
        _polygon(geo, _road_polygon(road), (0.16, 0.18, 0.22), "road", alpha=0.8)
    for parcel in district.parcels:
        is_joint = placement is not None and parcel.id == placement.joint_parcel_id
        colour = (0.92, 0.34, 0.78) if is_joint else (0.42, 0.48, 0.58)
        _polygon(
            geo,
            [(x, y, 0.04) for x, y in parcel.polygon],
            colour,
            "parcel_outline",
            closed=True,
            alpha=0.9,
            status="joint_site" if is_joint else "parcel",
            parcel_id=parcel.id,
        )
    if selected:
        for parcel_id in selected:
            try:
                parcel = district.parcel(parcel_id)
            except StopIteration:
                continue
            _polygon(
                geo,
                [(x, y, 0.055) for x, y in parcel.polygon],
                (1.0, 0.42, 0.08),
                "source_parcel_outline",
                closed=True,
                status="source_parcel",
                parcel_id=parcel_id,
            )
    if placement is not None:
        for space in placement.open_spaces:
            colour = {
                "road_widening": (1.0, 0.68, 0.08),
                "ventilation_corridor": (0.04, 0.86, 0.92),
                "shared_court": (0.16, 0.78, 0.30),
                "pocket_park": (0.12, 0.72, 0.24),
            }.get(space.kind, (0.45, 0.72, 0.42))
            _polygon(
                geo,
                [(x, y, 0.07) for x, y in space.polygon],
                colour,
                "open_space",
                alpha=0.82,
                status="allocated",
                space_kind=space.kind,
            )
    for building in district.buildings:
        is_joint = placement is not None and building.id == placement.joint_building_id
        _building_box(
            geo,
            building,
            (0.92, 0.18, 0.68) if is_joint else (0.52, 0.56, 0.64),
            "joint_building" if is_joint else "existing",
        )


def _wind_cells(geo, district, ratio, heightmap, display_res=32):
    display_res = min(display_res, int(np.asarray(ratio).shape[0]), int(np.asarray(ratio).shape[1]))
    ratio = _block_mean(np.asarray(ratio, dtype=float), display_res)
    solid = _block_mean((np.asarray(heightmap) > 1e-7).astype(float), display_res) > 0.25
    cell_x = district.width_m / display_res
    cell_y = district.height_m / display_res
    vertices = []
    for iy in range(display_res + 1):
        row = []
        for ix in range(display_res + 1):
            adjacent = ratio[
                max(iy - 1, 0) : min(iy + 1, display_res),
                max(ix - 1, 0) : min(ix + 1, display_res),
            ]
            value = float(adjacent.mean())
            row.append(
                _point(
                    geo,
                    ix * cell_x,
                    iy * cell_y,
                    0.005,
                    _wind_colour(value),
                    0.78,
                )
            )
        vertices.append(row)
    for iy in range(display_res):
        for ix in range(display_res):
            if solid[iy, ix]:
                continue
            value = float(ratio[iy, ix])
            primitive = geo.createPolygon()
            primitive.setIsClosed(True)
            for point in (
                vertices[iy][ix],
                vertices[iy][ix + 1],
                vertices[iy + 1][ix + 1],
                vertices[iy + 1][ix],
            ):
                primitive.addVertex(point)
            primitive.setAttribValue("kind", "wind_cell")
            primitive.setAttribValue("status", _wind_class(value))


def _block_mean(field: np.ndarray, target: int) -> np.ndarray:
    height, width = field.shape
    if height % target or width % target:
        rows = np.array_split(np.arange(height), target)
        columns = np.array_split(np.arange(width), target)
        return np.asarray(
            [[field[np.ix_(row, column)].mean() for column in columns] for row in rows]
        )
    return field.reshape(target, height // target, target, width // target).mean(axis=(1, 3))


def _mix(a, b, amount):
    amount = max(0.0, min(float(amount), 1.0))
    return tuple(a[index] + (b[index] - a[index]) * amount for index in range(3))


def _wind_colour(value):
    if value < 0.3:
        return _mix((0.02, 0.05, 0.28), (0.04, 0.25, 0.95), value / 0.3)
    if value < 1.0:
        return _mix((0.04, 0.25, 0.95), (0.08, 0.82, 0.38), (value - 0.3) / 0.7)
    if value < 1.3:
        return _mix((0.08, 0.82, 0.38), (1.0, 0.82, 0.06), (value - 1.0) / 0.3)
    return _mix((1.0, 0.82, 0.06), (0.92, 0.08, 0.04), (value - 1.3) / 0.7)


def _wind_class(value):
    if value < 0.3:
        return "weak"
    if value > 1.3:
        return "strong"
    return "comfortable"


def _design_details(geo, candidate, placement, result, *, status, elapsed_ms):
    _set_detail(geo, "status", status)
    _set_detail(geo, "candidate_parcels", ",".join(candidate.parcel_ids))
    _set_detail(geo, "candidate_parcel_count", candidate.parcel_count)
    _set_detail(geo, "building_width_m", placement.width_m)
    _set_detail(geo, "building_depth_m", placement.depth_m)
    _set_detail(geo, "building_height_m", placement.height_m)
    _set_detail(geo, "building_floors", placement.floors)
    _set_detail(geo, "connected_open_space_m2", placement.connected_open_space_m2)
    _set_detail(geo, "road_dedication_m2", placement.road_widening_area_m2)
    _set_detail(geo, "evaluation_ms", elapsed_ms)
    _set_detail(geo, "cache_hit", 0)
    if result is None:
        return
    _set_detail(geo, "wind_penalty_baseline", result.baseline.objectives.wind_penalty)
    _set_detail(geo, "wind_penalty_design", result.objectives.wind_penalty)
    _set_detail(
        geo,
        "burned_area_baseline_m2",
        result.baseline.objectives.expected_burned_area_m2,
    )
    _set_detail(geo, "burned_area_design_m2", result.objectives.expected_burned_area_m2)
    _set_detail(geo, "floor_area_retention", result.selected_floor_area_retention)
    _set_detail(geo, "gross_floor_area_m2", result.gross_floor_area_m2)
    _set_detail(geo, "hud_text", _result_hud(result))


def _worker_design_details(geo, candidate, placement, response, *, elapsed_ms):
    _design_details(
        geo,
        candidate,
        placement,
        None,
        status="screened",
        elapsed_ms=elapsed_ms,
    )
    metrics = response["metrics"]
    baseline = metrics["baseline_objectives"]
    design = metrics["objectives"]
    _set_detail(geo, "wind_penalty_baseline", float(baseline["wind_penalty"]))
    _set_detail(geo, "wind_penalty_design", float(design["wind_penalty"]))
    _set_detail(
        geo,
        "burned_area_baseline_m2",
        float(baseline["expected_burned_area_m2"]),
    )
    _set_detail(
        geo,
        "burned_area_design_m2",
        float(design["expected_burned_area_m2"]),
    )
    _set_detail(geo, "floor_area_retention", float(metrics["selected_floor_area_retention"]))
    _set_detail(geo, "gross_floor_area_m2", float(metrics["gross_floor_area_m2"]))
    _set_detail(geo, "worker_device", str(response["device"]))
    _set_detail(geo, "worker_cache_hit", int(response["worker_cache_hit"]))
    _set_detail(geo, "worker_ms", float(response["worker_ms"]))
    _set_detail(geo, "hud_text", _worker_hud(candidate, placement, response))


def _geometry_hud(candidate, placement):
    return (
        f"JOINT DESIGN | {candidate.parcel_count} parcels | {','.join(candidate.parcel_ids)}\n"
        f"MASSING {placement.width_m:.1f} x {placement.depth_m:.1f} m | "
        f"{placement.floors}F / {placement.height_m:.1f} m | "
        f"GFA {placement.floor_area_m2:.0f} m2\n"
        f"CONNECTED OPEN SPACE {placement.connected_open_space_m2:.1f} m2 | "
        f"ROAD LAND {placement.road_widening_area_m2:.1f} m2"
    )


def _worker_hud(candidate, placement, response):
    metrics = response["metrics"]
    baseline = metrics["baseline_objectives"]
    design = metrics["objectives"]
    return (
        _geometry_hud(candidate, placement)
        + f"\nFLOOR AREA RETENTION {metrics['selected_floor_area_retention']:.2f} | "
        f"WIND PENALTY {baseline['wind_penalty']:.3f} -> {design['wind_penalty']:.3f} | "
        f"EXPECTED BURNED AREA {baseline['expected_burned_area_m2']:.0f} -> "
        f"{design['expected_burned_area_m2']:.0f} m2\n"
        f"{response['device'].upper()} WORKER {response['worker_ms']:.0f} ms | "
        "SCREENING PREVIEW: verify shortlisted designs with XLB and validated fire analysis."
    )


def _result_hud(result):
    baseline = result.baseline.objectives
    design = result.objectives
    return (
        _geometry_hud(result.candidate, result.placement)
        + f"\nFLOOR AREA RETENTION {result.selected_floor_area_retention:.2f} | "
        f"WIND PENALTY {baseline.wind_penalty:.3f} -> {design.wind_penalty:.3f} | "
        f"EXPECTED BURNED AREA {baseline.expected_burned_area_m2:.0f} -> "
        f"{design.expected_burned_area_m2:.0f} m2\n"
        "SCREENING PREVIEW: shortlist only; verify selected designs with XLB "
        "and validated fire analysis."
    )


def _set_detail(geo: hou.Geometry, name: str, value) -> None:
    if geo.findGlobalAttrib(name) is None:
        if isinstance(value, str):
            default = ""
        elif isinstance(value, int):
            default = 0
        else:
            default = 0.0
        geo.addAttrib(hou.attribType.Global, name, default)
    geo.setGlobalAttribValue(name, value)
