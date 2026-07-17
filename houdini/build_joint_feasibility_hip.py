"""Build a cached Houdini timeline for the phased joint-renewal feasibility result.

The scene is intentionally separate from houdini_mokumitsu.hip. Frame 1 is the
baseline; frames 2 onward are completed joint projects. It reads the precomputed
feasibility JSON, so playback does not rerun the FNO, fire model, or optimizer.

    $env:PYTHONUTF8=1
    hython houdini/build_joint_feasibility_hip.py
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PROJECT_ROOT / "outputs" / "joint_feasibility.json"


SOP_CODE = r"""
import json, math, os, hou

node = hou.pwd()
geo = node.geometry()
geo.clear()
result_path = hou.expandString(node.evalParm("result_json"))
if not os.path.exists(result_path):
    raise hou.NodeError("Missing feasibility JSON: " + result_path)
with open(result_path, "r", encoding="utf-8") as stream:
    payload = json.load(stream)
if "initial_district" not in payload:
    raise hou.NodeError(
        "The result JSON has no geometry. Re-run evaluate_joint_feasibility.py "
        "with --include-districts, then rebuild the HIP."
    )
phases = payload.get("phases", [])
if any("district" not in phase for phase in phases):
    raise hou.NodeError("Every feasibility phase must include its district geometry.")
wind_visualization = payload.get("wind_visualization")
if not isinstance(wind_visualization, dict):
    raise hou.NodeError(
        "The result JSON has no cached wind field. Re-run evaluate_joint_feasibility.py."
    )
wind_frames = wind_visualization.get("frames", [])
if len(wind_frames) != len(phases) + 1:
    raise hou.NodeError("Cached wind-frame count must equal baseline plus feasibility phases.")
phase_index = max(0, min(int(node.evalParm("active_phase")), len(phases)))
district = payload["initial_district"] if phase_index == 0 else phases[phase_index - 1]["district"]
wind_frame = wind_frames[phase_index]
wind_res = int(wind_visualization["display_resolution"])
wind_ratio_grid = wind_frame["speed_ratio"]
wind_solid_grid = wind_frame["solid"]
if (
    len(wind_ratio_grid) != wind_res
    or len(wind_solid_grid) != wind_res
    or any(len(row) != wind_res for row in wind_ratio_grid)
    or any(len(row) != wind_res for row in wind_solid_grid)
):
    raise hou.NodeError("Cached wind field does not match its display resolution.")
wind_thresholds = wind_visualization["thresholds"]
wind_weak = float(wind_thresholds["weak_below"])
wind_strong = float(wind_thresholds["strong_above"])
show_roads = bool(node.evalParm("show_roads"))
show_parcels = bool(node.evalParm("show_parcels"))
show_open_space = bool(node.evalParm("show_open_space"))
show_next = bool(node.evalParm("show_next"))
height_scale = float(node.evalParm("height_scale"))

geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
geo.addAttrib(hou.attribType.Point, "Alpha", 1.0)
geo.addAttrib(hou.attribType.Prim, "kind", "")
geo.addAttrib(hou.attribType.Prim, "status", "")
geo.addAttrib(hou.attribType.Prim, "building_id", "")
geo.addAttrib(hou.attribType.Prim, "parcel_id", "")
geo.addAttrib(hou.attribType.Prim, "project_id", "")
geo.addAttrib(hou.attribType.Prim, "delivery_phase", 0)
geo.addAttrib(hou.attribType.Prim, "area_m2", 0.0)
geo.addAttrib(hou.attribType.Prim, "wind_ratio", 0.0)
geo.addAttrib(hou.attribType.Prim, "wind_class", "")


def point(x, y, z, colour, alpha=1.0):
    p = geo.createPoint()
    p.setPosition(hou.Vector3(float(x), float(y), float(z)))
    p.setAttribValue("Cd", tuple(float(v) for v in colour))
    p.setAttribValue("Alpha", float(alpha))
    return p


def polygon(coords, colour, kind, closed=True, alpha=1.0, attrs=None):
    prim = geo.createPolygon()
    prim.setIsClosed(closed)
    for x, y, z in coords:
        prim.addVertex(point(x, y, z, colour, alpha))
    prim.setAttribValue("kind", kind)
    if attrs:
        for name, value in attrs.items():
            prim.setAttribValue(name, value)
    return prim


def mix_colour(a, b, t):
    t = max(0.0, min(float(t), 1.0))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def wind_colour(ratio):
    blue = (0.04, 0.22, 0.92)
    green = (0.08, 0.82, 0.38)
    yellow = (1.0, 0.82, 0.06)
    red = (0.92, 0.08, 0.04)
    if ratio < wind_weak:
        return mix_colour((0.02, 0.05, 0.30), blue, ratio / max(wind_weak, 1e-6))
    if ratio < 1.0:
        return mix_colour(blue, green, (ratio - wind_weak) / max(1.0 - wind_weak, 1e-6))
    if ratio < wind_strong:
        return mix_colour(green, yellow, (ratio - 1.0) / max(wind_strong - 1.0, 1e-6))
    return mix_colour(yellow, red, (ratio - wind_strong) / max(0.6, wind_strong * 0.5))


def wind_class(ratio):
    if ratio < wind_weak:
        return "weak"
    if ratio > wind_strong:
        return "strong"
    return "comfortable"


def road_polygon(road, z=-0.015):
    x0, y0 = road["start"]
    x1, y1 = road["end"]
    dx, dy = x1 - x0, y1 - y0
    length = max(math.hypot(dx, dy), 1e-9)
    nx, ny = -dy / length, dx / length
    hx, hy = 0.5 * float(road["width_m"]) * nx, 0.5 * float(road["width_m"]) * ny
    return [
        (x0 + hx, y0 + hy, z),
        (x1 + hx, y1 + hy, z),
        (x1 - hx, y1 - hy, z),
        (x0 - hx, y0 - hy, z),
    ]


def building_corners(building):
    cx, cy = float(building["cx"]), float(building["cy"])
    hx, hy = 0.5 * float(building["width_m"]), 0.5 * float(building["depth_m"])
    theta = float(building.get("theta", 0.0))
    c, s = math.cos(theta), math.sin(theta)
    result = []
    for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        result.append((cx + c * x - s * y, cy + s * x + c * y))
    return result


def building_box(building, colour, status, phase, project_id):
    corners = building_corners(building)
    z0 = 0.06
    z1 = z0 + max(0.25, float(building["height_m"]) * height_scale)
    points = (
        [point(x, y, z0, colour) for x, y in corners]
        + [point(x, y, z1, colour) for x, y in corners]
    )
    attrs = {
        "status": status,
        "building_id": building["id"],
        "parcel_id": building["parcel_id"],
        "project_id": project_id,
        "delivery_phase": int(phase),
    }
    for face in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        prim = geo.createPolygon()
        prim.setIsClosed(True)
        for index in face:
            prim.addVertex(points[index])
        prim.setAttribValue("kind", "building")
        for name, value in attrs.items():
            prim.setAttribValue(name, value)


width = float(district["width_m"])
height = float(district["height_m"])
polygon(
    [(0, 0, -0.08), (width, 0, -0.08), (width, height, -0.08), (0, height, -0.08)],
    (0.12, 0.14, 0.16),
    "ground",
)
cell_width = width / wind_res
cell_height = height / wind_res
for iy in range(wind_res):
    for ix in range(wind_res):
        if wind_solid_grid[iy][ix]:
            continue
        ratio = float(wind_ratio_grid[iy][ix])
        x0, x1 = ix * cell_width, (ix + 1) * cell_width
        y0, y1 = iy * cell_height, (iy + 1) * cell_height
        polygon(
            [(x0, y0, -0.035), (x1, y0, -0.035), (x1, y1, -0.035), (x0, y1, -0.035)],
            wind_colour(ratio),
            "wind_cell",
            attrs={"wind_ratio": ratio, "wind_class": wind_class(ratio)},
        )
if show_roads:
    for road in district["roads"]:
        colour = (0.34, 0.36, 0.39)
        if road.get("kind") == "widened_frontage":
            colour = (0.68, 0.42, 0.12)
        road_coords = road_polygon(road)
        polygon(road_coords, colour, "road", attrs={"project_id": road.get("id", "")})
        polygon(
            road_coords + [road_coords[0]],
            (0.82, 0.86, 0.92),
            "road_outline",
            closed=False,
            attrs={"project_id": road.get("id", "")},
        )

completed = phases[:phase_index]
completed_joint_ids = {phase["joint_building_id"] for phase in completed}
current_joint_id = completed[-1]["joint_building_id"] if completed else ""
next_phase = phases[phase_index] if phase_index < len(phases) else None
next_building_ids = set(next_phase["building_ids"]) if next_phase and show_next else set()
next_parcel_ids = set(next_phase["parcel_ids"]) if next_phase and show_next else set()
joint_project = {phase["joint_building_id"]: phase["project_id"] for phase in completed}
joint_delivery_phase = {phase["joint_building_id"]: int(phase["index"]) for phase in completed}

if show_parcels:
    joint_parcels = {
        building["parcel_id"]
        for building in district["buildings"]
        if building["id"] in completed_joint_ids
    }
    for parcel in district["parcels"]:
        colour = (0.46, 0.50, 0.54)
        status = "parcel"
        if parcel["id"] in next_parcel_ids:
            colour, status = (1.0, 0.38, 0.05), "next_redevelopment"
        elif parcel["id"] in joint_parcels:
            colour, status = (0.76, 0.28, 0.90), "joint_parcel"
        coords = [(x, y, 0.035) for x, y in parcel["polygon"]]
        coords.append(coords[0])
        polygon(
            coords,
            colour,
            "parcel_outline",
            closed=False,
            attrs={"parcel_id": parcel["id"], "status": status},
        )

if show_open_space:
    open_colours = {
        "road_widening": (1.0, 0.58, 0.05),
        "ventilation_corridor": (0.0, 0.82, 1.0),
        "shared_court": (0.36, 0.88, 0.42),
        "pocket_park": (0.10, 0.72, 0.24),
    }
    for completed_phase in completed:
        for space in completed_phase.get("open_spaces", []):
            coords = [(x, y, 0.075) for x, y in space["polygon"]]
            polygon(
                coords,
                open_colours.get(space["kind"], (0.25, 0.72, 0.48)),
                space["kind"],
                alpha=0.92,
                attrs={
                    "status": "delivered_open_space",
                    "project_id": completed_phase["project_id"],
                    "delivery_phase": int(completed_phase["index"]),
                    "area_m2": float(space["area_m2"]),
                },
            )

for building in district["buildings"]:
    building_id = building["id"]
    if building_id == current_joint_id:
        colour, status = (0.92, 0.16, 0.78), "new_joint_building"
    elif building_id in completed_joint_ids:
        colour, status = (0.52, 0.22, 0.76), "completed_joint_building"
    elif building_id in next_building_ids:
        colour, status = (1.0, 0.32, 0.04), "next_redevelopment"
    else:
        colour, status = (0.28, 0.36, 0.46), "existing"
    building_box(
        building,
        colour,
        status,
        joint_delivery_phase.get(building_id, 0),
        joint_project.get(building_id, ""),
    )

legend = [
    ((1.0, 0.32, 0.04), "next"),
    ((0.92, 0.16, 0.78), "new_joint"),
    ((0.52, 0.22, 0.76), "completed_joint"),
    ((0.0, 0.82, 1.0), "wind_corridor"),
    ((0.10, 0.72, 0.24), "park"),
    ((1.0, 0.58, 0.05), "road_dedication"),
]
for index, (colour, name) in enumerate(legend):
    x0 = index * 16.0
    polygon(
        [(x0, -7.0, 0.08), (x0 + 4.0, -7.0, 0.08),
         (x0 + 4.0, -3.5, 0.08), (x0, -3.5, 0.08)],
        colour,
        "legend_swatch",
        attrs={"status": name},
    )

wind_legend = [
    (wind_colour(wind_weak * 0.5), "weak"),
    (wind_colour((wind_weak + 1.0) * 0.5), "below_reference"),
    (wind_colour((1.0 + wind_strong) * 0.5), "above_reference"),
    (wind_colour(wind_strong + 0.45), "strong"),
]
for index, (colour, name) in enumerate(wind_legend):
    x0 = index * 22.0
    polygon(
        [(x0, -14.0, 0.08), (x0 + 4.0, -14.0, 0.08),
         (x0 + 4.0, -10.5, 0.08), (x0, -10.5, 0.08)],
        colour,
        "wind_legend_swatch",
        attrs={"status": name},
    )


def set_detail(name, value, default):
    if geo.findGlobalAttrib(name) is None:
        geo.addAttrib(hou.attribType.Global, name, default)
    geo.setGlobalAttribValue(name, value)


summary = payload["summary"]
wind_min = float(wind_frame["outdoor_min"])
wind_mean = float(wind_frame["outdoor_mean"])
wind_max = float(wind_frame["outdoor_max"])
if phase_index == 0:
    current_year = int(payload["initial_district"]["reference_year"])
    project_id = "baseline"
    stage_name = "BASELINE"
    rights_this = temporary = returned = displaced = 0
    cost_million = cumulative_cost_million = 0.0
    dwellings = int(summary["initial_dwellings"])
    cumulative_rights = 0
    connected_open = cumulative_open = road_area = cumulative_road = 0.0
    wind_before = wind_after = float(phases[0]["wind_objective_before"]) if phases else 0.0
else:
    item = phases[phase_index - 1]
    current_year = int(item["year"])
    project_id = item["project_id"]
    stage_name = "JOINT PROJECT " + str(phase_index)
    rights_this = int(item["rights_count"])
    temporary = int(item["temporary_relocation_dwellings"])
    returned = int(item["return_dwellings"])
    displaced = int(item["permanently_displaced_dwellings"])
    cost_million = float(item["cost"]["total_jpy"]) / 1e6
    cumulative_cost_million = float(item["cumulative_cost_jpy"]) / 1e6
    dwellings = int(item["district_dwellings_after"])
    cumulative_rights = int(item["cumulative_rights_converted"])
    connected_open = float(item["connected_open_space_m2"])
    cumulative_open = sum(float(p["connected_open_space_m2"]) for p in completed)
    road_area = float(item["road_dedication_m2"])
    cumulative_road = sum(float(p["road_dedication_m2"]) for p in completed)
    wind_before = float(item["wind_objective_before"])
    wind_after = float(item["wind_objective_after"])

next_project = phases[phase_index]["project_id"] if phase_index < len(phases) else "complete"
if phase_index == 0:
    hud = (
        "PHASE 0 / BASELINE | YEAR %d\n"
        "ORANGE = NEXT 4 HOMES FOR JOINT RENEWAL\n"
        "DWELLINGS %d | WIND PENALTY %.3f\n"
        "PLAY FRAMES 1-%d TO FOLLOW THE DELIVERY SEQUENCE"
        % (current_year, dwellings, wind_before, len(phases) + 1)
    )
else:
    hud = (
        "PHASE %d/%d | YEAR %d | 4 LOTS -> 1 JOINT BUILDING\n"
        "TEMP %d | RETURN %d | DISPLACED %d | DISTRICT DWELLINGS %d\n"
        "SCENARIO COST %.1f M JPY | CUMULATIVE %.1f M JPY\n"
        "WIND %.3f -> %.3f | OPEN %.1f m2 | ROAD %.1f m2"
        % (phase_index, len(phases), current_year, temporary, returned, displaced,
           dwellings, cost_million, cumulative_cost_million, wind_before, wind_after,
           connected_open, road_area)
    )
hud += (
    "\nCACHED FNO WIND U/U0 MIN %.2f | MEAN %.2f | MAX %.2f"
    % (wind_min, wind_mean, wind_max)
)

set_detail("timeline_phase", int(phase_index), 0)
set_detail("phase_count", int(len(phases)), 0)
set_detail("current_year", current_year, 0)
set_detail("stage_name", stage_name, "")
set_detail("project_id", project_id, "")
set_detail("next_project_id", next_project, "")
set_detail("rights_this_phase", rights_this, 0)
set_detail("cumulative_rights", cumulative_rights, 0)
set_detail("temporary_dwellings", temporary, 0)
set_detail("return_dwellings", returned, 0)
set_detail("permanently_displaced", displaced, 0)
set_detail("district_dwellings", dwellings, 0)
set_detail("phase_cost_million_jpy", cost_million, 0.0)
set_detail("cumulative_cost_million_jpy", cumulative_cost_million, 0.0)
set_detail("wind_objective_before", wind_before, 0.0)
set_detail("wind_objective_after", wind_after, 0.0)
set_detail("wind_display_engine", str(wind_visualization["engine"]), "")
set_detail("wind_display_quantity", str(wind_visualization["quantity"]), "")
set_detail("wind_ratio_min", wind_min, 0.0)
set_detail("wind_ratio_mean", wind_mean, 0.0)
set_detail("wind_ratio_max", wind_max, 0.0)
set_detail("connected_open_space_m2", connected_open, 0.0)
set_detail("cumulative_open_space_m2", cumulative_open, 0.0)
set_detail("road_dedication_m2", road_area, 0.0)
set_detail("cumulative_road_dedication_m2", cumulative_road, 0.0)
set_detail("hud_text", hud, "")
set_detail("result_json", result_path, "")
"""


WIND_TOGGLE_CODE = r"""
import hou

node = hou.pwd()
geo = node.geometry()
show_wind = bool(node.evalParm("show_wind"))
hidden_kinds = {"ground", "road"} if show_wind else {"wind_cell", "road_outline"}
prims = [
    prim
    for prim in geo.prims()
    if prim.stringAttribValue("kind") in hidden_kinds
]
if prims:
    geo.deletePrims(prims, True)
if geo.findGlobalAttrib("wind_display_enabled") is None:
    geo.addAttrib(hou.attribType.Global, "wind_display_enabled", 0)
geo.setGlobalAttribValue("wind_display_enabled", int(show_wind))
"""


def _load_payload(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if "initial_district" not in payload:
        raise ValueError(
            "feasibility JSON needs geometry; rerun evaluate_joint_feasibility.py "
            "with --include-districts"
        )
    phases = payload.get("phases", [])
    if not phases:
        raise ValueError("feasibility JSON contains no completed joint projects")
    if any("district" not in phase for phase in phases):
        raise ValueError("every feasibility phase must include district geometry")
    if any("open_spaces" not in phase for phase in phases):
        raise ValueError("every feasibility phase must include open-space geometry")
    wind = payload.get("wind_visualization")
    if not isinstance(wind, dict):
        raise ValueError("feasibility JSON needs cached wind_visualization data")
    frames = wind.get("frames", [])
    if len(frames) != len(phases) + 1:
        raise ValueError("wind_visualization needs baseline plus one frame per phase")
    if int(wind.get("display_resolution", 0)) <= 0:
        raise ValueError("wind_visualization display_resolution must be positive")
    return payload


def _parameters(sop, result_path: str, phase_count: int) -> None:
    group = sop.parmTemplateGroup()
    group.append(
        hou.StringParmTemplate(
            "result_json",
            "Feasibility result JSON",
            1,
            default_value=("$HIP/../outputs/joint_feasibility.json",),
            string_type=hou.stringParmType.FileReference,
        )
    )
    group.append(
        hou.ToggleParmTemplate(
            "use_timeline",
            "Drive phase from timeline",
            default_value=True,
        )
    )
    group.append(
        hou.IntParmTemplate(
            "manual_phase",
            "Manual phase",
            1,
            default_value=(0,),
            min=0,
            max=phase_count,
        )
    )
    group.append(
        hou.IntParmTemplate(
            "active_phase",
            "Active phase",
            1,
            default_value=(0,),
            min=0,
            max=phase_count,
        )
    )
    group.append(hou.ToggleParmTemplate("show_next", "Highlight next project", True))
    group.append(hou.ToggleParmTemplate("show_roads", "Show roads", True))
    group.append(hou.ToggleParmTemplate("show_parcels", "Show parcel outlines", True))
    group.append(hou.ToggleParmTemplate("show_open_space", "Show delivered open space", True))
    group.append(
        hou.FloatParmTemplate(
            "height_scale",
            "Building height scale",
            1,
            default_value=(1.0,),
            min=0.1,
            max=3.0,
        )
    )
    sop.setParmTemplateGroup(group)
    sop.parm("result_json").set(result_path)
    sop.parm("active_phase").setExpression(
        f"if(ch('use_timeline'), clamp($F - 1, 0, {phase_count}), ch('manual_phase'))",
        language=hou.exprLanguage.Hscript,
    )


def _font(geo, name: str, text: str, y: float, size: float, dynamic=False):
    font = geo.createNode("font", name)
    if dynamic:
        font.parm("text").setExpression(text, language=hou.exprLanguage.Python)
    else:
        font.parm("text").set(text)
    font.parm("tx").set(0.0)
    font.parm("ty").set(y)
    font.parm("tz").set(0.18)
    font.parm("fontsize").set(size)
    colour = geo.createNode("color", name + "_COLOUR")
    colour.setFirstInput(font)
    colour.parm("colorr").set(0.94)
    colour.parm("colorg").set(0.96)
    colour.parm("colorb").set(1.0)
    return colour


def build(out_path: str, result_path: str) -> None:
    result = Path(result_path).resolve()
    out = Path(out_path).resolve()
    payload = _load_payload(result)
    phase_count = len(payload["phases"])
    frame_count = phase_count + 1
    hou.setFps(1.0)
    hou.playbar.setFrameRange(1, frame_count)
    hou.playbar.setPlaybackRange(1, frame_count)
    hou.setFrame(1)

    obj = hou.node("/obj")
    geo = obj.createNode("geo", "mokumitsu_joint_feasibility", run_init_scripts=False)
    sop = geo.createNode("python", "joint_delivery_timeline")
    try:
        relative_result = Path(os.path.relpath(result, out.parent)).as_posix()
        hip_result_path = "$HIP/" + relative_result
    except ValueError:
        hip_result_path = result.as_posix()
    _parameters(sop, hip_result_path, phase_count)
    sop.parm("python").set(SOP_CODE)
    normal = geo.createNode("normal", "timeline_normals")
    normal.setFirstInput(sop)
    normal.setDisplayFlag(False)
    normal.setRenderFlag(False)

    file_cache = geo.createNode("filecache::2.0", "CACHE_JOINT_TIMELINE")
    file_cache.setFirstInput(normal)
    file_cache.parm("filemethod").set("explicit")
    file_cache.parm("file").set("$HIP/cache/joint_feasibility_timeline.$F4.bgeo.sc")
    file_cache.parm("timedependent").set(1)
    file_cache.parm("trange").set("normal")
    file_cache.parm("f1").set(1)
    file_cache.parm("f2").set(frame_count)
    file_cache.parm("f3").set(1)
    file_cache.parm("cachesim").set(0)
    file_cache.parm("loadfromdiskonsave").set(1)
    file_cache.parm("missingframe").set("error")
    file_cache.parm("cachesize").set(frame_count)
    file_cache.parm("loadfromdisk").set(0)
    file_cache.setComment(
        "Persistent 4-stage joint-renewal cache.\n"
        "$HIP/cache/joint_feasibility_timeline.$F4.bgeo.sc\n"
        "Each frame includes plan geometry and the FNO wind field.\n"
        "Load from Disk OFF -> change JSON/display controls -> Save to Disk."
    )
    file_cache.setGenericFlag(hou.nodeFlag.DisplayComment, True)

    wind_toggle = geo.createNode("python", "WIND_DISPLAY_TOGGLE")
    wind_toggle.setFirstInput(file_cache)
    wind_group = wind_toggle.parmTemplateGroup()
    wind_group.append(
        hou.ToggleParmTemplate(
            "show_wind",
            "Show cached FNO wind field",
            default_value=True,
        )
    )
    wind_toggle.setParmTemplateGroup(wind_group)
    wind_toggle.parm("python").set(WIND_TOGGLE_CODE)
    wind_toggle.setComment(
        "Toggle the cached wind-ratio heatmap. This only filters cached primitives; "
        "it does not rerun the FNO or the optimizer."
    )
    wind_toggle.setGenericFlag(hou.nodeFlag.DisplayComment, True)

    status = _font(
        geo,
        "PHASE_STATUS",
        'hou.node("../CACHE_JOINT_TIMELINE").geometry().attribValue("hud_text")',
        -20.5,
        2.15,
        dynamic=True,
    )
    legend = _font(
        geo,
        "COLOUR_LEGEND",
        "NEXT       NEW JOINT       COMPLETED       WIND CORRIDOR       PARK       ROAD LAND",
        -10.5,
        1.55,
    )
    wind_legend = _font(
        geo,
        "WIND_LEGEND",
        "FNO U/U0       WEAK       BELOW REF       ABOVE REF       STRONG"
        "    |    SELECT WIND_DISPLAY_TOGGLE TO SHOW/HIDE",
        -17.0,
        1.35,
    )
    merge = geo.createNode("merge", "DISPLAY_WITH_HUD")
    merge.setInput(0, wind_toggle)
    merge.setInput(1, status)
    merge.setInput(2, legend)
    merge.setInput(3, wind_legend)
    output = geo.createNode("null", "OUT_JOINT_FEASIBILITY")
    output.setFirstInput(merge)
    output.setDisplayFlag(True)
    output.setRenderFlag(True)

    sop.setComment(
        f"PLAY frames 1-{frame_count}: baseline plus {phase_count} completed joint projects.\n"
        "Orange buildings are the next project; magenta is newly completed.\n"
        "Cyan = ventilation corridor, green = park/court, amber = road dedication.\n"
        "Wind cells are the weighted FNO U/U0 screening field.\n"
        "The SOP reads precomputed JSON. Playback does not rerun FNO, fire, or optimization."
    )
    sop.setGenericFlag(hou.nodeFlag.DisplayComment, True)
    note = geo.createStickyNote()
    note.setText(
        "JOINT RENEWAL DELIVERY TIMELINE\n"
        f"PLAY FRAMES 1-{frame_count}: BASELINE -> {phase_count} PROJECTS\n"
        "ORANGE next / MAGENTA new / PURPLE completed\n"
        "CYAN corridor / GREEN park / AMBER road dedication\n"
        "WIND_DISPLAY_TOGGLE: cached FNO U/U0 heatmap ON/OFF\n"
        "CACHE_JOINT_TIMELINE stores one .bgeo.sc per frame."
    )
    note.setSize(hou.Vector2(6.8, 2.7))
    geo.layoutChildren()

    target = obj.createNode("null", "LOOKAT_JOINT_SITE")
    target.parmTuple("t").set((50.0, 38.0, 0.0))
    target.setDisplayFlag(False)
    camera = obj.createNode("cam", "CAM_JOINT_TIMELINE")
    camera.parmTuple("t").set((145.0, -135.0, 145.0))
    camera.parm("lookatpath").set(target.path())
    camera.parm("focal").set(55.0)
    camera.setComment(
        "Recommended overview camera. Select CAM_JOINT_TIMELINE in the viewport camera menu."
    )
    camera.setGenericFlag(hou.nodeFlag.DisplayComment, True)
    obj.layoutChildren()

    out.parent.mkdir(parents=True, exist_ok=True)
    hou.hscript("set -g POSE =")
    hou.hipFile.save(str(out))
    file_cache.parm("execute").pressButton()
    file_cache.parm("loadfromdisk").set(1)

    for frame in range(1, frame_count + 1):
        hou.setFrame(frame)
        output.cook(force=True)
        detail = output.geometry()
        print(
            f"frame={frame} phase={detail.attribValue('timeline_phase')} "
            f"year={detail.attribValue('current_year')} "
            f"dwellings={detail.attribValue('district_dwellings')} "
            f"temp={detail.attribValue('temporary_dwellings')} "
            f"cost={detail.attribValue('cumulative_cost_million_jpy'):.1f}M "
            f"open={detail.attribValue('cumulative_open_space_m2'):.1f}m2 "
            f"wind_U/U0={detail.attribValue('wind_ratio_mean'):.2f}"
        )
    hou.setFrame(1)
    hou.hipFile.save(str(out))
    print(
        f"saved {out} with {frame_count} cached frames; "
        "select CAM_JOINT_TIMELINE, use WIND_DISPLAY_TOGGLE, and play the timeline."
    )


if __name__ == "__main__":
    output = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else PROJECT_ROOT / "examples" / "houdini_joint_feasibility.hip"
    )
    result = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_RESULT)
    build(str(output), str(result))
