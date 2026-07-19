"""Build a Houdini scene for 木密 renewal, fire screening, and summer pedestrian wind.

The Python SOP regenerates the shared district from ``seed`` and recolours it by construction age,
access/rebuildability, renewal priority, fire risk, or FNO summer wind. Fire remains a fast graph
screen and wind is a surrogate screen; shortlisted schemes still require XLB verification.

    hython houdini/build_mokumitsu_hip.py
"""

# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import hou

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOP_CODE = r"""
import sys, math, os, hou, numpy as np
from pathlib import Path

hip_dir = Path(hou.getenv("HIP") or ".").resolve()
project_root = hip_dir.parent
sys.path.insert(0, str(project_root / "src"))
from mokumitsu import generate_mokumitsu, morphology_summary
from mokumitsu.fire import FireScenario, simulate_fire_spread, fire_aware_renewal_priorities
from mokumitsu.renewal import RenewalPolicy, simulate_renewal_trajectory
from mokumitsu.wind import (SummerWindScenario, WindDirection,
                            evaluate_district_wind, predict_directional_wind)
from mokumitsu.model import load_model

node = hou.pwd(); geo = node.geometry(); geo.clear()
seed = int(node.evalParm("seed")); mode = node.parm("display").evalAsString()
phase_index = int(node.evalParm("active_phase"))
scenario = FireScenario(wind_direction_deg=float(node.evalParm("wind_dir")),
                        wind_speed_m_s=float(node.evalParm("wind_speed")),
                        horizon_min=float(node.evalParm("horizon")),
                        runs=int(node.evalParm("runs")), seed=seed)
initial = generate_mokumitsu(seed); phase_actions = {}
if phase_index > 0:
    policy = RenewalPolicy(start_year=initial.reference_year, phases=phase_index,
                           years_per_phase=int(node.evalParm("years_per_phase")),
                           rebuild_fraction_per_phase=float(node.evalParm("rebuild_fraction")))
    trajectory = simulate_renewal_trajectory(initial, policy, scenario)
    selected = trajectory.phases[-1]; district = selected.district; fire = selected.fire
    phase_actions = {a.building_id: a for p in trajectory.phases for a in p.actions}
    renewed_total = selected.metrics.renewed_total
    deferred_total = selected.metrics.deferred_joint_total
else:
    district = initial; fire = simulate_fire_spread(district, scenario)
    renewed_total = deferred_total = 0
ranked = fire_aware_renewal_priorities(district, fire)
renewal = {x.building_id: x for x in ranked}
exposure = dict(zip(fire.building_ids, fire.burn_probability))
consequence = dict(zip(fire.building_ids, fire.source_consequence))

# Load the constant TorchScript resource once per Houdini session.  Geometry and wind still recook
# from the node parameters; only the model and empty-domain reference speed are cached.
wind_direction = float(node.parm("summer_wind_dir").evalAsString())
S = hou.session
if not hasattr(S, "_mokumitsu_district_model"):
    checkpoint_dir = os.environ.get(
        "MOKUMITSU_CHECKPOINT_DIR", str(project_root / "checkpoints")
    )
    S._mokumitsu_district_model = load_model("residential", ckpt_dir=checkpoint_dir)
    S._mokumitsu_district_u0 = S._mokumitsu_district_model.reference_speed()
wind_model = S._mokumitsu_district_model
wind_u0 = float(S._mokumitsu_district_u0)
wind_scenario = SummerWindScenario(directions=(WindDirection(wind_direction),))
wind_result = evaluate_district_wind(district, wind_scenario, wind_model, wind_u0)
wind_zones = wind_result.weighted_zones
wind_speed_world = wind_heightmap_world = None
if mode in ("wind_speed", "wind_class"):
    speed_eval, heightmap_eval, _, _ = predict_directional_wind(
        district, wind_direction, wind_model, reference_speed=wind_u0)
    k = int(round(wind_direction / 90.0)) % 4
    wind_speed_world = np.rot90(speed_eval, k=-k).copy()
    wind_heightmap_world = np.rot90(heightmap_eval, k=-k).copy()

geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
geo.addAttrib(hou.attribType.Prim, "kind", "")
geo.addAttrib(hou.attribType.Prim, "building_id", "")
geo.addAttrib(hou.attribType.Prim, "parcel_id", "")
geo.addAttrib(hou.attribType.Prim, "action", "")
geo.addAttrib(hou.attribType.Prim, "renewal_status", "existing")
geo.addAttrib(hou.attribType.Prim, "built_year", 0)
geo.addAttrib(hou.attribType.Prim, "priority", 0.0)
geo.addAttrib(hou.attribType.Prim, "fire_exposure", 0.0)
geo.addAttrib(hou.attribType.Prim, "fire_consequence", 0.0)
geo.addAttrib(hou.attribType.Prim, "feasibility", 0.0)
geo.addAttrib(hou.attribType.Prim, "wind_index", 0.0)
geo.addAttrib(hou.attribType.Prim, "wind_class", "")

def pt(x, y, z, colour):
    p = geo.createPoint(); p.setPosition(hou.Vector3(x, y, z)); p.setAttribValue("Cd", colour); return p

def poly(coords, colour, kind, closed=True, attrs=None):
    prim = geo.createPolygon(); prim.setIsClosed(closed)
    for x, y, z in coords: prim.addVertex(pt(x, y, z, colour))
    prim.setAttribValue("kind", kind)
    if attrs:
        for name, value in attrs.items(): prim.setAttribValue(name, value)
    return prim

def road_poly(road, z=0.0):
    x0, y0 = road.start; x1, y1 = road.end
    length = max(road.length_m, 1e-9); nx = -(y1-y0)/length; ny = (x1-x0)/length
    hx, hy = 0.5*road.width_m*nx, 0.5*road.width_m*ny
    return [(x0+hx,y0+hy,z),(x1+hx,y1+hy,z),(x1-hx,y1-hy,z),(x0-hx,y0-hy,z)]

def heat(value):
    v = max(0.0, min(1.0, float(value)))
    if v < 0.5: return (0.12 + 1.55*v, 0.25 + 1.20*v, 0.85 - 1.10*v)
    q = 2.0*(v-0.5); return (0.90, 0.82 - 0.68*q, 0.28 - 0.20*q)

def wind_colour(index):
    if mode == "wind_class":
        if index < 0.30: return (0.08,0.32,0.88)
        if index > 1.30: return (0.92,0.10,0.12)
        return (0.16,0.68,0.34)
    return heat(index / 1.5)

def building_colour(building, access, item):
    if mode in ("wind_speed", "wind_class"): return (0.46,0.48,0.50)
    if mode == "age":
        return {"pre_1981": (0.78,0.08,0.10), "1981_2000": (0.94,0.42,0.20),
                "post_2000": (0.22,0.55,0.78)}[building.age_cohort]
    if mode == "access":
        if access.individual_rebuildable: return (0.20,0.72,0.30)
        if access.path_frontage_m >= 2.0: return (0.64,0.28,0.72)
        return (0.88,0.10,0.18)
    if mode == "renewal_status":
        action = phase_actions.get(building.id)
        if action and action.status == "completed": return (0.12,0.72,0.42)
        if action and action.status == "deferred": return (0.94,0.48,0.16)
        return (0.32,0.35,0.38)
    if mode == "fire_exposure": return heat(exposure[building.id])
    if mode == "fire_consequence": return heat(consequence[building.id])
    return heat(item.combined_score)

def box(building, colour, attrs):
    corners = building.corners; z0,z1 = 0.05, building.height_m + 0.05
    p = ([pt(x,y,z0,colour) for x,y in corners] +
         [pt(x,y,z1,colour) for x,y in corners])
    for face in ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)):
        prim = geo.createPolygon(); prim.setIsClosed(True)
        for i in face: prim.addVertex(p[i])
        prim.setAttribValue("kind", "building")
        for name, value in attrs.items(): prim.setAttribValue(name, value)

# In wind modes the ground becomes a downsampled U/U0 field in world orientation.  Filled road
# polygons would hide the most important pedestrian cells, so those modes draw only road outlines.
if wind_speed_world is None:
    poly([(0,0,-0.08),(district.width_m,0,-0.08),(district.width_m,district.height_m,-0.08),
          (0,district.height_m,-0.08)], (0.09,0.11,0.10), "ground")
else:
    res = wind_speed_world.shape[0]; stride = max(1, res // 48)
    for iy in range(0, res, stride):
        for ix in range(0, res, stride):
            y1i, x1i = min(iy+stride, res), min(ix+stride, res)
            open_patch = wind_heightmap_world[iy:y1i, ix:x1i] <= 1e-7
            if np.any(open_patch):
                index = float(np.mean(wind_speed_world[iy:y1i, ix:x1i][open_patch]) / wind_u0)
                klass = "dead" if index < 0.30 else "strong" if index > 1.30 else "comfortable"
                colour = wind_colour(index)
            else:
                index, klass, colour = 0.0, "solid", (0.10,0.11,0.12)
            x0 = ix/res*district.width_m; x1 = x1i/res*district.width_m
            y0 = iy/res*district.height_m; y1 = y1i/res*district.height_m
            poly([(x0,y0,-0.015),(x1,y0,-0.015),(x1,y1,-0.015),(x0,y1,-0.015)],
                 colour, "wind_cell", attrs={"wind_index": index, "wind_class": klass})
for road in district.roads:
    colour = ((0.32,0.34,0.36) if road.legal_class == "article_42_1" else
              (0.72,0.55,0.16) if road.legal_class == "article_42_2" else (0.34,0.25,0.42))
    coords = road_poly(road, z=0.025)
    if wind_speed_world is None:
        poly(coords, colour, "road")
    else:
        poly(coords + [coords[0]], (0.72,0.74,0.76), "road_outline", closed=False)
for parcel in district.parcels:
    poly([(x,y,0.035) for x,y in parcel.polygon] + [(parcel.polygon[0][0],parcel.polygon[0][1],0.035)],
         (0.38,0.42,0.44), "parcel", closed=False)
for building in district.buildings:
    access = district.access(building.parcel_id); item = renewal[building.id]
    attrs = {"building_id": building.id, "parcel_id": building.parcel_id,
             "action": item.recommended_action, "built_year": int(building.built_year),
             "renewal_status": (phase_actions[building.id].status
                                 if building.id in phase_actions else "existing"),
             "priority": float(item.combined_score), "fire_exposure": float(exposure[building.id]),
             "fire_consequence": float(consequence[building.id]),
             "feasibility": float(item.feasibility_score)}
    box(building, building_colour(building, access, item), attrs)

def setd(name, value, default):
    if geo.findGlobalAttrib(name) is None: geo.addAttrib(hou.attribType.Global, name, default)
    geo.setGlobalAttribValue(name, value)
summary = district.summary()
morphology = morphology_summary(district)
setd("building_count", int(summary.buildings), 0)
setd("legal_access_rate", float(summary.legal_access_rate), 0.0)
setd("rebuildable_rate", float(summary.individual_rebuildable_rate), 0.0)
setd("pre_1981_rate", float(summary.pre_1981_rate), 0.0)
setd("gross_coverage", float(summary.footprint_coverage), 0.0)
setd("net_coverage", float(summary.net_building_coverage), 0.0)
setd("gross_far", float(summary.gross_floor_area_ratio), 0.0)
setd("renewal_phase", int(phase_index), 0)
setd("current_year", int(district.reference_year), 0)
setd("renewed_total", int(renewed_total), 0)
setd("deferred_joint_total", int(deferred_total), 0)
setd("expected_burned", float(fire.expected_burned_buildings), 0.0)
setd("p95_burned", float(fire.p95_burned_buildings), 0.0)
setd("large_fire_probability", float(fire.probability_large_fire), 0.0)
setd("summer_wind_direction", float(wind_direction), 0.0)
setd("wind_reference_speed", float(wind_result.reference_speed), 0.0)
setd("wind_mean_index", float(wind_zones["all_outdoor"].mean_index), 0.0)
setd("wind_dead_fraction", float(wind_zones["all_outdoor"].dead_fraction), 0.0)
setd("wind_strong_fraction", float(wind_zones["all_outdoor"].strong_fraction), 0.0)
setd("road_dead_fraction", float(wind_zones["roads"].dead_fraction), 0.0)
setd("yard_dead_fraction", float(wind_zones["parcel_open_space"].dead_fraction), 0.0)
setd("building_edge_strong_fraction", float(wind_zones["building_edge"].strong_fraction), 0.0)
setd("non_axis_road_rate", float(morphology.non_axis_aligned_road_rate), 0.0)
setd("orientation_entropy", float(morphology.road_orientation_entropy), 0.0)
setd("dead_end_road_rate", float(morphology.dead_end_road_rate), 0.0)
setd("irregular_parcel_rate", float(morphology.irregular_parcel_rate), 0.0)
setd("flag_lot_rate", float(morphology.flag_lot_rate), 0.0)
setd("back_lot_rate", float(morphology.back_lot_rate), 0.0)
setd("top_priority_building", ranked[0].building_id, "")
"""


def _parameters(sop):
    group = sop.parmTemplateGroup()
    group.append(hou.IntParmTemplate("seed", "District seed", 1, default_value=(0,)))
    group.append(
        hou.MenuParmTemplate(
            "display",
            "Display",
            (
                "age",
                "access",
                "renewal_status",
                "priority",
                "fire_exposure",
                "fire_consequence",
                "wind_speed",
                "wind_class",
            ),
            (
                "Construction age",
                "Access / rebuildability",
                "Phased renewal status",
                "Combined priority",
                "Fire exposure",
                "Ignition consequence",
                "Summer wind speed U/U0",
                "Summer weak / comfortable / strong",
            ),
            default_value=2,
        )
    )
    group.append(
        hou.ToggleParmTemplate(
            "use_timeline",
            "Drive renewal from timeline",
            default_value=True,
        )
    )
    group.append(
        hou.IntParmTemplate(
            "renewal_phase",
            "Manual renewal phase",
            1,
            default_value=(0,),
            min=0,
            max=6,
        )
    )
    group.append(
        hou.IntParmTemplate(
            "active_phase",
            "Active renewal phase",
            1,
            default_value=(0,),
            min=0,
            max=6,
        )
    )
    group.append(
        hou.IntParmTemplate(
            "years_per_phase", "Years per phase", 1, default_value=(5,), min=1, max=20
        )
    )
    group.append(
        hou.FloatParmTemplate(
            "rebuild_fraction",
            "Rebuild fraction / phase",
            1,
            default_value=(0.08,),
            min=0.01,
            max=0.30,
        )
    )
    group.append(
        hou.MenuParmTemplate(
            "summer_wind_dir",
            "Summer wind direction (toward)",
            ("0", "90", "180", "270"),
            ("East / +X", "North / +Y", "West / -X", "South / -Y"),
            default_value=0,
        )
    )
    group.append(
        hou.FloatParmTemplate(
            "wind_dir",
            "Fire wind direction (deg, toward)",
            1,
            default_value=(0.0,),
            min=0.0,
            max=360.0,
        )
    )
    group.append(
        hou.FloatParmTemplate(
            "wind_speed",
            "Fire wind speed (m/s)",
            1,
            default_value=(5.0,),
            min=0.0,
            max=20.0,
        )
    )
    group.append(
        hou.FloatParmTemplate(
            "horizon",
            "Fire horizon (min)",
            1,
            default_value=(180.0,),
            min=30.0,
            max=480.0,
        )
    )
    group.append(
        hou.IntParmTemplate(
            "runs",
            "Monte Carlo runs",
            1,
            default_value=(300,),
            min=10,
            max=5000,
        )
    )
    sop.setParmTemplateGroup(group)
    sop.parm("active_phase").setExpression(
        "if(ch('use_timeline'), clamp($F - 1, 0, 6), ch('renewal_phase'))",
        language=hou.exprLanguage.Hscript,
    )


def build(out_path: str):
    hou.setFps(1.0)
    hou.playbar.setFrameRange(1, 7)
    hou.playbar.setPlaybackRange(1, 7)
    hou.setFrame(1)
    geo = hou.node("/obj").createNode("geo", "mokumitsu_resilience", run_init_scripts=False)
    sop = geo.createNode("python", "district_fire_screen")
    _parameters(sop)
    sop.parm("python").set(SOP_CODE)
    normal = geo.createNode("normal", "building_normals")
    normal.setFirstInput(sop)
    normal.setDisplayFlag(False)
    normal.setRenderFlag(False)

    file_cache = geo.createNode("filecache::2.0", "CACHE_TIMELINE")
    file_cache.setFirstInput(normal)
    file_cache.parm("filemethod").set("explicit")
    file_cache.parm("file").set("$HIP/cache/mokumitsu_timeline.$F4.bgeo.sc")
    file_cache.parm("timedependent").set(1)
    file_cache.parm("trange").set("normal")
    file_cache.parm("f1").set(1)
    file_cache.parm("f2").set(7)
    file_cache.parm("f3").set(1)
    file_cache.parm("cachesim").set(0)
    file_cache.parm("loadfromdiskonsave").set(1)
    file_cache.parm("missingframe").set("error")
    file_cache.parm("cachesize").set(7)
    file_cache.parm("loadfromdisk").set(0)
    file_cache.setComment(
        "Persistent timeline cache: $HIP/cache/mokumitsu_timeline.$F4.bgeo.sc\n"
        "For a new scenario, turn Load from Disk off, edit parameters, then press Save to Disk."
    )
    file_cache.setGenericFlag(hou.nodeFlag.DisplayComment, True)

    output = geo.createNode("null", "OUT_MOKUMITSU")
    output.setFirstInput(file_cache)
    output.setDisplayFlag(True)
    output.setRenderFlag(True)
    sop.setComment(
        "PLAY frames 1-7: renewal phases 0-6 (2026-2056 at the default 5-year interval).\n"
        "CACHE_TIMELINE is a standard File Cache SOP storing one .bgeo.sc file per frame.\n"
        "Turn Load from Disk off before editing; Save to Disk rebuilds frames 1-7 and enables it again.\n"
        "Regenerate a dense non-grid wooden district and compare AGE / ACCESS / RENEWAL / FIRE / WIND.\n"
        "Fire and FNO are planning screens; shortlisted schemes still require XLB validation."
    )
    sop.setGenericFlag(hou.nodeFlag.DisplayComment, True)
    note = geo.createStickyNote()
    note.setText(
        "MOKUMITSU RENEWAL TIMELINE\n"
        "PLAY frames 1-7: 2026 -> 2056\n"
        "CACHE_TIMELINE stores frames 1-7 in $HIP/cache as .bgeo.sc files.\n"
        "EDIT: Load from Disk OFF -> change parameters -> Save to Disk."
    )
    note.setSize(hou.Vector2(6.0, 2.3))
    hou.setFrame(1)
    geo.layoutChildren()

    # Save once so $HIP resolves beside the generated scene, then use Houdini's
    # own File Cache SOP to build the complete timeline cache.
    hou.hscript("set -g POSE =")
    hou.hipFile.save(out_path)
    file_cache.parm("execute").pressButton()
    file_cache.parm("loadfromdisk").set(1)
    output.cook(force=True)
    detail = output.geometry()
    print(
        f"mokumitsu: buildings={detail.attribValue('building_count')} "
        f"phase={detail.attribValue('renewal_phase')} "
        f"gross_coverage={detail.attribValue('gross_coverage'):.3f} "
        f"access={detail.attribValue('legal_access_rate'):.3f} "
        f"expected_burned={detail.attribValue('expected_burned'):.2f} "
        f"p95={detail.attribValue('p95_burned'):.0f} "
        f"weak={detail.attribValue('wind_dead_fraction'):.3f} "
        f"strong={detail.attribValue('wind_strong_fraction'):.3f} "
        f"top={detail.attribValue('top_priority_building')}"
    )
    hou.hipFile.save(out_path)
    print(
        f"saved {out_path} with a 7-frame Houdini File Cache; "
        "use CACHE_TIMELINE to reload or rebuild the cache."
    )


if __name__ == "__main__":
    output = Path(
        sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "examples" / "houdini_mokumitsu.hip"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    build(str(output))
