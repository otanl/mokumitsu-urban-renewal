"""Build the live parametric joint-renewal Houdini sample."""

from __future__ import annotations

import sys
from pathlib import Path

import hou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "houdini"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mokumitsu import (  # noqa: E402
    JointRenewalPolicy,
    default_joint_design_parameters,
    generate_mokumitsu,
    joint_renewal_candidates,
)

SOP_CODE = r"""
import sys
from pathlib import Path
import hou

project_root = Path(hou.getenv("HIP") or ".").resolve().parent
for path in (project_root / "src", project_root / "houdini"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from live_joint_design import cook
cook(hou.pwd())
"""

FILTER_CODE = r"""
import hou

node = hou.pwd()
geo = node.geometry()
geo.clear()
geo.merge(node.inputs()[0].geometry())
hidden = set()
if not node.evalParm("show_wind"):
    hidden.add("wind_cell")
if not node.evalParm("show_roads"):
    hidden.add("road")
if not node.evalParm("show_parcels"):
    hidden.update(("parcel_outline", "source_parcel_outline"))
if not node.evalParm("show_open_space"):
    hidden.add("open_space")
if not node.evalParm("show_buildings"):
    hidden.add("building")
delete = [primitive for primitive in geo.prims() if primitive.attribValue("kind") in hidden]
if delete:
    geo.deletePrims(delete, True)
"""

RESET_CALLBACK = r"""
import sys
from pathlib import Path
import hou

project_root = Path(hou.getenv("HIP") or ".").resolve().parent
for path in (project_root / "src", project_root / "houdini"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from live_joint_design import reset_parameters
reset_parameters(kwargs["node"])
"""

RESTART_WORKER_CALLBACK = r"""
import sys
from pathlib import Path
import hou

project_root = Path(hou.getenv("HIP") or ".").resolve().parent
for path in (project_root / "src", project_root / "houdini"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from live_joint_design import restart_worker
restart_worker()
kwargs["node"].cook(force=True)
"""


def _float(
    name: str,
    label: str,
    value: float,
    minimum: float,
    maximum: float,
    *,
    strict_min: bool = False,
    strict_max: bool = False,
):
    return hou.FloatParmTemplate(
        name,
        label,
        1,
        default_value=(value,),
        min=minimum,
        max=maximum,
        min_is_strict=strict_min,
        max_is_strict=strict_max,
    )


def _direction(name: str, label: str, value: int):
    return hou.IntParmTemplate(
        name,
        label,
        1,
        default_value=(value,),
        menu_items=("0", "90", "180", "270"),
        menu_labels=("+X / 0 deg", "+Y / 90 deg", "-X / 180 deg", "-Y / 270 deg"),
    )


def _default_setup():
    district = generate_mokumitsu(seed=0)
    policy = JointRenewalPolicy(
        candidate_limit=24,
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
        minimum_floors=3,
        maximum_floors=8,
        placement_grid=7,
        placement_variants=3,
    )
    candidates = joint_renewal_candidates(district, policy)
    parcel_count = 3 if any(candidate.parcel_count == 3 for candidate in candidates) else 2
    candidate = next(
        candidate for candidate in candidates if candidate.parcel_count == parcel_count
    )
    parameters = default_joint_design_parameters(district, candidate, policy)
    return parcel_count, parameters


def _live_parameters(node, parcel_count, parameters) -> None:
    group = hou.ParmTemplateGroup()
    site = hou.FolderParmTemplate(
        "site_controls",
        "Site and candidate",
        parm_templates=(
            hou.IntParmTemplate("seed", "Synthetic district seed", 1, default_value=(0,), min=0),
            hou.IntParmTemplate(
                "cluster_size",
                "Joint parcel count",
                1,
                default_value=(parcel_count,),
                menu_items=("2", "3", "4"),
                menu_labels=("2 parcels", "3 parcels", "4 parcels"),
            ),
            hou.IntParmTemplate(
                "candidate_rank",
                "Candidate priority rank",
                1,
                default_value=(0,),
                min=0,
                max=11,
                min_is_strict=True,
            ),
            hou.IntParmTemplate(
                "renewal_year", "Renewal year", 1, default_value=(2036,), min=2025, max=2100
            ),
        ),
    )
    massing = hou.FolderParmTemplate(
        "massing_controls",
        "Editable massing",
        parm_templates=(
            _float(
                "center_u",
                "Position U",
                parameters.center_u,
                0.0,
                1.0,
                strict_min=True,
                strict_max=True,
            ),
            _float(
                "center_v",
                "Position V",
                parameters.center_v,
                0.0,
                1.0,
                strict_min=True,
                strict_max=True,
            ),
            _float(
                "building_coverage",
                "Building coverage",
                parameters.building_coverage,
                0.08,
                0.60,
                strict_min=True,
                strict_max=True,
            ),
            _float(
                "aspect_ratio",
                "Width / depth",
                parameters.aspect_ratio,
                0.4,
                3.5,
                strict_min=True,
                strict_max=True,
            ),
            _float("rotation_deg", "Rotation (deg)", parameters.rotation_deg, 0.0, 180.0),
            hou.IntParmTemplate(
                "floors",
                "Floors",
                1,
                default_value=(parameters.floors,),
                min=3,
                max=8,
                min_is_strict=True,
                max_is_strict=True,
            ),
            hou.ButtonParmTemplate(
                "reset_feasible",
                "Reset to searched feasible design",
                script_callback=RESET_CALLBACK,
                script_callback_language=hou.scriptLanguage.Python,
            ),
        ),
    )
    land = hou.FolderParmTemplate(
        "land_controls",
        "Land allocation rules",
        parm_templates=(
            _float(
                "target_road_width",
                "Target road width (m)",
                6.0,
                4.0,
                10.0,
                strict_min=True,
                strict_max=True,
            ),
            _float(
                "corridor_width",
                "Ventilation corridor width (m)",
                2.0,
                1.0,
                5.0,
                strict_min=True,
                strict_max=True,
            ),
            _float(
                "open_space_fraction",
                "Shared open-space fraction",
                0.10,
                0.03,
                0.30,
                strict_min=True,
                strict_max=True,
            ),
            hou.IntParmTemplate(
                "minimum_floors", "Minimum floors", 1, default_value=(3,), min=1, max=8
            ),
            hou.IntParmTemplate(
                "maximum_floors", "Maximum floors", 1, default_value=(8,), min=3, max=12
            ),
        ),
    )
    environment = hou.FolderParmTemplate(
        "environment_controls",
        "Environmental screening",
        parm_templates=(
            _direction("wind_primary_deg", "Primary summer wind", 0),
            _direction("wind_secondary_deg", "Secondary summer wind", 90),
            _float(
                "wind_primary_weight",
                "Primary wind frequency",
                0.70,
                0.05,
                0.95,
                strict_min=True,
                strict_max=True,
            ),
            hou.StringParmTemplate(
                "wind_model", "Wind model family", 1, default_value=("residential",)
            ),
            hou.StringParmTemplate(
                "checkpoint_dir",
                "Checkpoint directory",
                1,
                default_value=("$MOKUMITSU_CHECKPOINT_DIR",),
                string_type=hou.stringParmType.FileReference,
            ),
            hou.ToggleParmTemplate(
                "use_worker",
                "Use persistent external worker",
                default_value=True,
            ),
            hou.StringParmTemplate(
                "worker_python",
                "Worker Python",
                1,
                default_value=("$MOKUMITSU_PYTHON",),
                string_type=hou.stringParmType.FileReference,
            ),
            hou.StringParmTemplate(
                "worker_device",
                "Worker device",
                1,
                default_value=("auto",),
                menu_items=("auto", "cuda", "cpu"),
                menu_labels=("Auto (CUDA if available)", "CUDA", "CPU"),
            ),
            hou.ButtonParmTemplate(
                "restart_worker",
                "Restart external worker",
                script_callback=RESTART_WORKER_CALLBACK,
                script_callback_language=hou.scriptLanguage.Python,
            ),
            hou.IntParmTemplate(
                "display_resolution",
                "Wind display resolution",
                1,
                default_value=(20,),
                menu_items=("16", "20", "24", "32"),
                menu_labels=("16 x 16", "20 x 20", "24 x 24", "32 x 32"),
            ),
            _direction("fire_wind_deg", "Fire wind direction", 0),
            _float(
                "fire_wind_speed",
                "Fire wind speed (m/s)",
                5.0,
                0.0,
                15.0,
                strict_min=True,
                strict_max=True,
            ),
            hou.IntParmTemplate(
                "fire_runs", "Fire Monte Carlo runs", 1, default_value=(40,), min=4, max=500
            ),
            hou.ToggleParmTemplate("disk_cache", "Automatic keyed disk cache", default_value=True),
        ),
    )
    for folder in (site, massing, land, environment):
        group.append(folder)
    node.setParmTemplateGroup(group)


def _view_parameters(node) -> None:
    group = hou.ParmTemplateGroup()
    for name, label, default in (
        ("show_wind", "Show FNO wind field", True),
        ("show_roads", "Show roads", True),
        ("show_parcels", "Show parcel outlines", True),
        ("show_open_space", "Show allocated open space", True),
        ("show_buildings", "Show buildings", True),
    ):
        group.append(hou.ToggleParmTemplate(name, label, default_value=default))
    node.setParmTemplateGroup(group)


def _font(container, name: str, expression: str, y: float, size: float):
    font = container.createNode("font", name)
    font.parm("text").setExpression(expression, language=hou.exprLanguage.Python)
    font.parm("tx").set(0.0)
    font.parm("ty").set(y)
    font.parm("tz").set(0.20)
    font.parm("fontsize").set(size)
    colour = container.createNode("color", name + "_COLOUR")
    colour.setFirstInput(font)
    colour.parmTuple("color").set((0.94, 0.96, 1.0))
    return colour


def build(out_path: str) -> None:
    out = Path(out_path).resolve()
    parcel_count, parameters = _default_setup()
    hou.setFps(24.0)
    hou.playbar.setFrameRange(1, 240)
    hou.playbar.setPlaybackRange(1, 240)
    hou.setFrame(1)

    obj = hou.node("/obj")
    container = obj.createNode("geo", "mokumitsu_joint_design", run_init_scripts=False)
    live = container.createNode("python", "LIVE_JOINT_DESIGN")
    _live_parameters(live, parcel_count, parameters)
    live.parm("python").set(SOP_CODE)
    live.setComment(
        "Edit massing and land-allocation parameters: FNO and fire screening "
        "recook automatically.\n"
        "Each unique design is cached by a SHA-256 content key under $HIP/cache/joint_design.\n"
        "Use Reset only to return to a searched feasible starting geometry; "
        "it is not an analysis button."
    )
    live.setGenericFlag(hou.nodeFlag.DisplayComment, True)

    normal = container.createNode("normal", "DESIGN_NORMALS")
    normal.setFirstInput(live)
    view = container.createNode("python", "VIEW_OPTIONS")
    view.setFirstInput(normal)
    _view_parameters(view)
    view.parm("python").set(FILTER_CODE)
    view.setComment("Display toggles only filter cached geometry; they do not rerun screening.")
    view.setGenericFlag(hou.nodeFlag.DisplayComment, True)

    hud = _font(
        container,
        "DESIGN_METRICS",
        'hou.node("../LIVE_JOINT_DESIGN").geometry().attribValue("hud_text")',
        -20.5,
        1.45,
    )
    cache = _font(
        container,
        "CACHE_STATUS",
        '(lambda g: ("CACHE HIT" if g.attribValue("cache_hit") else "CACHE MISS") + '
        '" | cook %.0f ms" % g.attribValue("evaluation_ms"))'
        '(hou.node("../LIVE_JOINT_DESIGN").geometry())',
        -12.0,
        1.15,
    )
    merge = container.createNode("merge", "DISPLAY_WITH_HUD")
    merge.setInput(0, view)
    merge.setInput(1, hud)
    merge.setInput(2, cache)
    output = container.createNode("null", "OUT_JOINT_DESIGN")
    output.setFirstInput(merge)
    output.setDisplayFlag(True)
    output.setRenderFlag(True)

    note = container.createStickyNote()
    note.setText(
        "LIVE JOINT-RENEWAL DESIGN\n"
        "1. Select LIVE_JOINT_DESIGN.\n"
        "2. Edit position / coverage / aspect / rotation / floors.\n"
        "3. The persistent project-Python worker keeps FNO + fire updates automatic.\n"
        "4. Infeasible geometry is reported instead of silently repaired.\n"
        "5. Use VIEW_OPTIONS for display-only toggles.\n"
        "6. XLB remains an explicit verification step for shortlisted designs."
    )
    note.setSize(hou.Vector2(7.2, 3.4))
    container.layoutChildren()

    target = obj.createNode("null", "LOOKAT_JOINT_DESIGN")
    target.parmTuple("t").set((50.0, 38.0, 0.0))
    target.setDisplayFlag(False)
    camera = obj.createNode("cam", "CAM_JOINT_DESIGN")
    camera.parmTuple("t").set((142.0, -132.0, 142.0))
    camera.parm("lookatpath").set(target.path())
    camera.parm("focal").set(55.0)
    camera.setComment("Recommended overview camera for the live design sample.")
    camera.setGenericFlag(hou.nodeFlag.DisplayComment, True)
    obj.layoutChildren()

    out.parent.mkdir(parents=True, exist_ok=True)
    hou.hscript("set -g POSE =")
    hou.hipFile.save(str(out))
    output.cook(force=True)
    detail = live.geometry()
    print(
        f"status={detail.attribValue('status')} cache={detail.attribValue('cache_hit')} "
        f"cook={detail.attribValue('evaluation_ms'):.1f}ms "
        f"size={detail.attribValue('building_width_m'):.1f}x"
        f"{detail.attribValue('building_depth_m'):.1f}m "
        f"floors={detail.attribValue('building_floors')}"
    )
    hou.hipFile.save(str(out))
    print(f"saved {out}; select LIVE_JOINT_DESIGN and edit parameters")


if __name__ == "__main__":
    output = Path(
        sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "examples" / "houdini_joint_design.hip"
    )
    build(str(output))
