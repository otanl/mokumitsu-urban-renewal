"""Headless cook and cache check for the live joint-design HIP sample."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import hou


def _detail(geometry, name, default=None):
    return geometry.attribValue(name) if geometry.findGlobalAttrib(name) is not None else default


def _worker_state():
    registry = getattr(hou.session, "_mokumitsu_joint_design_workers", {})
    return [
        {"pid": worker.process.pid, "poll": worker.process.poll()} for worker in registry.values()
    ]


def _cook(node, output, label):
    start = time.perf_counter()
    output.cook(force=True)
    wall_ms = 1000.0 * (time.perf_counter() - start)
    geometry = node.geometry()
    print(
        f"{label}: status={_detail(geometry, 'status')} "
        f"cache={_detail(geometry, 'cache_hit', 0)} "
        f"reported={_detail(geometry, 'evaluation_ms', 0.0):.1f}ms wall={wall_ms:.1f}ms "
        f"workers={_worker_state()}"
    )
    if _detail(geometry, "status") != "screened":
        raise RuntimeError(_detail(geometry, "hud_text", "missing HUD"))
    if _detail(geometry, "worker_error", ""):
        raise RuntimeError(_detail(geometry, "worker_error"))
    return (
        wall_ms,
        int(_detail(geometry, "cache_hit", 0)),
        str(_detail(geometry, "cache_key", "")),
    )


def _edit(node, output, parm_name, value, label):
    start = time.perf_counter()
    node.parm(parm_name).set(value)
    output.cook(force=False)
    wall_ms = 1000.0 * (time.perf_counter() - start)
    geometry = node.geometry()
    print(
        f"{label}: status={_detail(geometry, 'status')} "
        f"cache={_detail(geometry, 'cache_hit', 0)} "
        f"worker={_detail(geometry, 'worker_ms', 0.0):.1f}ms wall={wall_ms:.1f}ms "
        f"worker_error={_detail(geometry, 'worker_error', '')} workers={_worker_state()}"
    )
    if _detail(geometry, "status") != "screened":
        raise RuntimeError(_detail(geometry, "hud_text", "missing HUD"))
    if _detail(geometry, "worker_error", ""):
        raise RuntimeError(_detail(geometry, "worker_error"))
    return (
        wall_ms,
        int(_detail(geometry, "cache_hit", 0)),
        str(_detail(geometry, "cache_key", "")),
    )


def verify(path: str) -> None:
    hou.hipFile.load(str(Path(path).resolve()), suppress_save_prompt=True)
    live = hou.node("/obj/mokumitsu_joint_design/LIVE_JOINT_DESIGN")
    output = hou.node("/obj/mokumitsu_joint_design/OUT_JOINT_DESIGN")
    if live is None or output is None:
        raise RuntimeError("live joint-design nodes are missing")

    original_rotation = float(live.evalParm("rotation_deg"))
    _, _, default_key = _cook(live, output, "saved-default")
    _, _, edited_key = _edit(
        live,
        output,
        "rotation_deg",
        original_rotation + 0.371,
        "new-design",
    )
    _, cache_hit, restored_key = _edit(
        live,
        output,
        "rotation_deg",
        original_rotation,
        "return-to-default",
    )
    if not default_key or edited_key == default_key:
        raise RuntimeError("a geometry edit did not produce a distinct cache key")
    if restored_key != default_key:
        raise RuntimeError("returning to the saved design did not restore its cache key")
    if not cache_hit:
        raise RuntimeError("returning to a saved design did not hit the keyed cache")
    if live.errors() or output.errors():
        raise RuntimeError(f"node errors: live={live.errors()} output={output.errors()}")


if __name__ == "__main__":
    hip = (
        sys.argv[1]
        if len(sys.argv) > 1
        else Path(__file__).parents[1] / "examples" / "houdini_joint_design.hip"
    )
    verify(str(hip))
