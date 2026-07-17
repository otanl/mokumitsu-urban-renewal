# Houdini examples

These scenes are Houdini adapters for the Mokumitsu research prototype. They
were generated with Houdini Indie 20.5.684.

## houdini_joint_design.hip

This is the live parametric study. Select `LIVE_JOINT_DESIGN` to edit the
selected two-to-four-parcel project, massing, road dedication, ventilation
corridor, shared open space and scenarios. FNO wind, graph-fire and capacity
metrics recook automatically. Invalid geometry is reported rather than silently
repaired.

`VIEW_OPTIONS` changes display layers without re-running analysis. Evaluated
designs use content-addressed files under:

    cache/joint_design/

Those dynamic files are ignored by Git. A persistent project-Python worker keeps
the accelerated FNO and reusable baseline data warm; the first cook is expected
to be slower than subsequent edits.

## houdini_joint_feasibility.hip

Frames 1–4 show:

1. the 2026 baseline;
2. joint project 1 delivered in 2036;
3. joint project 2 delivered in 2039;
4. joint project 3 delivered in 2042.

Select CAM_JOINT_TIMELINE for the intended overview. Select
WIND_DISPLAY_TOGGLE and change Show cached FNO wind field to switch between the
plan surface and weighted scalar U/U0.

The scene reads:

    cache/joint_feasibility_timeline.$F4.bgeo.sc

## houdini_mokumitsu.hip

Frames 1–7 show the baseline and six individual-renewal phases. The scene reads:

    cache/mokumitsu_timeline.$F4.bgeo.sc

## What the cache means

The bgeo.sc files contain already-computed geometry and display attributes. With
Load from Disk enabled, timeline playback does not rerun the FNO, fire model,
renewal logic or optimizer.

The included FNO wind layer is a screening-level scalar speed field. It is not
an XLB velocity field and does not include vectors, turbulence statistics or
indoor ventilation.

Playback caches are included only for the two timeline examples. Newly generated
caches, JSON outputs, checkpoints and datasets should remain outside Git.

## Rebuilding

From the repository root:

    uv pip install -e ".[dev,viz,interactive]"
    $env:MOKUMITSU_CHECKPOINT_DIR = "D:\models\mokumitsu"
    $env:MOKUMITSU_PYTHON = "$PWD\.venv\Scripts\python.exe"
    .venv\Scripts\python.exe scripts\download_models.py --profile all
    hython houdini\build_joint_design_hip.py
    .venv\Scripts\python.exe scripts\evaluate_joint_feasibility.py --include-districts
    hython houdini\build_joint_feasibility_hip.py
    hython houdini\build_mokumitsu_hip.py

The live worker requires fno_residential_xlb.pt. The embedded-Python fallback
and the timeline builders require fno_residential_ts.pt and its JSON metadata.
The verified downloader obtains them from the
[residential model release](https://github.com/otanl/mokumitsu-urban-renewal/releases/tag/models-residential-v1).
