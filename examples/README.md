# Houdini examples

These scenes are curated playback artifacts for the Mokumitsu research
prototype. They were generated with Houdini Indie 20.5.684.

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

The cache is included only for these two small examples. Newly generated caches,
JSON outputs, checkpoints and datasets should remain outside Git.

## Rebuilding

From the repository root:

    .venv\Scripts\python.exe scripts\evaluate_joint_feasibility.py --include-districts
    hython houdini\build_joint_feasibility_hip.py
    hython houdini\build_mokumitsu_hip.py

Rebuilding wind-dependent scenes requires fno_residential_ts.pt and its JSON
metadata in the configured checkpoint directory.
