# Residential FNO model assets v1 (quarantined)

> **Do not use these assets for design evaluation or train a replacement model
> from this dataset.** The normal downloader intentionally refuses this release.

## Audit status — 2026-07-19

The v1 release is retained only so the failure can be reproduced. It is not a
supported model release.

The audit found three invalidating defects:

- finite but catastrophic XLB sample 374 (`max(U/U0) ≈ 6719`), which inflated
  the target standard deviation by about 9.1×;
- building heights normalized by the 100 m horizontal domain while the old
  lattice used a different vertical spacing, so the nominal pedestrian slice
  and geometry did not have a consistent metric meaning;
- normalization and checkpoint selection used information outside the training
  split, including selecting the checkpoint on the test split.

The corrected direct-raster, isotropic-grid protocol has also not yet passed its
grid-independence gate. Therefore filtering sample 374 is not sufficient and the
v1 dataset must not be retrained.

Full measurements and acceptance criteria are in
[`docs/WIND_VALIDATION_STATUS.md`](../docs/WIND_VALIDATION_STATUS.md).

## Audit-only retrieval

The manifest marks this release as `quarantined`. Historical reproduction
requires an explicit acknowledgement:

    .venv\Scripts\python.exe scripts\download_models.py --allow-quarantined --profile all --include-dataset

Every file is still checked against the exact byte size and SHA-256 recorded in
`models/manifest.json`. The override changes only access to the historical
artifact; it does not make the model valid.

## Requirements for a future v2 release

A replacement release must be generated only after the forward solver passes the
documented grid, boundary-condition and averaging gates. Its dataset and model
metadata must record the physical domain, isotropic lattice, exact raster shape,
resolved pedestrian height, backend signature, split provenance and quality
statistics. Training statistics must come from the training split, model
selection from validation, and the test split must remain untouched until final
reporting.

The asset files remain MIT-licensed; XLB is Apache-2.0. Quarantine is a
scientific-validity status, not a malware or licensing warning.
