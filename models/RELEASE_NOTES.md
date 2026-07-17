# Residential FNO model assets v1

This release makes the Mokumitsu residential wind-screening model reproducible
without storing large binaries in Git.

## Assets

- `fno_residential_ts.pt` and `fno_residential_ts.json`: portable
  TorchScript model used by the CLI, timeline builders and Houdini CPU fallback.
- `fno_residential_xlb.pt`: NeuralOperator training checkpoint used by the
  persistent CUDA/CPU live-design worker.
- `residential_xlb.npz`: 500 synthetic 128 × 128 input/speed pairs generated
  with XLB on the NVIDIA Warp backend.

Download and verify the checkpoint files from a repository clone:

    .venv\Scripts\python.exe scripts\download_models.py --profile all

Add `--include-dataset` to obtain the training dataset. Every download is
checked against its exact byte size and SHA-256 in
`models/manifest.json`. Existing mismatched files are never overwritten unless
`--force` is supplied.

The released dataset can recreate both runtime formats:

    .venv\Scripts\python.exe scripts\train_residential_fno.py --best
    .venv\Scripts\python.exe scripts\export_residential_torchscript.py

## Scope and license

The weights and dataset are released under the repository's MIT license. XLB,
the forward simulation engine used to generate the dataset, is Apache-2.0.

The model is a screening surrogate for comparing synthetic dense-residential
layouts. It is not a calibrated regulatory wind assessment, does not predict
indoor ventilation, and does not replace XLB or other validated CFD verification
of shortlisted designs.
