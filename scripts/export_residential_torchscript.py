"""Export a trained residential FNO checkpoint as portable TorchScript."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "fno_residential_xlb.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "checkpoints" / "fno_residential_ts.pt",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from neuralop.models import FNO

    checkpoint = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    model = FNO(**checkpoint["fno_cfg"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    ny, nx = int(checkpoint["ny"]), int(checkpoint["nx"])
    example = torch.zeros(1, 1, ny, nx)
    with warnings.catch_warnings(), torch.no_grad():
        # NeuralOperator branches on the fixed grid dimensions while tracing.
        # This export intentionally supports only the checkpoint's recorded grid.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        traced = torch.jit.trace(model, example, check_trace=False)
        test_input = torch.rand(1, 1, ny, nx)
        max_difference = float((model(test_input) - traced(test_input)).abs().max())
    if max_difference > 1e-5:
        raise RuntimeError(f"TorchScript trace mismatch: max difference {max_difference:.3e}")

    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    metadata = {
        key: float(checkpoint[key]) for key in ("xmean", "xstd", "ymean", "ystd", "sharpness")
    }
    metadata.update(ny=ny, nx=nx)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {output}")
    print(f"saved {metadata_path}")
    print(f"trace max difference: {max_difference:.3e}")


if __name__ == "__main__":
    main()
