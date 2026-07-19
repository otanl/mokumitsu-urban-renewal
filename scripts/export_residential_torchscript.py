"""Export a trained residential FNO checkpoint as portable TorchScript."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _validate_checkpoint(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    required = {
        "state_dict",
        "fno_cfg",
        "ny",
        "nx",
        "xmean",
        "xstd",
        "ymean",
        "ystd",
        "sharpness",
        "physics",
        "dataset_sha256",
        "split",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint lacks required fields: {sorted(missing)}")
    physics = checkpoint["physics"]
    verification = physics.get("grid_verification") if isinstance(physics, dict) else None
    if not (
        isinstance(physics, dict)
        and physics.get("contract_version") == 1
        and physics.get("height_encoding") == "fraction_of_domain_height"
        and physics.get("output_grid") == [int(checkpoint["ny"]), int(checkpoint["nx"])]
    ):
        raise ValueError("checkpoint physical contract is missing or inconsistent")
    if not (
        isinstance(verification, dict)
        and verification.get("passed") is True
        and verification.get("compatible") is True
        and verification.get("override") is False
    ):
        raise ValueError("checkpoint does not carry a passing physical-grid verification")
    dataset_sha256 = str(checkpoint["dataset_sha256"]).lower()
    split = checkpoint["split"]
    if (
        len(dataset_sha256) != 64
        or not all(character in "0123456789abcdef" for character in dataset_sha256)
        or not isinstance(split, dict)
        or not {"train_indices", "validation_indices", "test_indices"}.issubset(split)
    ):
        raise ValueError("checkpoint dataset/split provenance is invalid")
    try:
        split_sets = [
            set(map(int, split[name]))
            for name in ("train_indices", "validation_indices", "test_indices")
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint split indices must be integer sequences") from exc
    if not all(split_sets) or any(
        left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1 :]
    ):
        raise ValueError("checkpoint splits must be non-empty and disjoint")
    statistics = ("xmean", "xstd", "ymean", "ystd", "sharpness")
    if not all(math.isfinite(float(checkpoint[field])) for field in statistics):
        raise ValueError("checkpoint normalization statistics must be finite")
    if (
        float(checkpoint["xstd"]) <= 0
        or float(checkpoint["ystd"]) <= 0
        or float(checkpoint["sharpness"]) <= 0
    ):
        raise ValueError("checkpoint scales and sharpness must be positive")
    return checkpoint


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

    checkpoint = _validate_checkpoint(
        torch.load(
            args.checkpoint.resolve(),
            map_location="cpu",
            weights_only=False,
        )
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
    metadata.update(
        ny=ny,
        nx=nx,
        physics=checkpoint["physics"],
        dataset_sha256=checkpoint["dataset_sha256"],
        split=checkpoint["split"],
    )
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {output}")
    print(f"saved {metadata_path}")
    print(f"trace max difference: {max_difference:.3e}")


if __name__ == "__main__":
    main()
