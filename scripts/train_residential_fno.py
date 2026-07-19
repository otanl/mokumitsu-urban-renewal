"""Train the residential FNO from a validated physical-coordinate XLB dataset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from validate_residential_dataset import validate

ROOT = Path(__file__).resolve().parents[1]


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "residential_xlb_v2.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "checkpoints" / "fno_residential_xlb.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restore the checkpoint with the lowest validation relative L2",
    )
    return parser


def _split_indices(sample_kind: np.ndarray, seed: int):
    count = len(sample_kind)
    rng = np.random.default_rng(seed)
    empty = np.flatnonzero(sample_kind == 2)
    regular = np.flatnonzero(sample_kind != 2)
    regular = regular[rng.permutation(len(regular))]
    test_count = max(2, round(count * 0.15))
    validation_count = max(2, round(count * 0.15))
    if test_count + validation_count >= len(regular):
        raise ValueError("dataset is too small for independent train/validation/test splits")
    test = regular[:test_count]
    validation = regular[test_count : test_count + validation_count]
    train = np.concatenate((regular[test_count + validation_count :], empty))
    return train, validation, test


def _evaluate(
    model,
    inputs: torch.Tensor,
    targets_raw: torch.Tensor,
    indices: torch.Tensor,
    ymean: float,
    ystd: float,
) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(inputs[indices]) * ystd + ymean
        return relative_l2(prediction, targets_raw[indices])


def main() -> None:
    args = build_parser().parse_args()
    from neuralop.models import FNO

    if min(args.modes, args.hidden, args.layers, args.epochs, args.batch_size) <= 0:
        raise ValueError("modes, hidden, layers, epochs and batch-size must be positive")
    dataset_path = args.data.resolve()
    report = validate(dataset_path)
    if not report["ok"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with np.load(dataset_path, allow_pickle=False) as data:
        input_fields = np.asarray(data["input"], dtype=np.float32)
        speed_fields = np.asarray(data["speed"], dtype=np.float32)
        sample_kind = np.asarray(data["sample_kind"], dtype=np.uint8)
        dataset_metadata = json.loads(str(np.asarray(data["metadata"]).item()))
    sample_count, ny, nx = input_fields.shape
    if sample_count < 24:
        raise ValueError("at least 24 validated samples are required")
    if args.modes > min(ny, nx) // 2:
        raise ValueError("modes cannot exceed half the smaller dataset grid dimension")

    train_np, validation_np, test_np = _split_indices(sample_kind, args.seed)
    xmean = float(np.mean(input_fields[train_np], dtype=np.float64))
    xstd = float(np.std(input_fields[train_np], dtype=np.float64))
    ymean = float(np.mean(speed_fields[train_np], dtype=np.float64))
    ystd = float(np.std(speed_fields[train_np], dtype=np.float64))
    if xstd <= 0 or ystd <= 0:
        raise ValueError("training fields must have non-zero variance")

    device = torch.device(args.device)
    inputs_raw = torch.from_numpy(input_fields).unsqueeze(1).to(device)
    targets_raw = torch.from_numpy(speed_fields).unsqueeze(1).to(device)
    inputs = (inputs_raw - xmean) / xstd
    targets = (targets_raw - ymean) / ystd
    train_indices = torch.from_numpy(train_np).to(device)
    validation_indices = torch.from_numpy(validation_np).to(device)
    test_indices = torch.from_numpy(test_np).to(device)

    config = {
        "n_modes": (args.modes, args.modes),
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": args.hidden,
        "n_layers": args.layers,
    }
    model = FNO(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_function = torch.nn.MSELoss()
    best_validation_l2 = float("inf")
    best_state = None
    best_epoch = -1

    print(
        f"device={device} samples={sample_count} train={len(train_indices)} "
        f"validation={len(validation_indices)} test={len(test_indices)} grid={ny}x{nx}"
    )
    for epoch in range(args.epochs):
        model.train()
        shuffled = train_indices[torch.randperm(len(train_indices), device=device)]
        total_loss = 0.0
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()

        validation_l2 = _evaluate(
            model,
            inputs,
            targets_raw,
            validation_indices,
            ymean,
            ystd,
        )
        if args.best and validation_l2 < best_validation_l2:
            best_validation_l2 = validation_l2
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch={epoch:03d} train_mse={total_loss / len(shuffled):.4e} "
                f"validation_rel_l2={validation_l2:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"restored best validation checkpoint: epoch={best_epoch} "
            f"relative_l2={best_validation_l2:.4f}"
        )
    selected_validation_l2 = (
        best_validation_l2
        if best_state is not None
        else _evaluate(model, inputs, targets_raw, validation_indices, ymean, ystd)
    )
    test_l2 = _evaluate(model, inputs, targets_raw, test_indices, ymean, ystd)
    print(f"final untouched test relative_l2={test_l2:.4f}")

    split = {
        "seed": args.seed,
        "train_indices": train_np.tolist(),
        "validation_indices": validation_np.tolist(),
        "test_indices": test_np.tolist(),
        "best_epoch": best_epoch if best_state is not None else args.epochs - 1,
        "validation_relative_l2": selected_validation_l2,
        "test_relative_l2": test_l2,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "ymean": ymean,
        "ystd": ystd,
        "xmean": xmean,
        "xstd": xstd,
        "ny": ny,
        "nx": nx,
        "sharpness": 60.0,
        "fno_cfg": config,
        "physics": dataset_metadata["physics"],
        "dataset_sha256": report["sha256"],
        "split": split,
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
