"""Retrain the residential FNO from the release-hosted XLB dataset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "residential_xlb.npz")
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
        action="store_true",
        help="restore the checkpoint with the lowest held-out relative L2",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from neuralop.models import FNO

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.data.resolve(), allow_pickle=False)
    if "input" not in data or "speed" not in data:
        raise ValueError("dataset must contain input and speed arrays")
    input_fields = np.asarray(data["input"], dtype=np.float32)
    speed_fields = np.asarray(data["speed"], dtype=np.float32)
    if input_fields.shape != speed_fields.shape or input_fields.ndim != 3:
        raise ValueError("input and speed must have the same (sample, ny, nx) shape")

    sample_count, ny, nx = input_fields.shape
    if sample_count < 12:
        raise ValueError("at least 12 samples are required for the held-out split")
    device = torch.device(args.device)
    inputs_raw = torch.from_numpy(input_fields).unsqueeze(1).to(device)
    targets_raw = torch.from_numpy(speed_fields).unsqueeze(1).to(device)
    xmean, xstd = float(inputs_raw.mean()), float(inputs_raw.std())
    ymean, ystd = float(targets_raw.mean()), float(targets_raw.std())
    if xstd <= 0 or ystd <= 0:
        raise ValueError("dataset fields must have non-zero variance")
    inputs = (inputs_raw - xmean) / xstd
    targets = (targets_raw - ymean) / ystd

    permutation = torch.randperm(sample_count)
    test_count = max(8, sample_count // 6)
    test_indices = permutation[:test_count].to(device)
    train_indices = permutation[test_count:].to(device)
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
    best_relative_l2 = float("inf")
    best_state = None

    print(
        f"device={device} samples={sample_count} train={len(train_indices)} "
        f"test={len(test_indices)} grid={ny}x{nx}"
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

        should_evaluate = args.best or epoch % 20 == 0 or epoch == args.epochs - 1
        if should_evaluate:
            model.eval()
            with torch.no_grad():
                prediction = model(inputs[test_indices]) * ystd + ymean
                score = relative_l2(prediction, targets_raw[test_indices])
            if args.best and score < best_relative_l2:
                best_relative_l2 = score
                best_state = copy.deepcopy(model.state_dict())
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                print(
                    f"epoch={epoch:03d} train_mse={total_loss / len(shuffled):.4e} "
                    f"test_rel_l2={score:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best held-out checkpoint: relative_l2={best_relative_l2:.4f}")
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
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
