"""Generate a physically-scaled Mokumitsu residential-wind dataset with XLB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from houdini_xlb import BACKEND_SIGNATURE, XlbConfig, analyze_heightmap

from mokumitsu import MokumitsuConfig, generate_mokumitsu
from mokumitsu.joint_renewal import (
    JointRenewalPolicy,
    joint_redevelopment_variants,
    joint_renewal_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--gridxy", type=int, default=200)
    parser.add_argument("--gridz", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--domain-height-m", type=float, default=60.0)
    parser.add_argument("--reference-height-m", type=float, default=10.0)
    parser.add_argument("--pedestrian-height-m", type=float, default=1.5)
    parser.add_argument("--wind", type=float, default=0.05)
    parser.add_argument("--reynolds", type=float, default=8000.0)
    parser.add_argument("--average-window", type=int, default=1000)
    parser.add_argument("--average-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument(
        "--grid-report",
        type=Path,
        default=ROOT / "outputs" / "residential_grid_independence.json",
    )
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="generate an experimental dataset even when the grid protocol has not passed",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "residential_xlb_v2.npz",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "artifacts" / "houdini" / "cache" / "residential-dataset-v2",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _generation_config(rng: np.random.Generator) -> MokumitsuConfig:
    return replace(
        MokumitsuConfig(),
        narrow_street_probability=float(rng.uniform(0.45, 0.72)),
        target_lot_frontage_m=float(rng.uniform(6.5, 8.0)),
        target_lot_depth_m=float(rng.uniform(8.0, 10.0)),
        target_net_building_coverage=float(rng.uniform(0.46, 0.62)),
        historic_routes=int(rng.integers(1, 4)),
        branch_alleys=int(rng.integers(6, 13)),
        dead_end_probability=float(rng.uniform(0.55, 0.78)),
        loop_probability=float(rng.uniform(0.12, 0.30)),
        target_lot_area_m2=float(rng.uniform(65.0, 95.0)),
        flag_lot_probability=float(rng.uniform(0.40, 0.65)),
    )


def _with_redevelopment(district, rng: np.random.Generator):
    requested = int(rng.choice((0, 1, 2), p=(0.50, 0.38, 0.12)))
    completed = 0
    for _ in range(requested):
        candidates = joint_renewal_candidates(district)
        if not candidates:
            break
        candidate = candidates[int(rng.integers(min(12, len(candidates))))]
        policy = replace(
            JointRenewalPolicy(),
            year=int(rng.integers(2030, 2046)),
            maximum_floors=int(rng.integers(4, 7)),
        )
        direction = float(rng.choice((0.0, 90.0, 180.0, 270.0)))
        variants = joint_redevelopment_variants(
            district,
            candidate,
            policy,
            wind_direction_deg=direction,
        )
        if not variants:
            continue
        district = variants[int(rng.integers(len(variants)))][0]
        completed += 1
    return district, completed


def _xlb_config(args: argparse.Namespace) -> XlbConfig:
    dx = 100.0 / args.gridxy
    grid_z = args.gridz or round(args.domain_height_m / dx)
    return XlbConfig(
        grid_x=args.gridxy,
        grid_y=args.gridxy,
        grid_z=grid_z,
        steps=args.steps,
        wind=args.wind,
        reynolds=args.reynolds,
        domain_length_x_m=100.0,
        domain_length_y_m=100.0,
        domain_height_m=args.domain_height_m,
        reference_height_m=args.reference_height_m,
        pedestrian_height_m=args.pedestrian_height_m,
        average_window=args.average_window,
        average_every=args.average_every,
    )


def _grid_verification(
    path: Path,
    config: XlbConfig,
    *,
    allow_unconverged: bool,
) -> dict[str, object]:
    path = path.resolve()
    if not path.exists():
        if not allow_unconverged:
            raise FileNotFoundError(
                f"grid report not found at {path}; run verify_residential_grid.py first"
            )
        return {"passed": False, "override": True, "reason": "missing grid report"}
    raw = path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    rows = report.get("rows", [])
    finest = rows[-1] if rows else {}
    compatible = (
        report.get("backend_signature") == BACKEND_SIGNATURE
        and finest.get("xlb_config") == config.to_dict()
        and report.get("height_encoding") == "fraction_of_domain_height"
    )
    passed = bool(report.get("passed")) and compatible
    if not passed and not allow_unconverged:
        raise RuntimeError(
            "grid-independence protocol has not passed for the requested dataset config"
        )
    return {
        "passed": passed,
        "override": bool(allow_unconverged and not passed),
        "compatible": compatible,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "mean_index_drift": report.get("mean_index_drift"),
        "dead_fraction_drift": report.get("dead_fraction_drift"),
    }


def _physics(
    config: XlbConfig,
    res: int,
    grid_verification: dict[str, object],
) -> dict[str, object]:
    return {
        "contract_version": 1,
        "solver": "XLB",
        "backend_signature": BACKEND_SIGNATURE,
        "height_encoding": "fraction_of_domain_height",
        "domain_length_x_m": config.domain_length_x_m,
        "domain_length_y_m": config.domain_length_y_m,
        "domain_height_m": config.domain_height_m,
        "reference_height_m": config.reference_height_m,
        "pedestrian_height_m": config.pedestrian_height_m,
        "resolved_pedestrian_height_m": config.resolved_pedestrian_height_m,
        "cell_sizes_m": list(config.cell_sizes_m),
        "xlb_grid_xyz": list(config.grid_xyz),
        "output_grid": [res, res],
        "xlb_config": config.to_dict(),
        "grid_verification": grid_verification,
    }


def _save(
    path: Path,
    arrays: dict[str, list],
    *,
    physics: dict[str, object],
    target_samples: int,
    next_seed: int,
    rejected: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_contract_version": 2,
        "generator": "mokumitsu-residential-xlb-v2",
        "physics": physics,
        "target_samples": target_samples,
        "accepted_samples": len(arrays["input"]),
        "next_seed": next_seed,
        "rejected": rejected,
    }
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            input=np.asarray(arrays["input"], dtype=np.float32),
            speed=np.asarray(arrays["speed"], dtype=np.float32),
            seed=np.asarray(arrays["seed"], dtype=np.int64),
            redevelopment_count=np.asarray(arrays["redevelopment_count"], dtype=np.int16),
            building_count=np.asarray(arrays["building_count"], dtype=np.int16),
            sample_kind=np.asarray(arrays["sample_kind"], dtype=np.uint8),
            peak_speed_ratio=np.asarray(arrays["peak_speed_ratio"], dtype=np.float32),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _empty_arrays() -> dict[str, list]:
    return {
        "input": [],
        "speed": [],
        "seed": [],
        "redevelopment_count": [],
        "building_count": [],
        "sample_kind": [],
        "peak_speed_ratio": [],
    }


def _resume(path: Path, physics: dict[str, object], res: int):
    arrays = _empty_arrays()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        if not isinstance(metadata, dict):
            raise ValueError("existing dataset metadata must be a JSON object")
        if (
            metadata.get("dataset_contract_version") != 2
            or metadata.get("generator") != "mokumitsu-residential-xlb-v2"
        ):
            raise ValueError("existing dataset does not use the resumable v2 contract")
        if metadata.get("physics") != physics:
            raise ValueError("existing dataset uses a different physical contract")
        if data["input"].ndim != 3 or data["input"].shape[1:] != (res, res):
            raise ValueError("existing dataset grid does not match --res")
        count = int(data["input"].shape[0])
        if data["speed"].shape != data["input"].shape:
            raise ValueError("existing input and speed arrays do not share a shape")
        for name in arrays:
            if name not in data or data[name].shape[0] != count:
                raise ValueError(f"existing dataset array {name!r} has the wrong length")
            arrays[name].extend(np.asarray(data[name]))
        if metadata.get("accepted_samples") != count:
            raise ValueError("existing accepted_samples metadata is inconsistent")
        if count and sum(int(kind == 2) for kind in arrays["sample_kind"]) != 1:
            raise ValueError("existing partial dataset must contain one empty reference")
        if not isinstance(metadata.get("next_seed"), int):
            raise ValueError("existing dataset has no valid next_seed")
    return arrays, int(metadata["next_seed"]), list(metadata.get("rejected", []))


def main() -> None:
    from scipy.ndimage import zoom

    args = build_parser().parse_args()
    if (
        args.n <= 0
        or args.res < 16
        or args.gridxy < 32
        or args.save_every <= 0
        or args.max_attempts < 0
    ):
        raise ValueError("n/save-every must be positive, res >= 16 and gridxy >= 32")
    config = _xlb_config(args)
    grid_verification = _grid_verification(
        args.grid_report,
        config,
        allow_unconverged=args.allow_unconverged,
    )
    physics = _physics(config, args.res, grid_verification)
    output = args.out.resolve()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"{output} exists; use --resume or choose another output")
        arrays, candidate_seed, rejected = _resume(output, physics, args.res)
    else:
        arrays, candidate_seed, rejected = _empty_arrays(), args.seed, []
    if len(arrays["input"]) > args.n:
        raise ValueError("existing dataset already exceeds the requested sample count")

    max_attempts = args.max_attempts or max(args.n * 4, 32)
    attempts = 0
    while len(arrays["input"]) < args.n and attempts < max_attempts:
        sample_seed = candidate_seed
        candidate_seed += 1
        attempts += 1
        rng = np.random.default_rng(sample_seed)
        if not arrays["input"]:
            redevelopment_count = 0
            sample_kind = 2
            heightmap = np.zeros((args.res, args.res), dtype=np.float32)
            xlb_heightmap = np.zeros((config.grid_y, config.grid_x), dtype=np.float32)
            building_count = 0
        else:
            district = generate_mokumitsu(sample_seed, _generation_config(rng))
            district, redevelopment_count = _with_redevelopment(district, rng)
            sample_kind = int(redevelopment_count > 0)
            heightmap = district.heightmap(args.res, config.domain_height_m)
            xlb_heightmap = district.heightmap(config.grid_x, config.domain_height_m)
            building_count = len(district.buildings)
        try:
            result = analyze_heightmap(xlb_heightmap, config, cache_dir=args.cache)
        except RuntimeError as exc:
            rejected.append({"seed": sample_seed, "error": str(exc)})
            print(f"rejected seed={sample_seed}: {exc}", flush=True)
            continue
        speed = zoom(
            result.speed,
            (args.res / config.grid_y, args.res / config.grid_x),
            order=1,
        ).astype(np.float32)
        if speed.shape != heightmap.shape or not np.isfinite(speed).all():
            raise RuntimeError("XLB output resampling produced an invalid field")
        speed[heightmap > 1e-7] = 0.0

        arrays["input"].append(heightmap)
        arrays["speed"].append(speed)
        arrays["seed"].append(sample_seed)
        arrays["redevelopment_count"].append(redevelopment_count)
        arrays["building_count"].append(building_count)
        arrays["sample_kind"].append(sample_kind)
        arrays["peak_speed_ratio"].append(float(result.speed.max() / config.wind))
        count = len(arrays["input"])
        print(
            f"accepted {count}/{args.n} seed={sample_seed} projects={redevelopment_count} "
            f"peak/U={arrays['peak_speed_ratio'][-1]:.3f} cache={result.cache_hit}",
            flush=True,
        )
        if count % args.save_every == 0 or count == args.n:
            _save(
                output,
                arrays,
                physics=physics,
                target_samples=args.n,
                next_seed=candidate_seed,
                rejected=rejected,
            )

    if len(arrays["input"]) != args.n:
        if arrays["input"]:
            _save(
                output,
                arrays,
                physics=physics,
                target_samples=args.n,
                next_seed=candidate_seed,
                rejected=rejected,
            )
        raise RuntimeError(
            f"accepted {len(arrays['input'])}/{args.n} samples after {attempts} attempts"
        )
    print(f"saved validated dataset: {output}", flush=True)


if __name__ == "__main__":
    main()
