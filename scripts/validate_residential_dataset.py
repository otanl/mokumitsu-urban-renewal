"""Validate a Mokumitsu XLB dataset before normalization or training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from houdini_xlb import BACKEND_SIGNATURE, XlbConfig

REQUIRED_ARRAYS = {
    "input",
    "speed",
    "seed",
    "redevelopment_count",
    "building_count",
    "sample_kind",
    "peak_speed_ratio",
    "metadata",
}


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_indices(fields: np.ndarray) -> list[tuple[int, int]]:
    seen: dict[bytes, int] = {}
    duplicates = []
    for index, field in enumerate(fields):
        digest = hashlib.sha256(np.ascontiguousarray(field).tobytes()).digest()
        if digest in seen:
            duplicates.append((seen[digest], index))
        else:
            seen[digest] = index
    return duplicates


def validate(
    path: Path,
    *,
    require_grid_verified: bool = True,
) -> dict[str, object]:
    path = path.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_ARRAYS.difference(data.files)
        if "input" in missing or "speed" in missing:
            raise ValueError("dataset must contain input and speed arrays")
        if missing:
            errors.append(f"dataset is missing arrays: {sorted(missing)}")
        raw_metadata = (
            json.loads(str(np.asarray(data["metadata"]).item())) if "metadata" in data else {}
        )
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        if metadata is not raw_metadata:
            errors.append("dataset metadata must be a JSON object")
        inputs = np.asarray(data["input"], dtype=np.float32)
        speeds = np.asarray(data["speed"], dtype=np.float32)
        count = inputs.shape[0] if inputs.ndim == 3 else 0
        seeds = (
            np.asarray(data["seed"], dtype=np.int64)
            if "seed" in data
            else np.arange(count, dtype=np.int64)
        )
        projects = (
            np.asarray(data["redevelopment_count"], dtype=np.int64)
            if "redevelopment_count" in data
            else np.zeros(count, dtype=np.int64)
        )
        buildings = (
            np.asarray(data["building_count"], dtype=np.int64)
            if "building_count" in data
            else np.zeros(count, dtype=np.int64)
        )
        kinds = (
            np.asarray(data["sample_kind"], dtype=np.uint8)
            if "sample_kind" in data
            else np.zeros(count, dtype=np.uint8)
        )
        stored_peaks = (
            np.asarray(data["peak_speed_ratio"], dtype=np.float32)
            if "peak_speed_ratio" in data
            else None
        )

    physics = metadata.get("physics")
    if metadata.get("dataset_contract_version") != 2 or not isinstance(physics, dict):
        errors.append("missing Mokumitsu dataset contract v2 metadata")
        physics = {}
    if metadata.get("generator") != "mokumitsu-residential-xlb-v2":
        errors.append("dataset generator identity is missing or unsupported")
    if physics.get("contract_version") != 1:
        errors.append("unsupported physical contract version")
    if physics.get("backend_signature") != BACKEND_SIGNATURE:
        errors.append("dataset backend signature does not match installed houdini-xlb")
    if physics.get("height_encoding") != "fraction_of_domain_height":
        errors.append("height encoding must be fraction_of_domain_height")
    grid_verification = physics.get("grid_verification")
    grid_passed = (
        isinstance(grid_verification, dict)
        and grid_verification.get("passed") is True
        and grid_verification.get("compatible") is True
        and grid_verification.get("override") is False
        and _is_sha256(grid_verification.get("report_sha256"))
    )
    if require_grid_verified and not grid_passed:
        errors.append("dataset was not generated from a passing grid-independence protocol")
    elif not grid_passed:
        warnings.append("grid-independence protocol did not pass; experimental data only")

    config_data = physics.get("xlb_config")
    try:
        config = XlbConfig.from_dict(config_data) if isinstance(config_data, dict) else None
    except (TypeError, ValueError) as exc:
        config = None
        errors.append(f"invalid XLB config: {exc}")
    if config is None:
        errors.append("physical contract has no valid XLB config")
    elif config_data != config.to_dict():
        errors.append("physical xlb_config must record the complete canonical XLB config")
    if config is not None:
        scalar_contract = {
            "domain_length_x_m": config.domain_length_x_m,
            "domain_length_y_m": config.domain_length_y_m,
            "domain_height_m": config.domain_height_m,
            "reference_height_m": config.reference_height_m,
            "pedestrian_height_m": config.pedestrian_height_m,
            "resolved_pedestrian_height_m": config.resolved_pedestrian_height_m,
        }
        for field, expected in scalar_contract.items():
            try:
                actual = float(physics[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"physical contract is missing numeric field {field}")
                continue
            if not np.isclose(actual, expected, rtol=1e-8, atol=1e-8):
                errors.append(f"physical {field} is inconsistent with XLB config")
        try:
            cell_sizes = np.asarray(physics["cell_sizes_m"], dtype=float)
        except (KeyError, TypeError, ValueError):
            errors.append("physical contract has no valid cell_sizes_m")
        else:
            if cell_sizes.shape != (3,) or not np.allclose(
                cell_sizes,
                config.cell_sizes_m,
                rtol=1e-8,
                atol=1e-8,
            ):
                errors.append("physical cell_sizes_m is inconsistent with XLB config")

    if inputs.ndim != 3 or speeds.shape != inputs.shape:
        errors.append("input and speed must share shape (sample, ny, nx)")
    elif physics.get("output_grid") != list(inputs.shape[1:]):
        errors.append("physical output_grid does not match dataset arrays")
    if config is not None and physics.get("xlb_grid_xyz") != list(config.grid_xyz):
        errors.append("physical xlb_grid_xyz does not match XLB config")
    sample_count = inputs.shape[0] if inputs.ndim == 3 else 0
    sample_arrays = [
        ("seed", seeds),
        ("redevelopment_count", projects),
        ("building_count", buildings),
        ("sample_kind", kinds),
    ]
    if stored_peaks is not None:
        sample_arrays.append(("peak_speed_ratio", stored_peaks))
    metadata_arrays_ok = True
    for name, array in sample_arrays:
        if array.shape != (sample_count,):
            metadata_arrays_ok = False
            errors.append(f"{name} must have one value per sample")
    if metadata.get("accepted_samples") != sample_count:
        errors.append("metadata accepted_samples does not match the arrays")
    if metadata.get("target_samples") != sample_count:
        errors.append("dataset is incomplete: target_samples does not match the arrays")
    if not isinstance(metadata.get("next_seed"), int):
        errors.append("metadata next_seed must be an integer")
    if metadata_arrays_ok:
        if sample_count and len(np.unique(seeds)) != sample_count:
            errors.append("sample seeds are not unique")
        if np.any(projects < 0) or np.any(projects > 2) or np.any(buildings < 0):
            errors.append("redevelopment_count must be in [0, 2] and building_count non-negative")
        if not np.isin(kinds, (0, 1, 2)).all():
            errors.append("sample_kind must use only baseline=0, redevelopment=1, empty=2")
        if np.any((kinds == 1) != (projects > 0)):
            errors.append("redevelopment sample_kind is inconsistent with redevelopment_count")
        if np.any((kinds != 2) & (buildings == 0)):
            errors.append("non-empty samples must contain at least one building")

    if not np.isfinite(inputs).all() or inputs.min(initial=0) < 0 or inputs.max(initial=0) > 1:
        errors.append("input contains non-finite values or leaves [0, 1]")
    if not np.isfinite(speeds).all() or speeds.min(initial=0) < 0:
        errors.append("speed contains non-finite or negative values")

    duplicate_inputs = _duplicate_indices(inputs) if inputs.ndim == 3 else []
    if duplicate_inputs:
        errors.append(f"duplicate input fields: {duplicate_inputs[:8]}")
    if sample_count and speeds.shape == inputs.shape and metadata_arrays_ok:
        actual_peaks = speeds.reshape(sample_count, -1).max(axis=1)
        if config is not None:
            peak_ratios = actual_peaks / config.wind
            if np.any(peak_ratios > config.max_speed_ratio + 1e-5):
                bad = np.flatnonzero(peak_ratios > config.max_speed_ratio + 1e-5)
                errors.append(f"unstable peak speed in samples {bad[:16].tolist()}")
            if stored_peaks is not None:
                if (
                    not np.isfinite(stored_peaks).all()
                    or np.any(stored_peaks < peak_ratios - 1e-5)
                    or np.any(stored_peaks > config.max_speed_ratio + 1e-5)
                ):
                    errors.append(
                        "stored lattice peak_speed_ratio is inconsistent with output fields"
                    )
        sample_rms = np.sqrt(np.mean(speeds.astype(np.float64) ** 2, axis=(1, 2)))
        median = float(np.median(sample_rms))
        mad = float(np.median(np.abs(sample_rms - median)))
        if mad > 0:
            robust_z = np.abs(sample_rms - median) / (1.4826 * mad)
            suspicious = np.flatnonzero((robust_z > 15) & (sample_rms > 3 * median))
            if len(suspicious):
                errors.append(f"statistical speed outliers: {suspicious[:16].tolist()}")
        empty_indices = np.flatnonzero(kinds == 2)
        if len(empty_indices) != 1:
            errors.append("dataset must contain exactly one empty-domain reference sample")
        else:
            empty_index = int(empty_indices[0])
            if (
                np.any(inputs[empty_index] != 0)
                or projects[empty_index] != 0
                or buildings[empty_index] != 0
                or float(speeds[empty_index].mean()) <= 0
            ):
                errors.append("empty-domain reference sample has inconsistent geometry metadata")
        if np.any(np.abs(speeds[inputs > 1e-7]) > 1e-6):
            errors.append("speed must be zero inside solid height-map cells")
        if sample_count >= 50 and not np.any(projects > 0):
            errors.append("dataset contains no joint-redevelopment samples")
        if float(speeds.std()) <= 0 or float(inputs.std()) <= 0:
            errors.append("dataset fields must have non-zero variance")
    elif sample_count == 0:
        errors.append("dataset is empty")

    if config is not None:
        resolved = float(physics.get("resolved_pedestrian_height_m", -1))
        if not np.isclose(resolved, config.resolved_pedestrian_height_m):
            errors.append("resolved pedestrian height metadata is inconsistent")
        if abs(resolved - config.pedestrian_height_m) > config.cell_sizes_m[2] / 2 + 1e-8:
            errors.append("pedestrian slice is farther than half a cell from its target")

    report = {
        "ok": not errors,
        "path": str(path),
        "sha256": _sha256(path),
        "samples": sample_count,
        "shape": list(inputs.shape),
        "input_range": [float(inputs.min(initial=0)), float(inputs.max(initial=0))],
        "speed_range": [float(speeds.min(initial=0)), float(speeds.max(initial=0))],
        "speed_std": float(speeds.std()) if speeds.size else 0.0,
        "errors": errors,
        "warnings": warnings,
        "physics": physics,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="audit experimental data without requiring a passing grid protocol",
    )
    args = parser.parse_args()
    report = validate(
        args.dataset,
        require_grid_verified=not args.allow_unconverged,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"ok={report['ok']} samples={report['samples']} shape={report['shape']} "
            f"speed={report['speed_range']} sha256={report['sha256']}"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
