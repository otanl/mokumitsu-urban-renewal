from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from houdini_xlb import BACKEND_SIGNATURE, XlbConfig

from scripts.export_residential_torchscript import _validate_checkpoint
from scripts.gen_residential_dataset import _grid_verification
from scripts.validate_residential_dataset import validate


def _config() -> XlbConfig:
    return XlbConfig(
        grid_x=80,
        grid_y=80,
        grid_z=48,
        steps=64,
        domain_length_x_m=100.0,
        domain_length_y_m=100.0,
        domain_height_m=60.0,
        reference_height_m=10.0,
        pedestrian_height_m=1.5,
        average_window=16,
        average_every=4,
    )


def _write_dataset(path: Path, *, grid_passed: bool = True) -> Path:
    config = _config()
    inputs = np.zeros((3, 8, 8), dtype=np.float32)
    inputs[1, 2:4, 2:4] = 0.15
    inputs[2, 4:7, 1:3] = 0.20
    speeds = np.stack(
        [
            np.full((8, 8), 0.040, dtype=np.float32),
            np.full((8, 8), 0.035, dtype=np.float32),
            np.full((8, 8), 0.045, dtype=np.float32),
        ]
    )
    speeds[inputs > 0] = 0.0
    peaks = speeds.reshape(3, -1).max(axis=1) / config.wind + 0.05
    verification = {
        "passed": grid_passed,
        "compatible": True,
        "override": not grid_passed,
        "report_sha256": "a" * 64,
    }
    physics = {
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
        "output_grid": [8, 8],
        "xlb_config": config.to_dict(),
        "grid_verification": verification,
    }
    metadata = {
        "dataset_contract_version": 2,
        "generator": "mokumitsu-residential-xlb-v2",
        "target_samples": 3,
        "accepted_samples": 3,
        "next_seed": 3,
        "rejected": [],
        "physics": physics,
    }
    np.savez_compressed(
        path,
        input=inputs,
        speed=speeds,
        seed=np.asarray([0, 1, 2], dtype=np.int64),
        redevelopment_count=np.asarray([0, 0, 1], dtype=np.int16),
        building_count=np.asarray([0, 4, 3], dtype=np.int16),
        sample_kind=np.asarray([2, 0, 1], dtype=np.uint8),
        peak_speed_ratio=peaks.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return path


def test_valid_v2_dataset_passes_the_strict_contract(tmp_path):
    report = validate(_write_dataset(tmp_path / "valid.npz"))
    assert report["ok"]
    assert report["errors"] == []


def test_unconverged_dataset_is_only_allowed_as_experimental_data(tmp_path):
    path = _write_dataset(tmp_path / "experimental.npz", grid_passed=False)
    strict = validate(path)
    assert not strict["ok"]
    assert any("grid-independence" in error for error in strict["errors"])

    audit = validate(path, require_grid_verified=False)
    assert audit["ok"]
    assert any("experimental data only" in warning for warning in audit["warnings"])


def test_grid_report_must_match_the_exact_xlb_configuration(tmp_path):
    config = _config()
    report_path = tmp_path / "grid.json"
    report_path.write_text(
        json.dumps(
            {
                "backend_signature": BACKEND_SIGNATURE,
                "height_encoding": "fraction_of_domain_height",
                "passed": True,
                "mean_index_drift": 0.01,
                "dead_fraction_drift": 0.02,
                "rows": [{"xlb_config": config.to_dict()}],
            }
        ),
        encoding="utf-8",
    )
    verified = _grid_verification(report_path, config, allow_unconverged=False)
    assert verified["passed"]
    assert verified["compatible"]
    assert not verified["override"]

    with pytest.raises(RuntimeError, match="has not passed"):
        _grid_verification(
            report_path,
            replace(config, steps=config.steps + 1),
            allow_unconverged=False,
        )


def test_torchscript_export_rejects_missing_or_unverified_provenance():
    with pytest.raises(ValueError, match="lacks required fields"):
        _validate_checkpoint({})

    checkpoint = {
        "state_dict": {},
        "fno_cfg": {},
        "ny": 8,
        "nx": 8,
        "xmean": 0.0,
        "xstd": 1.0,
        "ymean": 0.0,
        "ystd": 1.0,
        "sharpness": 60.0,
        "physics": {
            "contract_version": 1,
            "height_encoding": "fraction_of_domain_height",
            "output_grid": [8, 8],
            "grid_verification": {
                "passed": False,
                "compatible": True,
                "override": True,
            },
        },
        "dataset_sha256": "a" * 64,
        "split": {
            "train_indices": [0],
            "validation_indices": [1],
            "test_indices": [2],
        },
    }
    with pytest.raises(ValueError, match="passing physical-grid"):
        _validate_checkpoint(checkpoint)

    checkpoint["physics"]["grid_verification"] = {
        "passed": True,
        "compatible": True,
        "override": False,
    }
    assert _validate_checkpoint(checkpoint) is checkpoint
