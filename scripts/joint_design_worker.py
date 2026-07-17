"""Persistent JSON-lines worker for GPU-accelerated Houdini design previews."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mokumitsu.accelerated import load_accelerated_model  # noqa: E402
from mokumitsu.design import (  # noqa: E402
    evaluate_joint_design,
    prepare_joint_design_baseline,
)
from mokumitsu.fire import FireScenario  # noqa: E402
from mokumitsu.joint_renewal import (  # noqa: E402
    JointDesignParameters,
    JointRenewalPolicy,
    joint_renewal_candidates,
)
from mokumitsu.organic import generate_organic_mokumitsu  # noqa: E402
from mokumitsu.wind import (  # noqa: E402
    SummerWindScenario,
    WindDirection,
    district_wind_base_masks,
)

_MODELS = {}
_BASELINES = {}
_RESPONSES = {}
_DISTRICTS = {}
_CANDIDATES = {}
_WIND_BASE_MASKS = {}


def _digest(data) -> str:
    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _block_mean(field: np.ndarray, target: int) -> np.ndarray:
    height, width = field.shape
    if height % target or width % target:
        rows = np.array_split(np.arange(height), target)
        columns = np.array_split(np.arange(width), target)
        return np.asarray(
            [[field[np.ix_(row, column)].mean() for column in columns] for row in rows]
        )
    return field.reshape(target, height // target, target, width // target).mean(axis=(1, 3))


def _model(request):
    key = (
        request["wind_model"],
        str(Path(request["checkpoint_dir"]).resolve()),
        request.get("device", "auto"),
    )
    model = _MODELS.get(key)
    if model is None:
        with redirect_stdout(sys.stderr):
            model = load_accelerated_model(
                key[0],
                key[1],
                device=key[2],
            )
        _MODELS[key] = model
    return model


def evaluate(request: dict) -> dict:
    request_key = _digest(request)
    cached = _RESPONSES.get(request_key)
    if cached is not None:
        return {**cached, "worker_cache_hit": True, "worker_ms": 0.0}

    start = time.perf_counter()
    previous = start
    timings = {}

    def mark(name):
        nonlocal previous
        now = time.perf_counter()
        timings[name] = 1000.0 * (now - previous)
        previous = now

    seed = int(request["seed"])
    district = _DISTRICTS.get(seed)
    if district is None:
        district = generate_organic_mokumitsu(seed=seed)
        _DISTRICTS[seed] = district
    mark("district")
    policy = JointRenewalPolicy(**request["policy"])
    cluster_size = int(request["cluster_size"])
    candidate_slot = (seed, policy, cluster_size)
    candidates = _CANDIDATES.get(candidate_slot)
    if candidates is None:
        candidates = tuple(
            candidate
            for candidate in joint_renewal_candidates(district, policy)
            if candidate.parcel_count == cluster_size
        )
        _CANDIDATES[candidate_slot] = candidates
    mark("candidates")
    if not candidates:
        raise ValueError("the requested seed and cluster size have no renewal candidate")
    rank = max(0, min(int(request["candidate_rank"]), len(candidates) - 1))
    candidate = candidates[rank]
    parameters = JointDesignParameters(**request["parameters"])
    wind_scenario = SummerWindScenario(
        directions=tuple(WindDirection(**item) for item in request["wind_directions"]),
        model_name=request["wind_model"],
    )
    fire_scenario = FireScenario(**request["fire_scenario"])
    model = _model(request)
    mark("model")
    baseline_slot = (
        district.seed,
        repr(wind_scenario),
        repr(fire_scenario),
        model.resolved_stem,
        model.provenance(include_hash=True)["sha256"],
    )
    baseline = _BASELINES.get(baseline_slot)
    if baseline is None:
        baseline = prepare_joint_design_baseline(
            district,
            wind_scenario,
            fire_scenario,
            model,
        )
        _BASELINES[baseline_slot] = baseline
    mark("baseline")
    wind_base_slot = (seed, policy, candidate.parcel_ids, int(model.ny))
    wind_base_masks = _WIND_BASE_MASKS.get(wind_base_slot)
    result = evaluate_joint_design(
        district,
        candidate,
        parameters,
        policy,
        wind_scenario,
        fire_scenario,
        model,
        baseline=baseline,
        wind_base_masks=wind_base_masks,
        include_wind_field=True,
    )
    if wind_base_masks is None:
        _WIND_BASE_MASKS[wind_base_slot] = district_wind_base_masks(
            result.district,
            int(model.ny),
        )
    mark("design")
    if result.wind_ratio is None or result.heightmap is None:
        raise RuntimeError("interactive evaluation did not return a wind field")
    ratio, heightmap = result.wind_ratio, result.heightmap
    mark("field")
    display_res = int(request.get("display_resolution", 20))
    ratio_display = _block_mean(np.asarray(ratio, dtype=float), display_res)
    solid_display = _block_mean((np.asarray(heightmap) > 1e-7).astype(float), display_res) > 0.25
    mark("display")
    response = {
        "ok": True,
        "cache_key": result.cache_key,
        "district": result.district.to_dict(),
        "placement": asdict(result.placement),
        "metrics": {
            "baseline_objectives": asdict(result.baseline.objectives),
            "objectives": asdict(result.objectives),
            "gross_floor_area_m2": result.gross_floor_area_m2,
            "selected_floor_area_retention": result.selected_floor_area_retention,
        },
        "wind_ratio": ratio_display.tolist(),
        "solid": solid_display.astype(int).tolist(),
        "display_resolution": display_res,
        "device": str(model.device),
        "worker_cache_hit": False,
        "worker_ms": 1000.0 * (time.perf_counter() - start),
        "timings_ms": timings,
    }
    _RESPONSES[request_key] = response
    return response


def main() -> None:
    protocol = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                break
            response = evaluate(request)
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        protocol.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
