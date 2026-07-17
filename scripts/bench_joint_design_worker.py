"""Measure cold, warm-edit and repeated-design latency of the persistent worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mokumitsu import (  # noqa: E402
    FireScenario,
    JointRenewalPolicy,
    SummerWindScenario,
    WindDirection,
    default_joint_design_parameters,
    generate_mokumitsu,
    joint_renewal_candidates,
)


def main() -> None:
    checkpoint_dir = os.environ.get("MOKUMITSU_CHECKPOINT_DIR")
    if not checkpoint_dir:
        raise RuntimeError("MOKUMITSU_CHECKPOINT_DIR is required")
    district = generate_mokumitsu(seed=0)
    policy = JointRenewalPolicy(
        candidate_limit=24,
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
        minimum_floors=3,
        maximum_floors=8,
        placement_grid=7,
        placement_variants=3,
    )
    candidate = next(
        candidate
        for candidate in joint_renewal_candidates(district, policy)
        if candidate.parcel_count == 3
    )
    parameters = default_joint_design_parameters(district, candidate, policy)
    wind = SummerWindScenario(
        directions=(WindDirection(0.0, 0.7), WindDirection(90.0, 0.3)),
        model_name="residential",
    )
    fire = FireScenario(runs=40, seed=0)
    base = {
        "seed": 0,
        "policy": asdict(policy),
        "cluster_size": 3,
        "candidate_rank": 0,
        "wind_model": wind.model_name,
        "wind_directions": [asdict(direction) for direction in wind.directions],
        "fire_scenario": asdict(fire),
        "checkpoint_dir": checkpoint_dir,
        "device": "auto",
        "display_resolution": 20,
    }
    environment = os.environ.copy()
    worker_tmp = ROOT / "outputs" / "worker_tmp"
    worker_tmp.mkdir(parents=True, exist_ok=True)
    environment["TMP"] = str(worker_tmp)
    environment["TEMP"] = str(worker_tmp)
    environment.setdefault("WANDB_MODE", "disabled")
    environment.setdefault("WANDB_SILENT", "true")
    process = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts" / "joint_design_worker.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for label, design in (
            ("cold", parameters),
            ("warm-edit", replace(parameters, rotation_deg=parameters.rotation_deg + 0.371)),
            ("repeat", parameters),
        ):
            request = {**base, "parameters": asdict(design)}
            start = time.perf_counter()
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited with code {process.poll()}")
            response = json.loads(line)
            wall_ms = 1000.0 * (time.perf_counter() - start)
            print(
                f"{label}: ok={response.get('ok')} device={response.get('device')} "
                f"worker_cache={response.get('worker_cache_hit')} "
                f"worker={response.get('worker_ms', 0.0):.1f}ms wall={wall_ms:.1f}ms "
                f"timings={response.get('timings_ms', {})}"
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("traceback", response.get("error")))
    finally:
        if process.poll() is None:
            process.stdin.write('{"command":"shutdown"}\n')
            process.stdin.flush()
            process.wait(timeout=5.0)


if __name__ == "__main__":
    main()
