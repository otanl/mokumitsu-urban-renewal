"""Interactive joint-renewal evaluation and deterministic cache identities.

The geometry controls live in :mod:`mokumitsu.joint_renewal`; this module adds
the repeatable baseline/design comparison used by Houdini. FNO and graph-fire
results remain screening metrics and are not treated as validated predictions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .district import MokumitsuDistrict
from .fire import FireRiskResult, FireScenario, simulate_fire_spread
from .joint_renewal import (
    JointBuildingPlacement,
    JointDesignParameters,
    JointRenewalCandidate,
    JointRenewalPolicy,
    joint_wind_objective,
    parameterized_joint_redevelopment,
)
from .model import FnoModel, load_model
from .pareto import JointRenewalObjectives
from .wind import (
    DistrictWindMetrics,
    SummerWindScenario,
    evaluate_district_wind,
    evaluate_district_wind_with_field,
)

_CACHE_SCHEMA = "mokumitsu-joint-design-v1"


@dataclass(frozen=True)
class JointDesignBaseline:
    """Reusable baseline so interactive edits do not rerun unchanged analyses."""

    cache_key: str
    district_digest: str
    model_identity: dict[str, object]
    reference_speed: float
    wind: DistrictWindMetrics
    fire: FireRiskResult
    objectives: JointRenewalObjectives
    gross_floor_area_m2: float

    def to_dict(self) -> dict:
        return {
            "cache_key": self.cache_key,
            "district_digest": self.district_digest,
            "model": self.model_identity,
            "reference_speed": self.reference_speed,
            "objectives": asdict(self.objectives),
            "gross_floor_area_m2": self.gross_floor_area_m2,
            "wind": self.wind.to_dict(),
            "fire": _fire_summary(self.fire),
        }


@dataclass(frozen=True)
class JointDesignEvaluation:
    """Baseline-versus-design comparison for one editable joint project."""

    cache_key: str
    baseline: JointDesignBaseline
    parameters: JointDesignParameters
    policy: JointRenewalPolicy
    candidate: JointRenewalCandidate
    placement: JointBuildingPlacement
    district: MokumitsuDistrict
    wind: DistrictWindMetrics
    fire: FireRiskResult
    objectives: JointRenewalObjectives
    gross_floor_area_m2: float
    selected_floor_area_retention: float
    wind_ratio: np.ndarray | None = None
    heightmap: np.ndarray | None = None

    def to_dict(self, include_district: bool = False) -> dict:
        baseline_objectives = self.baseline.objectives
        data = {
            "schema": _CACHE_SCHEMA,
            "cache_key": self.cache_key,
            "model": self.baseline.model_identity,
            "parameters": asdict(self.parameters),
            "policy": asdict(self.policy),
            "candidate": asdict(self.candidate),
            "placement": asdict(self.placement),
            "baseline": self.baseline.to_dict(),
            "design": {
                "objectives": asdict(self.objectives),
                "gross_floor_area_m2": self.gross_floor_area_m2,
                "selected_floor_area_retention": self.selected_floor_area_retention,
                "wind": self.wind.to_dict(),
                "fire": _fire_summary(self.fire),
            },
            "delta": {
                "wind_penalty": self.objectives.wind_penalty - baseline_objectives.wind_penalty,
                "expected_burned_area_m2": self.objectives.expected_burned_area_m2
                - baseline_objectives.expected_burned_area_m2,
                "footprint_coverage": self.objectives.footprint_coverage
                - baseline_objectives.footprint_coverage,
                "gross_floor_area_m2": self.gross_floor_area_m2 - self.baseline.gross_floor_area_m2,
            },
        }
        if include_district:
            data["district"] = self.district.to_dict()
        return data


def prepare_joint_design_baseline(
    district: MokumitsuDistrict,
    wind_scenario: SummerWindScenario | None = None,
    fire_scenario: FireScenario | None = None,
    model: FnoModel | None = None,
    *,
    model_fingerprint: str | None = None,
) -> JointDesignBaseline:
    """Evaluate and identify the unchanged district side of an interactive study."""
    wind_scenario = wind_scenario or SummerWindScenario()
    fire_scenario = fire_scenario or FireScenario(runs=80, seed=district.seed)
    model = model or load_model(wind_scenario.model_name)
    identity = _model_identity(model, model_fingerprint)
    reference_speed = float(model.reference_speed())
    wind = evaluate_district_wind(
        district,
        wind_scenario,
        model,
        reference_speed=reference_speed,
    )
    fire = simulate_fire_spread(district, fire_scenario)
    summary = district.summary()
    objectives = JointRenewalObjectives(
        wind_penalty=joint_wind_objective(wind),
        expected_burned_area_m2=fire.expected_burned_area_m2,
        footprint_coverage=summary.footprint_coverage,
        rights_count=0,
        access_poor_resolved=0,
        connected_open_space_m2=0.0,
    )
    district_digest = _digest(district.to_dict())
    cache_key = _baseline_cache_key(
        district,
        wind_scenario,
        fire_scenario,
        identity,
    )
    return JointDesignBaseline(
        cache_key=cache_key,
        district_digest=district_digest,
        model_identity=identity,
        reference_speed=reference_speed,
        wind=wind,
        fire=fire,
        objectives=objectives,
        gross_floor_area_m2=summary.gross_floor_area_m2,
    )


def joint_design_cache_key(
    district: MokumitsuDistrict,
    candidate: JointRenewalCandidate,
    parameters: JointDesignParameters,
    policy: JointRenewalPolicy,
    wind_scenario: SummerWindScenario,
    fire_scenario: FireScenario,
    model: FnoModel,
    *,
    model_fingerprint: str | None = None,
) -> str:
    """Return the content key used by Houdini's automatic geometry cache."""
    identity = _model_identity(model, model_fingerprint)
    baseline_key = _baseline_cache_key(
        district,
        wind_scenario,
        fire_scenario,
        identity,
    )
    return _design_cache_key(baseline_key, candidate, parameters, policy)


def evaluate_joint_design(
    district: MokumitsuDistrict,
    candidate: JointRenewalCandidate,
    parameters: JointDesignParameters,
    policy: JointRenewalPolicy | None = None,
    wind_scenario: SummerWindScenario | None = None,
    fire_scenario: FireScenario | None = None,
    model: FnoModel | None = None,
    *,
    baseline: JointDesignBaseline | None = None,
    model_fingerprint: str | None = None,
    wind_base_masks: dict[str, np.ndarray] | None = None,
    include_wind_field: bool = False,
) -> JointDesignEvaluation:
    """Evaluate one live massing edit against a reusable baseline."""
    policy = policy or JointRenewalPolicy(
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
    )
    wind_scenario = wind_scenario or SummerWindScenario()
    fire_scenario = fire_scenario or FireScenario(runs=80, seed=district.seed)
    model = model or load_model(wind_scenario.model_name)
    identity = _model_identity(model, model_fingerprint)
    expected_baseline_key = _baseline_cache_key(
        district,
        wind_scenario,
        fire_scenario,
        identity,
    )
    if baseline is None:
        baseline = prepare_joint_design_baseline(
            district,
            wind_scenario,
            fire_scenario,
            model,
            model_fingerprint=model_fingerprint,
        )
    elif baseline.cache_key != expected_baseline_key:
        raise ValueError("baseline does not match the district, scenarios, or model")

    updated, placement = parameterized_joint_redevelopment(
        district,
        candidate,
        parameters,
        policy,
        wind_direction_deg=wind_scenario.directions[0].direction_deg,
    )
    if include_wind_field:
        wind, wind_ratio, heightmap = evaluate_district_wind_with_field(
            updated,
            wind_scenario,
            model,
            reference_speed=baseline.reference_speed,
            base_masks=wind_base_masks,
        )
    else:
        wind = evaluate_district_wind(
            updated,
            wind_scenario,
            model,
            reference_speed=baseline.reference_speed,
            base_masks=wind_base_masks,
        )
        wind_ratio = None
        heightmap = None
    fire = simulate_fire_spread(updated, fire_scenario)
    summary = updated.summary()
    access_resolved = sum(
        not district.access(parcel_id).individual_rebuildable for parcel_id in candidate.parcel_ids
    )
    objectives = JointRenewalObjectives(
        wind_penalty=joint_wind_objective(wind),
        expected_burned_area_m2=fire.expected_burned_area_m2,
        footprint_coverage=summary.footprint_coverage,
        rights_count=candidate.parcel_count,
        access_poor_resolved=int(access_resolved),
        connected_open_space_m2=placement.connected_open_space_m2,
    )
    retention = placement.floor_area_m2 / max(candidate.old_floor_area_m2, 1e-9)
    cache_key = _design_cache_key(baseline.cache_key, candidate, parameters, policy)
    return JointDesignEvaluation(
        cache_key=cache_key,
        baseline=baseline,
        parameters=parameters,
        policy=policy,
        candidate=candidate,
        placement=placement,
        district=updated,
        wind=wind,
        fire=fire,
        objectives=objectives,
        gross_floor_area_m2=summary.gross_floor_area_m2,
        selected_floor_area_retention=float(retention),
        wind_ratio=wind_ratio,
        heightmap=heightmap,
    )


def _baseline_cache_key(
    district: MokumitsuDistrict,
    wind_scenario: SummerWindScenario,
    fire_scenario: FireScenario,
    model_identity: dict[str, object],
) -> str:
    return _digest(
        {
            "schema": _CACHE_SCHEMA,
            "kind": "baseline",
            "district": district.to_dict(),
            "wind_scenario": asdict(wind_scenario),
            "fire_scenario": asdict(fire_scenario),
            "model": model_identity,
        }
    )


def _design_cache_key(
    baseline_key: str,
    candidate: JointRenewalCandidate,
    parameters: JointDesignParameters,
    policy: JointRenewalPolicy,
) -> str:
    return _digest(
        {
            "schema": _CACHE_SCHEMA,
            "kind": "design",
            "baseline_key": baseline_key,
            "candidate": asdict(candidate),
            "parameters": asdict(parameters),
            "policy": asdict(policy),
        }
    )


def _model_identity(model: object, fingerprint: str | None) -> dict[str, object]:
    grid = [int(model.ny), int(model.nx)]
    if fingerprint is not None:
        return {"fingerprint": str(fingerprint), "grid": grid}
    provenance = getattr(model, "provenance", None)
    if callable(provenance):
        data = provenance(include_hash=True)
        return {
            "requested_name": data.get("requested_name"),
            "resolved_stem": data.get("resolved_stem"),
            "sha256": data.get("sha256"),
            "grid": data.get("grid", grid),
        }
    return {
        "implementation": f"{type(model).__module__}.{type(model).__qualname__}",
        "grid": grid,
    }


def _digest(data: object) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fire_summary(result: FireRiskResult) -> dict:
    return {
        "scenario": asdict(result.scenario),
        "expected_burned_buildings": result.expected_burned_buildings,
        "p95_burned_buildings": result.p95_burned_buildings,
        "probability_large_fire": result.probability_large_fire,
        "expected_burned_area_m2": result.expected_burned_area_m2,
    }
