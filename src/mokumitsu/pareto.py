"""Pareto screening for connected two-to-four-parcel mokumitsu renewal projects.

Each alternative is a single implementable joint project on the same baseline district.  The
geometry layer keeps gross floor area exactly, while the decision layer compares summer wind,
screening-level fire consequence, released ground footprint, road-connected open space, rights
burden, and the number of access-poor parcels resolved.  FNO and the graph fire model are
shortlist tools; Pareto winners still require XLB and a validated urban-fire model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from .district import MokumitsuDistrict
from .fire import FireRiskResult, FireScenario, simulate_fire_spread
from .joint_renewal import (
    JointBuildingPlacement,
    JointRenewalCandidate,
    JointRenewalPolicy,
    joint_redevelopment_variants,
    joint_renewal_candidates,
    joint_wind_objective,
)
from .model import FnoModel, load_model
from .wind import (
    DistrictWindMetrics,
    SummerWindScenario,
    evaluate_district_wind,
)


@dataclass(frozen=True)
class ParetoRenewalPolicy:
    """Sampling and compromise-score settings; none of the weights changes Pareto membership."""

    candidates_per_cluster_size: int = 4
    wind_weight: float = 0.30
    fire_weight: float = 0.30
    footprint_weight: float = 0.15
    open_space_weight: float = 0.15
    rights_weight: float = 0.10
    access_weight: float = 0.15


@dataclass(frozen=True)
class JointRenewalObjectives:
    wind_penalty: float
    expected_burned_area_m2: float
    footprint_coverage: float
    rights_count: int
    access_poor_resolved: int
    connected_open_space_m2: float = 0.0

    def minimization_vector(self) -> tuple[float, ...]:
        return (
            self.wind_penalty,
            self.expected_burned_area_m2,
            self.footprint_coverage,
            -self.connected_open_space_m2,
            float(self.rights_count),
            -float(self.access_poor_resolved),
        )


@dataclass(frozen=True)
class JointRenewalAlternative:
    index: int
    candidate: JointRenewalCandidate
    placement: JointBuildingPlacement
    district: MokumitsuDistrict
    wind: DistrictWindMetrics
    fire: FireRiskResult
    objectives: JointRenewalObjectives
    floor_area_retention: float
    balanced_score: float = math.nan

    @property
    def id(self) -> str:
        return f"{self.placement.joint_parcel_id}:{self.placement.variant}"


@dataclass(frozen=True)
class ParetoRenewalResult:
    policy: ParetoRenewalPolicy
    joint_policy: JointRenewalPolicy
    baseline_district: MokumitsuDistrict
    baseline_wind: DistrictWindMetrics
    baseline_fire: FireRiskResult
    baseline_objectives: JointRenewalObjectives
    alternatives: tuple[JointRenewalAlternative, ...]
    pareto_indices: tuple[int, ...]
    recommended_index: int | None

    @property
    def recommended(self) -> JointRenewalAlternative | None:
        if self.recommended_index is None:
            return None
        return self.alternatives[self.recommended_index]

    def to_dict(self, include_recommended_district: bool = False) -> dict:
        front = set(self.pareto_indices)
        alternatives = []
        for alternative in self.alternatives:
            data = {
                "index": alternative.index,
                "id": alternative.id,
                "candidate": asdict(alternative.candidate),
                "placement": asdict(alternative.placement),
                "objectives": asdict(alternative.objectives),
                "floor_area_retention": alternative.floor_area_retention,
                "balanced_score": alternative.balanced_score,
                "pareto": alternative.index in front,
                "wind": alternative.wind.to_dict(),
                "fire": _fire_summary(alternative.fire),
                "summary": asdict(alternative.district.summary()),
            }
            if include_recommended_district and alternative.index == self.recommended_index:
                data["district"] = alternative.district.to_dict()
            alternatives.append(data)
        return {
            "policy": asdict(self.policy),
            "joint_policy": asdict(self.joint_policy),
            "baseline": {
                "objectives": asdict(self.baseline_objectives),
                "wind": self.baseline_wind.to_dict(),
                "fire": _fire_summary(self.baseline_fire),
                "summary": asdict(self.baseline_district.summary()),
            },
            "alternatives": alternatives,
            "pareto_indices": self.pareto_indices,
            "recommended_index": self.recommended_index,
        }


def evaluate_joint_renewal_pareto(
    district: MokumitsuDistrict,
    joint_policy: JointRenewalPolicy | None = None,
    pareto_policy: ParetoRenewalPolicy | None = None,
    wind_scenario: SummerWindScenario | None = None,
    fire_scenario: FireScenario | None = None,
    model: FnoModel | None = None,
) -> ParetoRenewalResult:
    """Evaluate stratified 2/3/4-parcel alternatives and return their non-dominated front."""
    joint_policy = joint_policy or JointRenewalPolicy(
        minimum_cluster_parcels=2,
        maximum_cluster_parcels=4,
    )
    pareto_policy = pareto_policy or ParetoRenewalPolicy()
    wind_scenario = wind_scenario or SummerWindScenario()
    fire_scenario = fire_scenario or FireScenario(runs=80, seed=district.seed)
    _validate_pareto_policy(pareto_policy)
    model = model or load_model(wind_scenario.model_name)
    reference_speed = float(model.reference_speed())

    baseline_wind = evaluate_district_wind(
        district,
        scenario=wind_scenario,
        model=model,
        reference_speed=reference_speed,
    )
    baseline_fire = simulate_fire_spread(district, fire_scenario)
    baseline_objectives = JointRenewalObjectives(
        wind_penalty=joint_wind_objective(baseline_wind),
        expected_burned_area_m2=baseline_fire.expected_burned_area_m2,
        footprint_coverage=district.summary().footprint_coverage,
        rights_count=0,
        access_poor_resolved=0,
        connected_open_space_m2=0.0,
    )
    initial_floor_area = district.summary().gross_floor_area_m2

    candidates = joint_renewal_candidates(district, joint_policy)
    shortlisted = []
    for parcel_count in range(
        joint_policy.minimum_cluster_parcels,
        joint_policy.maximum_cluster_parcels + 1,
    ):
        shortlisted.extend(
            candidate for candidate in candidates if candidate.parcel_count == parcel_count
        )
        keep = sum(candidate.parcel_count == parcel_count for candidate in shortlisted)
        if keep > pareto_policy.candidates_per_cluster_size:
            del shortlisted[-(keep - pareto_policy.candidates_per_cluster_size) :]

    alternatives = []
    for candidate in shortlisted:
        access_resolved = sum(
            not district.access(parcel_id).individual_rebuildable
            for parcel_id in candidate.parcel_ids
        )
        variants = joint_redevelopment_variants(
            district,
            candidate,
            joint_policy,
            wind_direction_deg=wind_scenario.directions[0].direction_deg,
        )
        for variant_district, placement in variants:
            wind = evaluate_district_wind(
                variant_district,
                scenario=wind_scenario,
                model=model,
                reference_speed=reference_speed,
            )
            fire = simulate_fire_spread(variant_district, fire_scenario)
            summary = variant_district.summary()
            retention = summary.gross_floor_area_m2 / max(initial_floor_area, 1e-9)
            if not math.isclose(retention, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError("joint alternative failed the exact floor-area constraint")
            alternatives.append(
                JointRenewalAlternative(
                    index=len(alternatives),
                    candidate=candidate,
                    placement=placement,
                    district=variant_district,
                    wind=wind,
                    fire=fire,
                    objectives=JointRenewalObjectives(
                        wind_penalty=joint_wind_objective(wind),
                        expected_burned_area_m2=fire.expected_burned_area_m2,
                        footprint_coverage=summary.footprint_coverage,
                        rights_count=candidate.parcel_count,
                        access_poor_resolved=int(access_resolved),
                        connected_open_space_m2=placement.connected_open_space_m2,
                    ),
                    floor_area_retention=retention,
                )
            )

    scored = _add_balanced_scores(tuple(alternatives), pareto_policy)
    front = pareto_front(tuple(alternative.objectives for alternative in scored))
    recommended = (
        min(front, key=lambda index: (scored[index].balanced_score, scored[index].id))
        if front
        else None
    )
    return ParetoRenewalResult(
        policy=pareto_policy,
        joint_policy=joint_policy,
        baseline_district=district,
        baseline_wind=baseline_wind,
        baseline_fire=baseline_fire,
        baseline_objectives=baseline_objectives,
        alternatives=scored,
        pareto_indices=front,
        recommended_index=recommended,
    )


def pareto_front(objectives: tuple[JointRenewalObjectives, ...]) -> tuple[int, ...]:
    """Indices not dominated under all-minimization convention."""
    vectors = [np.asarray(objective.minimization_vector(), dtype=float) for objective in objectives]
    if any(not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("Pareto objectives must be finite")
    front = []
    for index, vector in enumerate(vectors):
        dominated = any(
            other_index != index and np.all(other <= vector) and np.any(other < vector)
            for other_index, other in enumerate(vectors)
        )
        if not dominated:
            front.append(index)
    return tuple(front)


def _add_balanced_scores(
    alternatives: tuple[JointRenewalAlternative, ...],
    policy: ParetoRenewalPolicy,
) -> tuple[JointRenewalAlternative, ...]:
    if not alternatives:
        return ()
    values = np.asarray(
        [alternative.objectives.minimization_vector() for alternative in alternatives],
        dtype=float,
    )
    spans = np.ptp(values, axis=0)
    normalized = (values - values.min(axis=0)) / np.where(spans > 1e-12, spans, 1.0)
    weights = np.asarray(
        (
            policy.wind_weight,
            policy.fire_weight,
            policy.footprint_weight,
            policy.open_space_weight,
            policy.rights_weight,
            policy.access_weight,
        ),
        dtype=float,
    )
    weights /= weights.sum()
    scores = normalized @ weights
    return tuple(
        replace(alternative, balanced_score=float(scores[index]))
        for index, alternative in enumerate(alternatives)
    )


def _fire_summary(result: FireRiskResult) -> dict:
    return {
        "scenario": asdict(result.scenario),
        "expected_burned_buildings": result.expected_burned_buildings,
        "p95_burned_buildings": result.p95_burned_buildings,
        "probability_large_fire": result.probability_large_fire,
        "expected_burned_area_m2": result.expected_burned_area_m2,
    }


def _validate_pareto_policy(policy: ParetoRenewalPolicy) -> None:
    if policy.candidates_per_cluster_size <= 0:
        raise ValueError("candidates_per_cluster_size must be positive")
    weights = (
        policy.wind_weight,
        policy.fire_weight,
        policy.footprint_weight,
        policy.open_space_weight,
        policy.rights_weight,
        policy.access_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Pareto compromise weights must be finite, non-negative, and non-zero")
