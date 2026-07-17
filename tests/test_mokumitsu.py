from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mokumitsu import (  # noqa: E402
    ARTICLE_42_2,
    MokumitsuConfig,
    MokumitsuDistrict,
    generate_mokumitsu,
    generate_mokumitsu_grid,
    load_district,
    morphology_summary,
    renewal_priorities,
    road_adjacency,
    save_district,
    validate_district,
)


def test_research_package_has_no_houdini_or_windcfd_imports():
    package = Path(__file__).parents[1] / "src" / "mokumitsu"
    forbidden = {"hou", "windcfd"}
    imports = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    assert imports.isdisjoint(forbidden)


def test_public_entry_points_are_standalone_repository_relative():
    project = Path(__file__).parents[1]
    entry_points = (
        project / "scripts" / "evaluate_joint_feasibility.py",
        project / "houdini" / "build_mokumitsu_hip.py",
        project / "houdini" / "build_joint_feasibility_hip.py",
    )
    for path in entry_points:
        source = path.read_text(encoding="utf-8")
        assert "parents[3]" not in source
        assert "projects/mokumitsu" not in source
        assert 'sys.path.insert(0, str(ROOT / "src"))' not in source


def test_houdini_builder_drives_renewal_phases_from_timeline():
    builder = (Path(__file__).parents[1] / "houdini" / "build_mokumitsu_hip.py").read_text(
        encoding="utf-8"
    )
    assert 'phase_index = int(node.evalParm("active_phase"))' in builder
    assert '"Drive renewal from timeline"' in builder
    assert "clamp($F - 1, 0, 6)" in builder
    assert "hou.setFps(1.0)" in builder
    assert "hou.playbar.setFrameRange(1, 7)" in builder
    assert 'geo.createNode("filecache::2.0", "CACHE_TIMELINE")' in builder
    assert '"$HIP/cache/mokumitsu_timeline.$F4.bgeo.sc"' in builder
    assert 'file_cache.parm("trange").set("normal")' in builder
    assert 'file_cache.parm("execute").pressButton()' in builder
    assert 'file_cache.parm("loadfromdisk").set(1)' in builder
    assert 'geo.createNode("null", "OUT_MOKUMITSU")' in builder


def test_joint_feasibility_builder_is_a_separate_cached_timeline():
    builder = (Path(__file__).parents[1] / "houdini" / "build_joint_feasibility_hip.py").read_text(
        encoding="utf-8"
    )
    assert '"Drive phase from timeline"' in builder
    assert "clamp($F - 1, 0," in builder
    assert '"initial_district" not in payload' in builder
    assert 'geo.createNode("filecache::2.0", "CACHE_JOINT_TIMELINE")' in builder
    assert '"$HIP/cache/joint_feasibility_timeline.$F4.bgeo.sc"' in builder
    assert 'geo.createNode("font", name)' in builder
    assert '"PHASE_STATUS"' in builder
    assert '"CAM_JOINT_TIMELINE"' in builder
    assert '"ventilation_corridor": (0.0, 0.82, 1.0)' in builder
    assert '"pocket_park": (0.10, 0.72, 0.24)' in builder
    assert '"temporary_dwellings"' in builder
    assert '"cumulative_cost_million_jpy"' in builder
    assert '"wind_visualization"' in builder
    assert '"wind_cell"' in builder
    assert '"road_outline"' in builder
    assert 'geo.createNode("python", "WIND_DISPLAY_TOGGLE")' in builder
    assert '"Show cached FNO wind field"' in builder
    assert 'hidden_kinds = {"ground", "road"} if show_wind else' in builder
    assert 'file_cache.parm("execute").pressButton()' in builder
    assert 'geo.createNode("null", "OUT_JOINT_FEASIBILITY")' in builder


def test_generator_is_deterministic_and_structurally_valid():
    a = generate_mokumitsu(seed=7)
    b = generate_mokumitsu(seed=7)
    c = generate_mokumitsu(seed=8)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != c.to_dict()
    assert validate_district(a) == ()
    assert len(a.parcels) == len(a.buildings)
    assert len(a.buildings) >= 40


def test_default_generator_is_organic_not_a_jittered_grid():
    district = generate_mokumitsu(seed=0)
    morphology = morphology_summary(district)
    assert district.generator == "organic"
    assert morphology.non_axis_aligned_road_rate > 0.35
    assert morphology.road_orientation_entropy > 0.55
    assert morphology.dead_end_road_rate > 0.05
    assert morphology.irregular_parcel_rate > 0.80
    assert morphology.parcel_area_cv > 0.15
    assert morphology.flag_lot_rate > 0
    assert morphology.back_lot_rate > 0
    assert any(len(parcel.polygon) > 4 for parcel in district.parcels)
    assert np.std([building.theta for building in district.buildings]) > 0.20


def test_rectilinear_generator_is_only_an_explicit_baseline():
    organic = morphology_summary(generate_mokumitsu(seed=2))
    baseline_district = generate_mokumitsu_grid(seed=2)
    baseline = morphology_summary(baseline_district)
    assert baseline_district.generator == "grid"
    assert baseline.non_axis_aligned_road_rate == 0
    assert baseline.irregular_parcel_rate == 0
    assert organic.non_axis_aligned_road_rate > baseline.non_axis_aligned_road_rate
    assert organic.irregular_parcel_rate > baseline.irregular_parcel_rate


def test_net_coverage_control_changes_density_without_breaking_parcel_containment():
    base = MokumitsuConfig()
    low = generate_mokumitsu(seed=5, config=replace(base, target_net_building_coverage=0.36))
    high = generate_mokumitsu(seed=5, config=replace(base, target_net_building_coverage=0.62))
    assert high.summary().net_building_coverage > low.summary().net_building_coverage + 0.05
    assert validate_district(high) == ()


def test_access_model_contains_legal_narrow_and_landlocked_cases():
    district = generate_mokumitsu(seed=0)
    access = [district.access(p.id) for p in district.parcels]
    assert any(a.individual_rebuildable for a in access)
    assert any(not a.individual_rebuildable for a in access)
    assert any(a.path_frontage_m > 0 for a in access)
    assert any(r.legal_class == ARTICLE_42_2 for r in district.roads)
    assert any(a.setback_required_m > 0 for a in access)
    for item in access:
        assert item.individual_rebuildable == (item.max_continuous_legal_frontage_m >= 2.0)


def test_age_is_correlated_but_all_policy_cohorts_exist():
    district = generate_mokumitsu(seed=3)
    cohorts = {b.age_cohort for b in district.buildings}
    assert cohorts == {"pre_1981", "1981_2000", "post_2000"}
    summary = district.summary()
    assert 0.05 < summary.pre_1981_rate < 0.80
    assert 0.40 < summary.timber_rate < 1.0
    assert 0.25 < summary.net_building_coverage < 0.70
    assert 0.20 < summary.footprint_coverage < 0.55
    assert summary.gross_floor_area_ratio > summary.footprint_coverage
    assert summary.building_density_per_ha >= 55


def test_priority_keeps_need_and_feasibility_separate():
    district = generate_mokumitsu(seed=0)
    ranked = renewal_priorities(district)
    assert [c.rank for c in ranked] == list(range(1, len(ranked) + 1))
    assert all(
        a.priority_score >= b.priority_score for a, b in zip(ranked, ranked[1:], strict=False)
    )
    assert any(c.recommended_action == "early_individual_rebuild" for c in ranked)
    assert any(c.recommended_action == "joint_rebuild_or_access_improvement" for c in ranked)
    assert any(c.need_score > 0.6 and c.feasibility_score < 0.5 for c in ranked)


def test_json_roundtrip_and_wind_heightmap(tmp_path):
    district = generate_mokumitsu(seed=11)
    path = save_district(district, tmp_path / "district.json")
    loaded = load_district(path)
    assert isinstance(loaded, MokumitsuDistrict)
    assert loaded.to_dict() == district.to_dict()
    heightmap = loaded.heightmap(64)
    assert heightmap.shape == (64, 64)
    assert heightmap.dtype == np.float32
    assert heightmap.max() > 0


def test_road_graph_has_connected_network_and_dead_end_paths():
    district = generate_mokumitsu(seed=0)
    graph = road_adjacency(district)
    assert set(graph) == {r.id for r in district.roads}
    assert any(len(neighbours) >= 3 for neighbours in graph.values())
    paths = [r for r in district.roads if r.dead_end]
    assert paths
    assert all(len(graph[r.id]) >= 1 for r in paths)


def test_invalid_priority_weights_are_rejected():
    from mokumitsu import PriorityWeights

    district = generate_mokumitsu(seed=0)
    with pytest.raises(ValueError):
        renewal_priorities(
            district,
            PriorityWeights(
                age=0,
                structure=0,
                fire_resistance=0,
                access=0,
                spacing=0,
                parcel_coverage=0,
            ),
        )
