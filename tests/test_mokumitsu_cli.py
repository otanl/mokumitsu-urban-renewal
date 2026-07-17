from __future__ import annotations

import json

import numpy as np

from mokumitsu.cli import main


class _FakeModel:
    ny = 32
    nx = 32

    def reference_speed(self):
        return 1.0

    def predict(self, heightmap):
        speed = np.broadcast_to(
            np.linspace(0.15, 1.35, self.nx, dtype=np.float32),
            (self.ny, self.nx),
        ).copy()
        speed[heightmap > 0] = 0.0
        return speed

    def provenance(self, include_hash=False):
        return {"family": "fake", "hash_included": include_hash}


def test_generate_and_prioritize_cli(tmp_path):
    district = tmp_path / "district.json"
    priorities = tmp_path / "priorities.json"

    assert main(["generate", "--seed", "4", "--out", str(district)]) == 0
    assert district.exists()
    assert (
        main(
            [
                "prioritize",
                "--district",
                str(district),
                "--limit",
                "3",
                "--out",
                str(priorities),
            ]
        )
        == 0
    )

    payload = json.loads(priorities.read_text(encoding="utf-8"))
    assert payload["district"]["generator"] == "organic"
    assert len(payload["priorities"]) == 3
    assert [item["rank"] for item in payload["priorities"]] == [1, 2, 3]


def test_feasibility_cli_serializes_delivery_assumptions(
    tmp_path,
    monkeypatch,
):
    district = tmp_path / "district.json"
    output = tmp_path / "feasibility.json"
    assert main(["generate", "--seed", "0", "--out", str(district)]) == 0
    monkeypatch.setattr("mokumitsu.cli.load_model", lambda *args, **kwargs: _FakeModel())

    assert (
        main(
            [
                "feasibility",
                "--district",
                str(district),
                "--projects",
                "1",
                "--candidate-limit",
                "3",
                "--maximum-parcels",
                "2",
                "--placement-grid",
                "5",
                "--placement-variants",
                "2",
                "--max-wind-objective-increase",
                "1.0",
                "--wind-rose",
                "0:1",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["projects"] == 1
    assert payload["summary"]["rights_converted"] == 2
    assert payload["summary"]["total_cost_jpy"] > 0
    assert payload["policy"]["construction_months"] == 18
    assert payload["policy"]["replacement_dwelling_area_m2"] == 55.0
    assert payload["model"]["family"] == "fake"
