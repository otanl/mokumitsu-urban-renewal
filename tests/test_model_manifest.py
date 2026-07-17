import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "models" / "manifest.json"


def test_model_manifest_has_unique_valid_assets():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["release_tag"] == "models-residential-v1"
    assert manifest["license"] == "MIT"
    assert manifest["training"]["samples"] == 500
    assert manifest["training"]["grid"] == [128, 128]

    assets = manifest["assets"]
    filenames = [asset["filename"] for asset in assets]
    assert len(filenames) == len(set(filenames))
    assert set(filenames) == {
        "fno_residential_ts.pt",
        "fno_residential_ts.json",
        "fno_residential_xlb.pt",
        "residential_xlb.npz",
    }
    for asset in assets:
        assert asset["bytes"] > 0
        assert len(asset["sha256"]) == 64
        int(asset["sha256"], 16)
        assert Path(asset["filename"]).name == asset["filename"]


def test_downloader_dry_run_selects_profiles(tmp_path):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "download_models.py"),
        "--profile",
        "portable",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--data-dir",
        str(tmp_path / "data"),
        "--dry-run",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "fno_residential_ts.pt" in result.stdout
    assert "fno_residential_ts.json" in result.stdout
    assert "fno_residential_xlb.pt" not in result.stdout
    assert "residential_xlb.npz" not in result.stdout

    result = subprocess.run(
        [*command, "--include-dataset"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "residential_xlb.npz" in result.stdout


def test_training_and_export_commands_have_help():
    for script in ("train_residential_fno.py", "export_residential_torchscript.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout
