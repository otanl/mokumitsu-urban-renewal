"""Download release-hosted Mokumitsu assets with SHA-256 and status checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_BYTES = 1024 * 1024


def load_manifest(path: Path) -> dict:
    """Load and minimally validate the public asset manifest."""
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    if not str(manifest.get("download_base_url", "")).startswith("https://"):
        raise ValueError("manifest download_base_url must use HTTPS")
    status = manifest.get("status", "active")
    if status not in {"active", "quarantined"}:
        raise ValueError(f"unsupported manifest status: {status!r}")
    if status == "quarantined" and not str(manifest.get("status_reason", "")).strip():
        raise ValueError("quarantined manifest must explain status_reason")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("manifest must contain at least one asset")
    filenames: set[str] = set()
    for asset in assets:
        filename = str(asset.get("filename", ""))
        if not filename or Path(filename).name != filename or filename in filenames:
            raise ValueError(f"invalid or duplicate asset filename: {filename!r}")
        filenames.add(filename)
        if asset.get("destination") not in {"checkpoints", "data"}:
            raise ValueError(f"invalid destination for {filename}")
        if not isinstance(asset.get("bytes"), int) or asset["bytes"] <= 0:
            raise ValueError(f"invalid byte count for {filename}")
        if not _SHA256.fullmatch(str(asset.get("sha256", ""))):
            raise ValueError(f"invalid SHA-256 for {filename}")
        profiles = asset.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            raise ValueError(f"missing profiles for {filename}")
    return manifest


def file_sha256(path: Path) -> str:
    """Hash a local file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_assets(manifest: dict, profile: str, include_dataset: bool) -> list[dict]:
    """Select portable/accelerated checkpoints and the optional dataset."""
    selected = []
    for asset in manifest["assets"]:
        profiles = set(asset["profiles"])
        if "dataset" in profiles:
            if include_dataset:
                selected.append(asset)
        elif profile == "all" or profile in profiles:
            selected.append(asset)
    return selected


def target_path(
    asset: dict,
    checkpoint_dir: Path,
    data_dir: Path,
) -> Path:
    base = checkpoint_dir if asset["destination"] == "checkpoints" else data_dir
    return base / asset["filename"]


def verify_existing(path: Path, asset: dict) -> bool:
    """Return True only when an existing asset matches size and digest."""
    return (
        path.is_file()
        and path.stat().st_size == asset["bytes"]
        and file_sha256(path) == asset["sha256"]
    )


def download_asset(url: str, target: Path, asset: dict, *, force: bool = False) -> None:
    """Download one asset atomically and reject size or digest mismatches."""
    if target.exists():
        if verify_existing(target, asset):
            print(f"verified {target}")
            return
        if not force:
            raise RuntimeError(
                f"{target} exists but does not match the manifest; use --force to replace it"
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    written = 0
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mokumitsu-model-downloader/1"},
    )
    print(f"downloading {asset['filename']} ({asset['bytes'] / 1_000_000:.1f} MB)")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as stream:
            while chunk := response.read(_CHUNK_BYTES):
                stream.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != asset["bytes"]:
            raise RuntimeError(
                f"size mismatch for {asset['filename']}: expected {asset['bytes']}, got {written}"
            )
        actual_hash = digest.hexdigest()
        if actual_hash != asset["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {asset['filename']}: "
                f"expected {asset['sha256']}, got {actual_hash}"
            )
        os.replace(partial, target)
        print(f"saved {target}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("all", "portable", "accelerated"),
        default="all",
        help="checkpoint runtime to download; all is the default",
    )
    parser.add_argument(
        "--include-dataset",
        action="store_true",
        help="also download the 500-sample XLB training dataset",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="default: MOKUMITSU_CHECKPOINT_DIR or <repo>/checkpoints",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-quarantined",
        action="store_true",
        help="download invalid historical assets for reproduction/audit only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected URLs and targets without downloading",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest.resolve())
    if manifest.get("status", "active") == "quarantined" and not args.allow_quarantined:
        raise RuntimeError(
            "this model release is quarantined and cannot be used for design evaluation: "
            f"{manifest['status_reason']} Use --allow-quarantined only for historical "
            "reproduction or audit."
        )
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        configured = os.environ.get("MOKUMITSU_CHECKPOINT_DIR", "").strip()
        checkpoint_dir = Path(configured) if configured else ROOT / "checkpoints"
    checkpoint_dir = checkpoint_dir.resolve()
    data_dir = args.data_dir.resolve()
    base_url = manifest["download_base_url"].rstrip("/")
    assets = selected_assets(manifest, args.profile, args.include_dataset)
    for asset in assets:
        url = f"{base_url}/{asset['filename']}"
        target = target_path(asset, checkpoint_dir, data_dir)
        if args.dry_run:
            print(f"{url} -> {target}")
        else:
            download_asset(url, target, asset, force=args.force)


if __name__ == "__main__":
    main()
