"""The surrogate wind model -- a TorchScript FNO that maps a building height-map to the pedestrian
wind-speed field in a few milliseconds.

TorchScript + a small JSON of normalization stats, so it loads with torch alone (no `neuralop`)
inside Houdini. ``predict`` is the fast forward; ``predict_torch`` keeps the graph so the optimizer
can backprop a wind loss through the model. A small registry resolves named models strictly;
cross-family fallback is available only when a caller requests it explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

# Name -> exact checkpoint followed by explicitly permitted compatibility candidates.
_REGISTRY = {
    "lowrise": ["fno_lowrise_ts", "fno_massing_ts", "fno_city_ts"],
    "block": ["fno_block_ts", "fno_lowrise_ts", "fno_massing_ts"],  # denser, more buildings
    "residential": [
        "fno_residential_ts",
        "fno_districtv2_ts",
        "fno_block_ts",
        "fno_lowrise_ts",
    ],
    "district": ["fno_districtv2_ts", "fno_block_ts", "fno_lowrise_ts"],
    "massing": ["fno_massing_ts", "fno_cluster_ts", "fno_city_ts"],
    "city": ["fno_city_ts"],
}


class FnoModel:
    """A loaded surrogate: ``predict`` (fast, numpy) and ``predict_torch`` (differentiable)."""

    def __init__(
        self,
        ts_path: str | Path,
        meta_path: str | Path,
        *,
        requested_name: str | None = None,
        resolved_stem: str | None = None,
    ):
        self.ts_path = Path(ts_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.requested_name = requested_name
        self.resolved_stem = resolved_stem or self.ts_path.stem
        self._sha256: str | None = None
        self.model = torch.jit.load(str(self.ts_path))
        self.model.eval()
        with self.meta_path.open(encoding="utf-8") as f:
            self.meta = json.load(f)
        self.ny, self.nx = int(self.meta["ny"]), int(self.meta["nx"])

    def predict(self, heightmap: np.ndarray) -> np.ndarray:
        hm = torch.from_numpy(np.ascontiguousarray(heightmap, dtype=np.float32))
        x = (hm - self.meta["xmean"]) / self.meta["xstd"]
        with torch.no_grad():
            sp = (self.model(x[None, None]) * self.meta["ystd"] + self.meta["ymean"])[0, 0]
        return sp.numpy()

    def predict_torch(self, heightmap: torch.Tensor) -> torch.Tensor:
        """Differentiable forward (NO no_grad) -- backprop a wind loss to a soft height-map."""
        x = (heightmap - self.meta["xmean"]) / self.meta["xstd"]
        return (self.model(x[None, None]) * self.meta["ystd"] + self.meta["ymean"])[0, 0]

    def reference_speed(self) -> float:
        """u0 -- the mean pedestrian speed of the empty domain (the comfort reference)."""
        return float(self.predict(np.zeros((self.ny, self.nx), np.float32)).mean())

    def provenance(self, *, include_hash: bool = False) -> dict[str, object]:
        """Describe the resolved model so research outputs can be reproduced."""
        if include_hash and self._sha256 is None:
            digest = hashlib.sha256()
            with self.ts_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._sha256 = digest.hexdigest()
        return {
            "requested_name": self.requested_name,
            "resolved_stem": self.resolved_stem,
            "torchscript_path": str(self.ts_path),
            "metadata_path": str(self.meta_path),
            "torchscript_bytes": self.ts_path.stat().st_size,
            "sha256": self._sha256 if include_hash else None,
            "grid": [self.ny, self.nx],
            "physics": self.meta.get("physics"),
            "dataset_sha256": self.meta.get("dataset_sha256"),
            "split": self.meta.get("split"),
        }


def load_model(
    name: str = "lowrise",
    ckpt_dir: str | Path | None = None,
    *,
    allow_fallback: bool = False,
) -> FnoModel:
    """Resolve a named model; cross-family fallback must be explicitly enabled."""
    if ckpt_dir is None:
        ckpt_dir = os.environ.get("MOKUMITSU_CHECKPOINT_DIR", Path.cwd() / "checkpoints")
    ckpt_dir = Path(ckpt_dir)
    registered = _REGISTRY.get(name, [name])
    candidates = registered if allow_fallback else registered[:1]
    for stem in candidates:
        ts = ckpt_dir / f"{stem}.pt"
        meta = ckpt_dir / f"{stem}.json"
        if ts.exists() and meta.exists():
            return FnoModel(
                ts,
                meta,
                requested_name=name,
                resolved_stem=stem,
            )
    raise FileNotFoundError(
        f"no checkpoint for model '{name}' in {ckpt_dir} "
        f"(tried {candidates}; allow_fallback={allow_fallback})"
    )
