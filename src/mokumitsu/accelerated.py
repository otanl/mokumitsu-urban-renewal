"""Optional CUDA-capable loader for training checkpoints used by the live worker.

The portable Houdini path remains TorchScript-on-CPU. This adapter loads the
corresponding NeuralOperator training checkpoint in an external project Python
process, where CUDA is available, without making Houdini itself depend on
NeuralOperator or a CUDA PyTorch build.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

_REGISTRY = {
    "lowrise": "fno_lowrise_xlb",
    "block": "fno_block_xlb",
    "residential": "fno_residential_xlb",
    "district": "fno_districtv2_xlb",
    "massing": "fno_massing_xlb",
    "city": "fno_city_xlb",
}


class AcceleratedFnoModel:
    """FNO training checkpoint with the same prediction contract as ``FnoModel``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        requested_name: str,
        resolved_stem: str,
        device: str = "auto",
    ):
        from neuralop.models import FNO

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.requested_name = requested_name
        self.resolved_stem = resolved_stem
        resolved_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        self.device = torch.device(resolved_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.meta = {
            key: checkpoint[key]
            for key in ("xmean", "xstd", "ymean", "ystd", "ny", "nx", "sharpness")
            if key in checkpoint
        }
        self.meta.setdefault("xmean", 0.0)
        self.meta.setdefault("xstd", 1.0)
        self.ny = int(checkpoint["ny"])
        self.nx = int(checkpoint["nx"])
        self.model = FNO(**checkpoint["fno_cfg"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self._sha256: str | None = None
        self._reference_speed: float | None = None

    def predict(self, heightmap: np.ndarray) -> np.ndarray:
        heightmap_tensor = torch.as_tensor(
            np.ascontiguousarray(heightmap, dtype=np.float32),
            device=self.device,
        )
        normalized = (heightmap_tensor - self.meta["xmean"]) / self.meta["xstd"]
        with torch.no_grad():
            speed = (self.model(normalized[None, None]) * self.meta["ystd"] + self.meta["ymean"])[
                0, 0
            ]
        return speed.detach().cpu().numpy()

    def predict_torch(self, heightmap: torch.Tensor) -> torch.Tensor:
        heightmap = heightmap.to(self.device)
        normalized = (heightmap - self.meta["xmean"]) / self.meta["xstd"]
        return (self.model(normalized[None, None]) * self.meta["ystd"] + self.meta["ymean"])[0, 0]

    def reference_speed(self) -> float:
        if self._reference_speed is None:
            self._reference_speed = float(
                self.predict(np.zeros((self.ny, self.nx), dtype=np.float32)).mean()
            )
        return self._reference_speed

    def provenance(self, *, include_hash: bool = False) -> dict[str, object]:
        if include_hash and self._sha256 is None:
            digest = hashlib.sha256()
            with self.checkpoint_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._sha256 = digest.hexdigest()
        return {
            "requested_name": self.requested_name,
            "resolved_stem": self.resolved_stem,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_bytes": self.checkpoint_path.stat().st_size,
            "sha256": self._sha256 if include_hash else None,
            "grid": [self.ny, self.nx],
            "device": str(self.device),
        }


def load_accelerated_model(
    name: str = "residential",
    checkpoint_dir: str | Path | None = None,
    *,
    device: str = "auto",
) -> AcceleratedFnoModel:
    """Strictly resolve one NeuralOperator checkpoint; no cross-family fallback."""
    checkpoint_dir = Path(checkpoint_dir or Path.cwd() / "checkpoints")
    stem = _REGISTRY.get(name, name)
    path = checkpoint_dir / f"{stem}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no accelerated checkpoint for model '{name}' at {path}")
    return AcceleratedFnoModel(
        path,
        requested_name=name,
        resolved_stem=stem,
        device=device,
    )
