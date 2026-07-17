"""Pedestrian wind-comfort metrics -- the shared vocabulary the whole engine optimizes and reports.

Wind comfort is a BAND, not "less is better": below ``u_dead`` the air is stagnant (no through-
ventilation -- 通風), above ``u_strong`` it is blustery / building-wind (ビル風). Comfortable is in
between. All thresholds are fractions of the empty-domain reference speed ``u0`` so they transfer
across sites and between the FNO and XLB. numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    """Comfort-band edges as fractions of u0."""

    dead: float = 0.30  # below this = stagnant (no ventilation)
    strong: float = 1.30  # above this = blustery / building wind

    def band(self) -> tuple:
        return (self.dead, self.strong)


@dataclass(frozen=True)
class Comfort:
    """Comfort summary over a region (the open, pedestrian area)."""

    u0: float
    mean_index: float  # mean U / u0
    dead_frac: float  # share below u_dead (stagnant)
    strong_frac: float  # share above u_strong (blustery)
    comfortable_frac: float  # share inside the band

    def as_dict(self) -> dict:
        return {
            "mean_index": self.mean_index,
            "dead_frac": self.dead_frac,
            "strong_frac": self.strong_frac,
            "comfortable_frac": self.comfortable_frac,
        }


def open_mask(heightmap: np.ndarray) -> np.ndarray:
    """Cells NOT covered by a building -- where pedestrians actually are."""
    return heightmap <= 1e-6


def comfort(
    speed: np.ndarray,
    heightmap: np.ndarray,
    u0: float,
    region: np.ndarray | None = None,
    thr: Thresholds | None = None,
) -> Comfort:
    """Comfort over the open cells (optionally restricted to a ``region`` boolean mask)."""
    thr = thr or Thresholds()
    m = open_mask(heightmap)
    if region is not None:
        m = m & region
    u = speed[m]
    if u.size == 0:
        return Comfort(u0, 0.0, 0.0, 0.0, 0.0)
    lo, hi = thr.dead * u0, thr.strong * u0
    return Comfort(
        u0=u0,
        mean_index=float(u.mean() / u0),
        dead_frac=float((u < lo).mean()),
        strong_frac=float((u > hi).mean()),
        comfortable_frac=float(((u >= lo) & (u <= hi)).mean()),
    )
