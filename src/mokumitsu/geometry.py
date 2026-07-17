"""Numerical geometry core for the wind-design engine.

torch + numpy ONLY (no `windcfd`, no `neuralop`), so this whole package imports and runs
unchanged inside Houdini's bundled Python (hython) as well as in the project venv -- the engine
is self-contained in the Houdini frontend. Footprints <-> height-maps, a DIFFERENTIABLE soft
raster for the optimizer, and exact oriented-box overlap for the non-overlapping display.

A "footprint" is the tuple ``(cx, cy, w, d, h, theta)`` in normalized [0, 1] world coords:
centre (cx, cy), plan size (w, d), building height h, rotation theta (rad, about +z).
"""

from __future__ import annotations

import numpy as np
import torch

Footprint = tuple  # (cx, cy, w, d, h, theta)


def grid_centres(ny: int, nx: int, L: float = 1.0):
    """Cell-centre x and y coordinate vectors of an (ny, nx) field over a square domain."""
    xs = (np.arange(nx) + 0.5) / nx * L
    ys = (np.arange(ny) + 0.5) / ny * L
    return xs, ys


def rect_corners(cx, cy, w, d, theta) -> np.ndarray:
    """World-space corners (4, 2) of an oriented footprint."""
    hx, hy = w / 2.0, d / 2.0
    loc = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return loc @ rot.T + np.array([cx, cy])


def _inside_convex(gp: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Boolean (M,) -- which of the grid points gp (M, 2) are inside the convex polygon poly."""
    k = len(poly)
    sign = np.empty((len(gp), k))
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        sign[:, i] = (b[0] - a[0]) * (gp[:, 1] - a[1]) - (b[1] - a[1]) * (gp[:, 0] - a[0])
    return (sign >= -1e-9).all(1) | (sign <= 1e-9).all(1)


def rasterize(rects, ny: int, nx: int, L: float = 1.0) -> np.ndarray:
    """Hard height-map (ny, nx) -- the max building height over each cell. The FNO input."""
    hm = np.zeros((ny, nx), np.float32)
    flat = hm.reshape(-1)
    xs, ys = grid_centres(ny, nx, L)
    gx, gy = np.meshgrid(xs, ys)
    gp = np.stack([gx.ravel(), gy.ravel()], axis=1)
    for cx, cy, w, d, h, th in rects:
        m = _inside_convex(gp, rect_corners(cx, cy, w, d, th))
        flat[m] = np.maximum(flat[m], float(h))
    return hm


def soft_heightmap(params: torch.Tensor, ny: int, nx: int, sharpness: float = 60.0, L: float = 1.0):
    """DIFFERENTIABLE soft height-map from params (N, 6) = (cx, cy, w, d, h, theta).

    Returns ``(hmap, occ)``, both (ny, nx) torch tensors: hmap is the soft max-composite of the
    buildings' heights (sigmoids on the signed distances in each building's local frame), occ is
    the soft sum of the [0, 1] indicators (used for a soft no-overlap penalty). Gradients flow to
    every param, so the optimizer can backprop a wind loss through the raster + the FNO.
    """
    dev, dt = params.device, params.dtype
    xs = (torch.arange(nx, device=dev, dtype=dt) + 0.5) / nx * L
    ys = (torch.arange(ny, device=dev, dtype=dt) + 0.5) / ny * L
    gx = xs[None, :].expand(ny, nx)
    gy = ys[:, None].expand(ny, nx)
    cx, cy, w, d, h, th = (params[:, i][:, None, None] for i in range(6))
    ct, st = th.cos(), th.sin()
    dx, dy = gx[None] - cx, gy[None] - cy
    lx = ct * dx + st * dy
    ly = -st * dx + ct * dy
    ind = torch.sigmoid(sharpness * (w / 2 - lx.abs())) * torch.sigmoid(
        sharpness * (d / 2 - ly.abs())
    )
    return (ind * h).max(0).values, ind.sum(0)


def _obb_axes(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c, s]), np.array([-s, c])


def obb_penetration(r1, r2, gap: float = 0.0) -> float:
    """Penetration depth of two oriented footprints via the separating-axis theorem.

    Returns 0 if they are separated by at least ``gap``; otherwise the smallest overlap depth.
    Exact for rotated rectangles.
    """
    c1, c2 = np.array(r1[:2]), np.array(r2[:2])
    hx1, hy1, hx2, hy2 = r1[2] / 2, r1[3] / 2, r2[2] / 2, r2[3] / 2
    u1, v1 = _obb_axes(r1[5])
    u2, v2 = _obb_axes(r2[5])
    d = c2 - c1
    depth = 1e18
    for a in (u1, v1, u2, v2):
        r1p = abs(hx1 * a.dot(u1)) + abs(hy1 * a.dot(v1))
        r2p = abs(hx2 * a.dot(u2)) + abs(hy2 * a.dot(v2))
        ov = r1p + r2p + gap - abs(d.dot(a))
        if ov <= 0:
            return 0.0
        depth = min(depth, ov)
    return float(depth)


def repair_overlaps(rects, gap: float, bounds, iters: int = 60):
    """Push overlapping footprints apart along their centre line until none collide.

    ``bounds`` = (lo_cx, hi_cx, lo_cy, hi_cy) clamps the centres. Guarantees the returned layout
    has no overlap within the bounds, so the displayed buildings never intersect.
    """
    lo_cx, hi_cx, lo_cy, hi_cy = bounds
    rs = [list(r) for r in rects]
    for _ in range(iters):
        moved = False
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                depth = obb_penetration(rs[i], rs[j], gap)
                if depth > 1e-6:
                    d = np.array(rs[j][:2]) - np.array(rs[i][:2])
                    nrm = float(np.hypot(*d))
                    d = d / nrm if nrm > 1e-9 else np.array([1.0, 0.0])
                    sh = (0.5 * depth + 1e-3) * d
                    rs[i][0] -= sh[0]
                    rs[i][1] -= sh[1]
                    rs[j][0] += sh[0]
                    rs[j][1] += sh[1]
                    moved = True
        for b in rs:
            b[0] = min(max(b[0], lo_cx), hi_cx)
            b[1] = min(max(b[1], lo_cy), hi_cy)
        if not moved:
            break
    return [tuple(r) for r in rs]
