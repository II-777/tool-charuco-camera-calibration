"""Decide when a detected board is a new, steady calibration view."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    centroid: tuple[float, float]
    cells: frozenset[tuple[int, int]]
    span_px: float


def centroid_of(points) -> tuple[float, float]:
    pts = _pairs(points)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def cells_of(points, width: int, height: int, divisions: int = 3) -> frozenset[tuple[int, int]]:
    if width <= 0 or height <= 0 or divisions < 1:
        return frozenset()
    pts = _pairs(points)
    cells: set[tuple[int, int]] = set()
    for x, y in pts:
        col = min(divisions - 1, max(0, int(x / width * divisions)))
        row = min(divisions - 1, max(0, int(y / height * divisions)))
        cells.add((col, row))
    return frozenset(cells)


def span_fraction(points, width: int) -> float:
    if width <= 0:
        return 0.0
    pts = _pairs(points)
    if len(pts) == 0:
        return 0.0
    wide = float(pts[:, 0].max() - pts[:, 0].min())
    tall = float(pts[:, 1].max() - pts[:, 1].min())
    return max(wide, tall) / float(width)


def is_steady(centroids: list[tuple[float, float]], tolerance_px: float, count: int = 6) -> bool:
    if len(centroids) < count:
        return False
    window = centroids[-count:]
    xs = [item[0] for item in window]
    ys = [item[1] for item in window]
    return max(xs) - min(xs) <= tolerance_px and max(ys) - min(ys) <= tolerance_px


def is_novel(pose: Pose, accepted: list[Pose], min_shift_px: float) -> bool:
    """A tilt in place changes the board's size even when its center stays put."""
    for previous in accepted:
        shift = math.hypot(pose.centroid[0] - previous.centroid[0], pose.centroid[1] - previous.centroid[1])
        resized = abs(pose.span_px - previous.span_px)
        if shift < min_shift_px and resized < min_shift_px:
            return False
    return True


def covered_cells(accepted: list[Pose]) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for pose in accepted:
        cells.update(pose.cells)
    return frozenset(cells)


def _pairs(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)
