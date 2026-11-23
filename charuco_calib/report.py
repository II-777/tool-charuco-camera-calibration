"""YAML record of the lens values to type into the ground plane."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from charuco_calib.board import BoardSpec
from charuco_calib.model import (
    SENSOR_HEIGHT_MM,
    Intrinsics,
    field_of_view_axes,
    focal_mm,
    fov_for_axis,
    redact_url,
)

HEADER = """\
# Ground-plane lens values. Type focal_length, k1, and k2 for this camera.
# Square pixels, principal point at the frame center, p1 = p2 = k3 = 0.
# Horizontal, vertical, and diagonal field of view are this same lens on this frame.
# A new zoom needs a new file. Height, tilt, and the ground homography are not here.
"""


def build_report(
    locked: Intrinsics,
    unconstrained: Intrinsics | None,
    *,
    image_size: tuple[int, int],
    fov_axis: str,
    views: int,
    dropped: int,
    per_view_rms: list[float],
    cells: int,
    min_cells: int,
    board: BoardSpec,
    source: str,
    camera: str | None,
) -> dict:
    width, height = image_size
    focal_px = locked.focal_px
    horizontal, vertical, diagonal = field_of_view_axes(width, height, locked.fx, locked.fy)
    shown = fov_for_axis(fov_axis, width, height, focal_px)
    report: dict = {
        "focal_length": round(focal_mm(focal_px, height), 3),
        "focal_unit": "mm",
        "lens_source": "focal",
        "fov_horizontal_deg": round(horizontal, 2),
        "fov_vertical_deg": round(vertical, 2),
        "fov_diagonal_deg": round(diagonal, 2),
        "fov_deg": round(shown, 2),
        "fov_axis": fov_axis,
        "k1": round(locked.k1, 5),
        "k2": round(locked.k2, 5),
        "p1": 0,
        "p2": 0,
        "k3": 0,
        "image": {"width": width, "height": height},
        "fx": round(locked.fx, 3),
        "fy": round(locked.fy, 3),
        "cx": round(locked.cx, 3),
        "cy": round(locked.cy, 3),
        "sensor_height_mm": SENSOR_HEIGHT_MM,
        "rms_px": round(locked.rms, 3),
        "views": views,
        "dropped_views": dropped,
        "per_view_rms_px": [round(value, 3) for value in per_view_rms],
        "board": {
            "dictionary": board.dictionary,
            "squares_x": board.squares_x,
            "squares_y": board.squares_y,
            "square_mm": board.square_mm,
            "marker_mm": board.marker_mm,
        },
        "source": redact_url(source),
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "opencv": cv2.__version__,
    }
    if camera:
        report["camera"] = camera
    if unconstrained is not None:
        free_h, free_v, free_d = field_of_view_axes(width, height, unconstrained.fx, unconstrained.fy)
        report["unconstrained"] = {
            "rms_px": round(unconstrained.rms, 3),
            "fx": round(unconstrained.fx, 3),
            "fy": round(unconstrained.fy, 3),
            "cx": round(unconstrained.cx, 3),
            "cy": round(unconstrained.cy, 3),
            "k1": round(unconstrained.k1, 5),
            "k2": round(unconstrained.k2, 5),
            "p1": round(unconstrained.p1, 5),
            "p2": round(unconstrained.p2, 5),
            "k3": round(unconstrained.k3, 5),
            "fov_horizontal_deg": round(free_h, 2),
            "fov_vertical_deg": round(free_v, 2),
            "fov_diagonal_deg": round(free_d, 2),
        }
    warnings = _warnings(locked, unconstrained, image_size, shown, cells, min_cells)
    if warnings:
        report["warnings"] = warnings
    return report


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = HEADER + yaml.safe_dump(report, sort_keys=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def format_summary(report: dict, path: Path) -> str:
    lines = [
        f"focal_length: {report['focal_length']} mm",
        f"fov_horizontal_deg: {report['fov_horizontal_deg']}",
        f"fov_vertical_deg: {report['fov_vertical_deg']}",
        f"fov_diagonal_deg: {report['fov_diagonal_deg']}",
        f"k1: {report['k1']}",
        f"k2: {report['k2']}",
        f"rms_px: {report['rms_px']}",
        f"views: {report['views']}",
    ]
    for warning in report.get("warnings") or []:
        lines.append(f"warning: {warning}")
    lines.append(f"wrote {path}")
    return "\n".join(lines)


def _warnings(
    locked: Intrinsics,
    unconstrained: Intrinsics | None,
    image_size: tuple[int, int],
    fov_deg: float,
    cells: int,
    min_cells: int,
) -> list[str]:
    width, height = image_size
    notes: list[str] = []
    if locked.rms > 1:
        notes.append(f"Reprojection error is {locked.rms:.2f} px. A sharp capture is usually under 1 px.")
    if abs(locked.k1) > 1 or abs(locked.k2) > 1:
        notes.append("k1 or k2 is outside -1 to 1, which is the range the ground-plane page can store.")
    if not 8 <= fov_deg <= 150:
        notes.append("fov_deg is outside 8 to 150, which is the range a ground-plane save accepts.")
    if cells < min_cells:
        notes.append(
            "The board did not reach enough of the frame. k2 is weakly determined. Cover the corners and run again."
        )
    if unconstrained is not None:
        offset = math.hypot(unconstrained.cx - width / 2.0, unconstrained.cy - height / 2.0)
        if offset > 0.02 * width:
            notes.append(
                f"An unconstrained fit puts the principal point {offset:.0f} px off center. The overlay cannot store that shift."
            )
        if abs(unconstrained.p1) > 0.02 or abs(unconstrained.p2) > 0.02:
            notes.append("An unconstrained fit wants tangential distortion. The overlay keeps p1 and p2 at 0.")
        free_focal = (unconstrained.fx + unconstrained.fy) / 2.0
        focal_gap = abs(free_focal - locked.focal_px) / locked.focal_px
        distortion_gap = max(abs(unconstrained.k1 - locked.k1), abs(unconstrained.k2 - locked.k2))
        if focal_gap > 0.05 or distortion_gap > 0.05:
            notes.append(
                f"Free fit is {free_focal:.0f} px, k1 {unconstrained.k1:.3f}. "
                f"Stored fit is {locked.focal_px:.0f} px, k1 {locked.k1:.3f}. "
                "A repeat capture keeps moving these until the reprojection error is under 1 px."
            )
    return notes
