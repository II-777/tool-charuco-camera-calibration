"""Lens numbers the ground-plane overlay can store."""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

# 1/2.8" 1080p sensor height at 2.9 µm. The overlay converts millimetres with this.
SENSOR_HEIGHT_MM = 3.132

FOV_AXES = ("vertical", "horizontal", "diagonal")


class CalibrationError(Exception):
    """A problem the operator can correct."""


@dataclass(frozen=True)
class Intrinsics:
    rms: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    @property
    def focal_px(self) -> float:
        return (self.fx + self.fy) / 2.0


def focal_mm(focal_px: float, frame_height: int, sensor_height_mm: float = SENSOR_HEIGHT_MM) -> float:
    """Millimetres the overlay turns back into pixels with `focal * height / sensor`."""
    if frame_height <= 0 or sensor_height_mm <= 0 or focal_px <= 0:
        raise CalibrationError("focal length needs a positive frame height and sensor height")
    return focal_px * sensor_height_mm / frame_height


def focal_px_from_mm(focal_mm_value: float, frame_height: int, sensor_height_mm: float = SENSOR_HEIGHT_MM) -> float:
    if frame_height <= 0 or sensor_height_mm <= 0 or focal_mm_value <= 0:
        raise CalibrationError("focal length needs a positive frame height and sensor height")
    return focal_mm_value * frame_height / sensor_height_mm


def field_of_view_deg(span_px: float, focal_px: float) -> float:
    """`2 * atan(span / (2 * f))` in degrees. Span is width, height, or the diagonal."""
    if span_px <= 0 or focal_px <= 0:
        raise CalibrationError("field of view needs a positive span and focal length")
    return math.degrees(2.0 * math.atan(span_px / (2.0 * focal_px)))


def seed_focal_px(frame_height: int, fov_deg: float) -> float:
    if frame_height <= 0 or not 0 < fov_deg < 180:
        raise CalibrationError("seed field of view must be between 0 and 180 degrees")
    return frame_height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def field_of_view_axes(width: int, height: int, fx: float, fy: float) -> tuple[float, float, float]:
    """Horizontal, vertical, and diagonal field of view in degrees.

    Horizontal uses fx and the frame width. Vertical uses fy and the frame height.
    Diagonal is the angle between the opposite corners.
    """
    horizontal = field_of_view_deg(width, fx)
    vertical = field_of_view_deg(height, fy)
    diagonal = math.degrees(2.0 * math.atan(math.hypot(width / (2.0 * fx), height / (2.0 * fy))))
    return horizontal, vertical, diagonal


def fov_for_axis(axis: str, width: int, height: int, focal_px: float) -> float:
    if axis == "horizontal":
        span = float(width)
    elif axis == "vertical":
        span = float(height)
    elif axis == "diagonal":
        span = math.hypot(width, height)
    else:
        raise CalibrationError(f"fov axis must be one of: {', '.join(FOV_AXES)}")
    return field_of_view_deg(span, focal_px)


def redact_url(url: str) -> str:
    """Drop userinfo so a saved calibration file does not keep the camera password."""
    raw = url.strip()
    parts = urlsplit(raw)
    if not parts.scheme or not parts.hostname:
        return raw
    host = parts.hostname
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def scrub_secret(message: str, secret: str) -> str:
    if secret and secret in message:
        return message.replace(secret, redact_url(secret))
    return message
