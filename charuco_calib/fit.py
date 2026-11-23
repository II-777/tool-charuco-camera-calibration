"""Fit the locked Brown-Conrady model OpenCV uses for k1 and k2."""

from __future__ import annotations

import math

import cv2
import numpy as np

from charuco_calib.model import CalibrationError, Intrinsics, seed_focal_px

LOCKED_FLAGS = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    | cv2.CALIB_FIX_ASPECT_RATIO
    | cv2.CALIB_FIX_PRINCIPAL_POINT
    | cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K3
)
FREE_FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-9)


class FitResult:
    def __init__(
        self,
        locked: Intrinsics,
        unconstrained: Intrinsics | None,
        per_view_rms: list[float],
        dropped: int,
        kept: int,
    ) -> None:
        self.locked = locked
        self.unconstrained = unconstrained
        self.per_view_rms = per_view_rms
        self.dropped = dropped
        self.kept = kept


def fit_views(
    views: list[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
    seed_fov_deg: float,
) -> FitResult:
    """Fit k1, k2, and one focal length. Principal point stays at the frame center.

    `image_size` is (width, height). One view with a large reprojection error is
    dropped and the fit is run once more.
    """
    prepared = [_as_view(obj, img) for obj, img in views]
    if len(prepared) < 3:
        raise CalibrationError("need at least 3 views of the board")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise CalibrationError("the stream frame has no size")

    locked, _rvecs, _tvecs = _fit(prepared, image_size, seed_fov_deg, locked=True)
    errors = _view_errors(prepared, locked)
    kept_index = _inliers(errors)
    dropped = len(prepared) - len(kept_index)
    if dropped and len(kept_index) >= 6:
        subset = [prepared[index] for index in kept_index]
        locked, _rvecs, _tvecs = _fit(subset, image_size, seed_fov_deg, locked=True)
        errors = _view_errors(subset, locked)
        kept = subset
    else:
        dropped = 0
        kept = prepared

    unconstrained = _try_free(kept, image_size, locked)
    return FitResult(locked, unconstrained, errors, dropped, len(kept))


def _fit(
    views: list[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
    seed_fov_deg: float,
    locked: bool,
    guess: Intrinsics | None = None,
) -> tuple[Intrinsics, list, list]:
    width, height = image_size
    if guess is None:
        focal = seed_focal_px(height, seed_fov_deg)
        camera = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        camera = np.array(
            [[guess.fx, 0.0, guess.cx], [0.0, guess.fy, guess.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.array([guess.k1, guess.k2, guess.p1, guess.p2, 0.0], dtype=np.float64).reshape(5, 1)
    obj = [item[0].reshape(-1, 1, 3) for item in views]
    img = [item[1].reshape(-1, 1, 2) for item in views]
    try:
        rms, matrix, dist_out, rvecs, tvecs = cv2.calibrateCamera(
            obj,
            img,
            image_size,
            camera,
            dist,
            flags=LOCKED_FLAGS if locked else FREE_FLAGS,
            criteria=CRITERIA,
        )
    except cv2.error as exc:
        raise CalibrationError(f"calibration failed: {exc}") from exc
    intrinsics = _unpack(float(rms), matrix, dist_out)
    if locked:
        intrinsics = Intrinsics(
            rms=intrinsics.rms,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=width / 2.0,
            cy=height / 2.0,
            k1=intrinsics.k1,
            k2=intrinsics.k2,
            p1=0.0,
            p2=0.0,
            k3=0.0,
        )
    if not math.isfinite(intrinsics.rms) or intrinsics.rms > 20:
        raise CalibrationError(f"reprojection error is {intrinsics.rms:.1f} px. Recapture the board.")
    if intrinsics.focal_px <= 0 or not math.isfinite(intrinsics.focal_px):
        raise CalibrationError("calibration returned a focal length that is not usable")
    return intrinsics, rvecs, tvecs


def _try_free(
    views: list[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
    locked: Intrinsics,
) -> Intrinsics | None:
    try:
        free, _rvecs, _tvecs = _fit(views, image_size, 60.0, locked=False, guess=locked)
    except CalibrationError:
        return None
    return free


def _view_errors(views: list[tuple[np.ndarray, np.ndarray]], intrinsics: Intrinsics) -> list[float]:
    camera = np.array(
        [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array([intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3], dtype=np.float64)
    errors: list[float] = []
    for obj, img in views:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj.reshape(-1, 1, 3),
                img.reshape(-1, 1, 2),
                camera,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            errors.append(1e6)
            continue
        if not ok:
            errors.append(1e6)
            continue
        projected, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, camera, dist)
        delta = projected.reshape(-1, 2) - img.reshape(-1, 2)
        errors.append(float(np.linalg.norm(delta, axis=1).mean()))
    return errors


def _inliers(errors: list[float]) -> list[int]:
    ordered = sorted(errors)
    median = ordered[len(ordered) // 2]
    limit = max(1.0, median * 3.0, median + 0.75)
    return [index for index, error in enumerate(errors) if error <= limit]


def _as_view(obj, img) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.ascontiguousarray(np.asarray(obj, dtype=np.float32).reshape(-1, 3))
    image_points = np.ascontiguousarray(np.asarray(img, dtype=np.float32).reshape(-1, 2))
    if len(object_points) != len(image_points) or len(object_points) < 6:
        raise CalibrationError("each view needs at least 6 matched corners")
    return object_points, image_points


def _unpack(rms: float, matrix, dist) -> Intrinsics:
    camera = np.asarray(matrix, dtype=np.float64)
    coeffs = np.asarray(dist, dtype=np.float64).reshape(-1)
    padded = np.zeros(5, dtype=np.float64)
    padded[: min(5, len(coeffs))] = coeffs[:5]
    return Intrinsics(
        rms=rms,
        fx=float(camera[0, 0]),
        fy=float(camera[1, 1]),
        cx=float(camera[0, 2]),
        cy=float(camera[1, 2]),
        k1=float(padded[0]),
        k2=float(padded[1]),
        p1=float(padded[2]),
        p2=float(padded[3]),
        k3=float(padded[4]),
    )
