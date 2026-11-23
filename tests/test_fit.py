import cv2
import numpy as np
import pytest

from charuco_calib.fit import fit_views
from charuco_calib.model import CalibrationError


def test_locked_fit_recovers_k1_k2_and_focal_length():
    width, height = 640, 480
    focal = 520.0
    k1, k2 = -0.22, 0.04
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)
    dist = np.array([k1, k2, 0, 0, 0], np.float64)
    xs = np.linspace(-0.16, 0.16, 9)
    ys = np.linspace(-0.11, 0.11, 7)
    obj = np.array([[x, y, 0.0] for y in ys for x in xs], np.float64)
    poses = []
    for rx in (-0.2, 0.0, 0.22):
        for ry in (-0.16, 0.18):
            for translation in ((0.0, 0.0, 0.38), (-0.025, 0.02, 0.42), (0.03, -0.015, 0.34)):
                poses.append(((rx, ry, 0.08), translation))
    views = []
    for rotation, translation in poses:
        image, _ = cv2.projectPoints(
            obj,
            np.array(rotation, np.float64),
            np.array(translation, np.float64),
            camera,
            dist,
        )
        points = image.reshape(-1, 2)
        if points[:, 0].min() < 4 or points[:, 1].min() < 4:
            continue
        if points[:, 0].max() >= width - 4 or points[:, 1].max() >= height - 4:
            continue
        if points[:, 0].max() - points[:, 0].min() < 0.45 * width:
            continue
        views.append((obj.copy(), points))
    assert len(views) >= 8
    result = fit_views(views[:12], (width, height), seed_fov_deg=70)
    assert result.locked.k1 == pytest.approx(k1, abs=0.02)
    assert result.locked.k2 == pytest.approx(k2, abs=0.03)
    assert result.locked.focal_px == pytest.approx(focal, abs=15)
    assert result.locked.cx == pytest.approx(width / 2)
    assert result.locked.cy == pytest.approx(height / 2)
    assert result.locked.p1 == 0
    assert result.locked.p2 == 0
    assert result.locked.k3 == 0


def test_fit_requires_three_views():
    obj = np.zeros((8, 3))
    img = np.zeros((8, 2))
    with pytest.raises(CalibrationError):
        fit_views([(obj, img), (obj, img)], (640, 480), 70)
