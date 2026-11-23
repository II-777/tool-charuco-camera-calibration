import math

from charuco_calib.model import (
    focal_mm,
    focal_px_from_mm,
    fov_for_axis,
    redact_url,
    seed_focal_px,
)


def test_focal_length_round_trips_through_the_sensor_height():
    height = 1080
    pixels = 800.0
    millimetres = focal_mm(pixels, height)
    assert math.isclose(focal_px_from_mm(millimetres, height), pixels, rel_tol=1e-12)


def test_vertical_seed_matches_the_field_of_view():
    height = 1080
    fov = 74.0
    pixels = seed_focal_px(height, fov)
    assert math.isclose(fov_for_axis("vertical", 1920, height, pixels), fov, abs_tol=1e-9)


def test_redact_url_drops_the_password():
    raw = "rtsp://user:s3cret@192.168.1.10:554/Streaming/Channels/101"
    assert redact_url(raw) == "rtsp://192.168.1.10:554/Streaming/Channels/101"
    assert "s3cret" not in redact_url(raw)
