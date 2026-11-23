from pathlib import Path

import numpy as np
import pytest

from charuco_calib.board import BoardSpec
from charuco_calib.model import CalibrationError, Intrinsics
from charuco_calib.report import build_report, format_summary, write_report


def _locked(**overrides) -> Intrinsics:
    values = dict(rms=0.4, fx=800, fy=800, cx=960, cy=540, k1=-0.22, k2=0.03, p1=0, p2=0, k3=0)
    values.update(overrides)
    return Intrinsics(**values)


def _board() -> BoardSpec:
    return BoardSpec(5, 7, "DICT_4X4_50", 32.0, 24.0)


def test_report_leads_with_the_values_to_type_in(tmp_path):
    report = build_report(
        _locked(),
        None,
        image_size=(1920, 1080),
        fov_axis="vertical",
        views=16,
        dropped=0,
        per_view_rms=[0.4, 0.5],
        cells=7,
        min_cells=6,
        board=_board(),
        source="rtsp://user:s3cret@192.168.1.10/Streaming/Channels/101",
        camera="cam4",
    )
    assert list(report)[0] == "focal_length"
    assert report["focal_unit"] == "mm"
    assert report["k1"] == pytest.approx(-0.22)
    assert report["k2"] == pytest.approx(0.03)
    assert report["p1"] == 0
    assert report["camera"] == "cam4"
    assert "s3cret" not in report["source"]
    path = tmp_path / "calibration.yml"
    write_report(path, report)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Ground-plane lens values")
    assert "s3cret" not in text
    assert "focal_length" in text


def test_report_warns_when_k_is_outside_the_page_range():
    report = build_report(
        _locked(k1=-1.4),
        None,
        image_size=(1920, 1080),
        fov_axis="vertical",
        views=8,
        dropped=0,
        per_view_rms=[0.2],
        cells=6,
        min_cells=6,
        board=_board(),
        source="rtsp://192.168.1.10/stream",
        camera=None,
    )
    assert any("outside -1 to 1" in note for note in report["warnings"])
    assert np.isfinite(report["focal_length"])


def test_diagonal_axis_is_wider_than_vertical():
    report = build_report(
        _locked(),
        None,
        image_size=(1920, 1080),
        fov_axis="diagonal",
        views=8,
        dropped=0,
        per_view_rms=[0.2],
        cells=6,
        min_cells=6,
        board=_board(),
        source="rtsp://192.168.1.10/stream",
        camera=None,
    )
    assert report["fov_deg"] == report["fov_diagonal_deg"]
    assert report["fov_horizontal_deg"] > report["fov_vertical_deg"]
    assert report["fov_diagonal_deg"] > report["fov_horizontal_deg"]
    summary = format_summary(report, Path("calibration.yml"))
    assert "fov_horizontal_deg:" in summary
    assert "fov_vertical_deg:" in summary
    assert "fov_diagonal_deg:" in summary


def test_unconstrained_field_of_view_uses_each_focal_length():
    report = build_report(
        _locked(),
        _locked(fx=700, fy=900, k1=-0.4, k2=0.1, cx=960, cy=540),
        image_size=(1920, 1080),
        fov_axis="vertical",
        views=8,
        dropped=0,
        per_view_rms=[0.2],
        cells=6,
        min_cells=6,
        board=_board(),
        source="rtsp://192.168.1.10/stream",
        camera=None,
    )
    free = report["unconstrained"]
    assert free["fov_horizontal_deg"] > free["fov_vertical_deg"]
    assert free["fov_diagonal_deg"] > free["fov_horizontal_deg"]
    assert any("Free fit" in note for note in report["warnings"])
