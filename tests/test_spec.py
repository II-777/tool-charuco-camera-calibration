from pathlib import Path

import pytest

from charuco_calib.board import spec_from_file
from charuco_calib.cli import log_path, report_path
from charuco_calib.model import CalibrationError


def test_spec_file_matches_the_printed_sheet(tmp_path):
    path = tmp_path / "charuco-board.yml"
    path.write_text(
        "\n".join(
            [
                "# Printed ChArUco sheet. Pass this file to tool-charuco-calib with --spec.",
                "dictionary: DICT_4X4_50",
                "squares_x: 5",
                "squares_y: 7",
                "square_mm: 30.0",
                "marker_mm: 22.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    spec = spec_from_file(path)
    assert spec.dictionary == "DICT_4X4_50"
    assert (spec.squares_x, spec.squares_y) == (5, 7)
    assert spec.square_mm == 30
    assert spec.marker_mm == 22.5


def test_bare_log_name_goes_in_outputs():
    assert log_path("camera-name.yml") == Path("outputs/camera-name.yml")
    assert log_path("outputs/camera-name.yml") == Path("outputs/camera-name.yml")
    assert log_path("/tmp/camera-name.yml") == Path("/tmp/camera-name.yml")


def test_camera_name_sets_the_log_path():
    assert report_path("camera-name", None) == Path("outputs/camera-name.yml")
    assert report_path("camera-name", "custom.yml") == Path("outputs/custom.yml")
    assert report_path("camera-name", "logs/custom.yml") == Path("logs/custom.yml")


def test_camera_name_rejects_a_path():
    with pytest.raises(CalibrationError):
        report_path("../camera-name", None)
