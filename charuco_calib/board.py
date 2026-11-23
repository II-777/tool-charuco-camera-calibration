"""ChArUco geometry read from the printed-sheet spec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from charuco_calib.model import CalibrationError

DICTIONARIES: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    dictionary: str
    square_mm: float
    marker_mm: float

    @property
    def corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)


@dataclass(frozen=True)
class CharucoHit:
    object_points: np.ndarray
    image_points: np.ndarray
    corners: np.ndarray
    ids: np.ndarray


def resolve_spec(spec: BoardSpec) -> BoardSpec:
    if spec.squares_x < 3 or spec.squares_y < 3:
        raise CalibrationError("the board needs at least 3 squares along each side")
    if spec.dictionary not in DICTIONARIES:
        known = ", ".join(sorted(DICTIONARIES))
        raise CalibrationError(f"unknown dictionary {spec.dictionary}. Choose one of: {known}")
    if spec.square_mm <= 5:
        raise CalibrationError("square size must be larger than 5 mm")
    if spec.marker_mm <= 0 or spec.marker_mm >= spec.square_mm:
        raise CalibrationError("marker size must be greater than 0 and smaller than the square")
    _require_dictionary_capacity(spec)
    return spec


def spec_from_file(path: Path) -> BoardSpec:
    """Read the yaml written beside the printed sheet."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationError(f"could not read the board spec {path}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"{path} is not a board spec")
    try:
        spec = BoardSpec(
            squares_x=int(data["squares_x"]),
            squares_y=int(data["squares_y"]),
            dictionary=str(data["dictionary"]),
            square_mm=float(data["square_mm"]),
            marker_mm=float(data["marker_mm"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{path} is missing the board sizes") from exc
    return resolve_spec(spec)


def make_cv_board(spec: BoardSpec) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[spec.dictionary])
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_mm / 1000.0,
        spec.marker_mm / 1000.0,
        dictionary,
    )


def make_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    return cv2.aruco.CharucoDetector(board)


def detect_charuco(image: np.ndarray, board: cv2.aruco.CharucoBoard, detector: cv2.aruco.CharucoDetector) -> CharucoHit | None:
    detected = detector.detectBoard(image)
    if detected is None or len(detected) < 2:
        return None
    corners, ids = detected[0], detected[1]
    if corners is None or ids is None or len(ids) < 4:
        return None
    try:
        matched = board.matchImagePoints(corners, ids)
    except cv2.error:
        return None
    if matched is None or len(matched) != 2:
        return None
    obj, img = matched
    if obj is None or img is None:
        return None
    object_points = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    if len(object_points) < 4 or len(object_points) != len(image_points):
        return None
    draw_corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    draw_ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
    if len(draw_corners) != len(draw_ids):
        return None
    return CharucoHit(
        object_points=object_points,
        image_points=image_points,
        corners=draw_corners,
        ids=draw_ids,
    )


def _require_dictionary_capacity(spec: BoardSpec) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[spec.dictionary])
    board = make_cv_board(spec)
    marker_ids = board.getIds()
    capacity = int(dictionary.bytesList.shape[0])
    if len(marker_ids) > capacity:
        raise CalibrationError(
            f"{spec.dictionary} has {capacity} markers and this board needs {len(marker_ids)}. Use a larger dictionary."
        )
