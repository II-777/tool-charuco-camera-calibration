"""Read an RTSP stream until the ChArUco sheet has been shown from enough angles."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2
import numpy as np

from charuco_calib.board import BoardSpec, CharucoHit, detect_charuco, make_cv_board, make_detector
from charuco_calib.model import CalibrationError, redact_url
from charuco_calib.select import Pose, cells_of, centroid_of, covered_cells, is_novel, is_steady, span_fraction


@dataclass(frozen=True)
class CaptureSettings:
    target_views: int
    min_views: int
    max_views: int
    min_cells: int
    seconds: float
    no_window: bool
    preview_path: Path | None


@dataclass
class CaptureBundle:
    image_size: tuple[int, int]
    hits: list[CharucoHit]
    cell_count: int
    warnings: list[str]


def capture_views(url: str, spec: BoardSpec, settings: CaptureSettings) -> CaptureBundle:
    if settings.min_views < 3:
        raise CalibrationError("need a minimum of 3 views")
    if settings.target_views < settings.min_views or settings.max_views < settings.target_views:
        raise CalibrationError("view counts must satisfy min <= target <= max")
    cap = _open_capture(url)
    board = make_cv_board(spec)
    detector = make_detector(board)
    window = _Window(settings.no_window)
    min_corners = max(8, int(spec.corner_count * 0.45))
    accepted: list[Pose] = []
    hits: list[CharucoHit] = []
    steady: list[tuple[float, float]] = []
    image_size: tuple[int, int] | None = None
    started = time.monotonic()
    first_deadline = started + 15.0
    idle = 0
    last_status = ""
    preview_tick = 0
    print(f"Reading {redact_url(url)}", flush=True)
    print("Hold the sheet still until a view is accepted, then move it. Cover the center and the corners.", flush=True)
    print("Enter finishes. Esc cancels.", flush=True)
    try:
        while True:
            if time.monotonic() - started > settings.seconds:
                print("Capture time is up.", flush=True)
                break
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                idle += 1
                time.sleep(0.05)
                if image_size is None and time.monotonic() > first_deadline:
                    raise CalibrationError(f"no frames from {redact_url(url)}")
                if idle > 150:
                    raise CalibrationError(f"the stream stopped: {redact_url(url)}")
                continue
            idle = 0
            height, width = frame.shape[:2]
            if image_size is None:
                image_size = (width, height)
                print(f"Stream frame {width}x{height}", flush=True)
            elif image_size != (width, height):
                raise CalibrationError("the stream changed resolution during capture")

            hit = detect_charuco(frame, board, detector)
            status = "show the board"
            took = False
            usable = (
                hit is not None
                and len(hit.image_points) >= min_corners
                and span_fraction(hit.image_points, width) >= 0.22
            )
            if not usable:
                steady.clear()
                if hit is not None:
                    status = "move the board closer"
            else:
                assert hit is not None
                pose = Pose(
                    centroid_of(hit.image_points),
                    cells_of(hit.image_points, width, height),
                    span_fraction(hit.image_points, width) * width,
                )
                steady.append(pose.centroid)
                tolerance = max(8.0, 0.008 * width)
                shift = max(24.0, 0.06 * width)
                if is_steady(steady, tolerance) and is_novel(pose, accepted, shift):
                    accepted.append(pose)
                    hits.append(hit)
                    steady.clear()
                    took = True
                    status = "accepted"
                    cells = len(covered_cells(accepted))
                    print(
                        f"accepted {len(hits)}/{settings.target_views}  "
                        f"corners {len(hit.image_points)}  coverage {cells}/9",
                        flush=True,
                    )
                elif is_steady(steady, tolerance):
                    status = "move the board"
                else:
                    status = "hold still"

            if not window.enabled and status != last_status:
                print(status, flush=True)
                last_status = status
            cells_now = len(covered_cells(accepted))
            preview = _overlay(frame, hit, status, len(hits), settings.target_views, cells_now)
            preview_tick += 1
            if settings.preview_path is not None and (took or preview_tick % 15 == 0):
                settings.preview_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(settings.preview_path), preview)
            action = window.show(preview)
            if action == "abort":
                raise CalibrationError("cancelled")
            if action == "quit":
                if len(hits) >= settings.min_views:
                    break
                print(f"need {settings.min_views - len(hits)} more views before finishing", flush=True)
            if len(hits) >= settings.max_views:
                break
            if len(hits) >= settings.target_views and cells_now >= settings.min_cells:
                break
    finally:
        cap.release()
        window.close()

    if image_size is None or len(hits) < settings.min_views:
        raise CalibrationError(f"only {len(hits)} views. Need {settings.min_views}.")
    return CaptureBundle(image_size, hits, len(covered_cells(accepted)), [])


def _open_capture(url: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise CalibrationError(f"could not open {redact_url(url)}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _overlay(frame, hit: CharucoHit | None, status: str, count: int, target: int, cells: int) -> np.ndarray:
    canvas = frame.copy()
    if hit is not None:
        try:
            cv2.aruco.drawDetectedCornersCharuco(canvas, hit.corners, hit.ids)
        except cv2.error:
            pass
    color = (0, 220, 0) if status == "accepted" else (0, 220, 255)
    lines = [status, f"views {count}/{target}   coverage {cells}/9", "Enter finishes   Esc cancels"]
    y = 36
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += 32
    return canvas


class _Window:
    def __init__(self, disabled: bool) -> None:
        self.enabled = not disabled and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.name = "ChArUco calibration"
        self.opened = False

    def show(self, frame: np.ndarray) -> str:
        if not self.enabled:
            return ""
        try:
            if not self.opened:
                cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
                self.opened = True
            cv2.imshow(self.name, frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            self.enabled = False
            return ""
        if key == 27:
            return "abort"
        if key in (13, 10, ord("q"), ord("Q")):
            return "quit"
        return ""

    def close(self) -> None:
        if self.opened:
            cv2.destroyWindow(self.name)
