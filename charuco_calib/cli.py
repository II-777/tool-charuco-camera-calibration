"""Calibrate a camera from its RTSP stream and a printed ChArUco sheet."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from charuco_calib.board import spec_from_file
from charuco_calib.capture import CaptureSettings, capture_views
from charuco_calib.fit import fit_views
from charuco_calib.model import CalibrationError, scrub_secret
from charuco_calib.report import build_report, format_summary, write_report

DEFAULT_SPEC = "../tool-charuco-board-generator/outputs/charuco-board.yml"


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except CalibrationError as exc:
        url = args.rtsp or os.environ.get("RTSP_URL", "")
        print(scrub_secret(str(exc), url), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


def _run(args: argparse.Namespace) -> int:
    url = (args.rtsp or os.environ.get("RTSP_URL") or "").strip()
    if not url:
        raise CalibrationError("pass --rtsp <rtsp-url>")
    if not 0 < args.seed_fov < 180:
        raise CalibrationError("--seed-fov must be between 0 and 180")
    if args.min_views < 3 or args.views < args.min_views or args.max_views < args.views:
        raise CalibrationError("view counts must satisfy 3 <= min <= target <= max")
    if not 1 <= args.min_cells <= 9:
        raise CalibrationError("--min-cells must be from 1 to 9")
    spec = spec_from_file(Path(args.spec))
    camera = _camera_name(args.camera)
    output = report_path(camera, args.output)
    preview = None
    if args.no_window or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        preview = output.with_name(f"{output.stem}-preview.jpg")
    bundle = capture_views(
        url,
        spec,
        CaptureSettings(
            target_views=args.views,
            min_views=args.min_views,
            max_views=args.max_views,
            min_cells=args.min_cells,
            seconds=args.seconds,
            no_window=args.no_window,
            preview_path=preview,
        ),
    )
    hits = [(hit.object_points, hit.image_points) for hit in bundle.hits]
    fit = fit_views(hits, bundle.image_size, args.seed_fov)
    report = build_report(
        fit.locked,
        fit.unconstrained,
        image_size=bundle.image_size,
        fov_axis=args.fov_axis,
        views=fit.kept,
        dropped=fit.dropped,
        per_view_rms=fit.per_view_rms,
        cells=bundle.cell_count,
        min_cells=args.min_cells,
        board=spec,
        source=url,
        camera=camera,
    )
    write_report(output, report)
    print(format_summary(report, output), flush=True)
    return 0


def log_path(raw: str) -> Path:
    """Put a bare filename in outputs/. A path that already has a directory is kept."""
    path = Path(raw)
    if path.parent == Path("."):
        return Path("outputs") / path.name
    return path


def report_path(camera: str, output: str | None) -> Path:
    """Default the log to outputs/<camera>.yml. --output replaces that name."""
    name = _camera_name(camera)
    if output:
        return log_path(output)
    return Path("outputs") / f"{name}.yml"


def _camera_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise CalibrationError("--camera must be a name such as <camera-name>")
    return cleaned


class _Help(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, width=100)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="charuco-calib",
        description="Calibrate a camera from an RTSP stream of a printed ChArUco sheet.",
        formatter_class=_Help,
        epilog=(
            "examples:\n"
            "  charuco-calib --rtsp <rtsp-url> --camera <camera-name>\n"
            "  charuco-calib --rtsp <rtsp-url> --camera <camera-name> --spec <board.yml> --output <file>\n"
        ),
    )
    parser.add_argument(
        "--rtsp",
        metavar="<rtsp-url>",
        help="Camera stream URL. RTSP_URL is used when this is omitted.",
    )
    parser.add_argument(
        "--camera",
        required=True,
        metavar="<camera-name>",
        help="Name stored in the log. The file is outputs/<camera-name>.yml.",
    )
    parser.add_argument(
        "--spec",
        metavar="<board.yml>",
        default=DEFAULT_SPEC,
        help="Printed-sheet spec. Default is the yaml from tool-charuco-board-generator.",
    )
    parser.add_argument(
        "--output",
        metavar="<file>",
        help="Log path. A bare filename is written under outputs/. Default: outputs/<camera-name>.yml.",
    )
    tuning = parser.add_argument_group("tuning")
    tuning.add_argument(
        "--fov-axis",
        choices=("vertical", "horizontal", "diagonal"),
        default="vertical",
        metavar="<axis>",
        help="Field of view stored as fov_deg: vertical, horizontal, or diagonal (default: vertical).",
    )
    tuning.add_argument(
        "--views",
        type=int,
        default=16,
        metavar="<count>",
        help="Views to collect before stopping (default: 16).",
    )
    tuning.add_argument(
        "--min-views",
        type=int,
        default=8,
        metavar="<count>",
        help="Fewest views Enter will accept (default: 8).",
    )
    tuning.add_argument(
        "--max-views",
        type=int,
        default=24,
        metavar="<count>",
        help="Stop here even when frame coverage is still short (default: 24).",
    )
    tuning.add_argument(
        "--min-cells",
        type=int,
        default=6,
        metavar="<count>",
        help="Of the 9 frame regions the sheet should visit (default: 6).",
    )
    tuning.add_argument(
        "--seed-fov",
        type=float,
        default=70.0,
        metavar="<degrees>",
        help="Vertical field of view used to start the fit (default: 70).",
    )
    tuning.add_argument(
        "--seconds",
        type=float,
        default=600.0,
        metavar="<seconds>",
        help="Stop capturing after this many seconds (default: 600).",
    )
    tuning.add_argument(
        "--no-window",
        action="store_true",
        help="Skip the preview window. A jpg is written beside the log.",
    )
    return parser
