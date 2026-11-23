# ChArUco lens calibration

Calibrate one camera from an RTSP stream of a printed ChArUco sheet. The log stores focal length, k1, and k2 for the ground-plane overlay.

## Quick start

1. Print the sheet with `tool-charuco-board-generator`.
2. From this directory, install and run:

```bash
uv sync
uv run charuco-calib --rtsp <rtsp-url> --camera <camera-name>
```

3. Hold the sheet still until a view is accepted. Tilt it or move it so it covers the center and the corners.
4. Press Enter to finish, or Esc to cancel.
5. Copy `focal_length`, `k1`, and `k2` from `outputs/<camera-name>.yml` into the overlay.

Lock zoom first, and use the full frame. A different zoom is a different log. On a PTZ, `fov_deg` is the vertical field of view. On a fixed camera, pass `--fov-axis diagonal`.

## Options

| Option | Meaning |
| --- | --- |
| `--rtsp <rtsp-url>` | Camera stream. `RTSP_URL` is used when this is omitted. |
| `--camera <camera-name>` | Name stored in the log. The file is `outputs/<camera-name>.yml`. |
| `--spec <board.yml>` | Printed-sheet spec. Default: `../tool-charuco-board-generator/outputs/charuco-board.yml`. |
| `--output <file>` | Log path. A bare filename is written under `outputs/`. |
| `--fov-axis <axis>` | Field of view stored as `fov_deg`: `vertical` (default), `horizontal`, or `diagonal`. |
| `--views <count>` | Views to collect before stopping. Default: 16. |
| `--min-views <count>` | Fewest views Enter will accept. Default: 8. |
| `--max-views <count>` | Stop here even when coverage is still short. Default: 24. |
| `--min-cells <count>` | Of the 9 frame regions the sheet should visit. Default: 6. |
| `--seed-fov <degrees>` | Vertical field of view used to start the fit. Default: 70. |
| `--seconds <seconds>` | Stop capturing after this many seconds. Default: 600. |
| `--no-window` | Skip the preview window. A jpg is written beside the log. |

```bash
uv run charuco-calib --rtsp <rtsp-url> --camera <camera-name> --spec <board.yml> --output <file>
```

Without a display, progress is printed and a preview jpg is updated beside the log.

The log also keeps an unconstrained fit. The overlay ignores it. A warning there means the locked k1 and k2 cannot absorb a shift of the principal point.
