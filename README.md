# Froth Velocity Measurement Toolkit

This project provides a Python-based machine vision pipeline for measuring
the velocity of froth travelling across a flotation cell using OpenCV optical
flow analysis. It offers a reusable analysis API as well as a command-line
interface that can process recorded video, compute per-frame flow statistics,
and optionally visualise or export the results.

## Features

- Rectangular region-of-interest (ROI) specification to focus processing on the
  area of the froth surface.
- Direction-aware velocity measurements that project optical flow vectors onto
  a user-specified flow direction.
- CSV export of per-frame velocities and summary statistics.
- Optional annotated preview window and MP4 video writer for quick validation.
- Configurable Farneback optical-flow parameters for tuning against different
  video conditions.
- Unit-tested core utilities and analysis logic to simplify further extension.

## Installation

1. Create a virtual environment (recommended) and install the runtime
dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. If you plan to run the automated test suite, install the additional
development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

## Command-line usage

Analyse a video by providing the input path, ROI and flow direction. The
following example assumes the froth flows from left to right across an ROI that
starts at pixel `(120, 80)` with a size of `320 × 180` pixels:

```bash
python -m froth_velocity.cli \
  --video data/froth_sample.mp4 \
  --roi 120 80 320 180 \
  --direction 1 0 \
  --print-details \
  --output-csv results/froth_velocities.csv
```

Key options:

- `--direction dx dy` or `--direction-angle degrees` specify the dominant flow
  direction. The direction is normalised automatically.
- `--fps` overrides the frames-per-second metadata when the video container
  lacks this information.
- `--visualize` opens an annotated preview window. Press `q` or `Esc` to stop.
- `--output-video path.mp4` writes an annotated MP4 file (requires a valid FPS).
- `--flow-*` options adjust the Farneback optical flow parameters, enabling you
  to fine-tune the analysis for challenging footage.

The tool prints a summary of the mean and median projected velocities. When the
FPS is known, the units are reported in `px/s`; otherwise, per-frame velocities
(`px/frame`) are reported.

## Python API

The `FrothVelocityAnalyzer` class under `froth_velocity.analyzer` exposes the
core functionality programmatically. It accepts NumPy frames directly, which is
useful for integration into larger pipelines or for unit testing with synthetic
video data.

Example snippet:

```python
from froth_velocity.analyzer import FrothVelocityAnalyzer
from froth_velocity.utils import RegionOfInterest, direction_from_components

roi = RegionOfInterest(100, 50, 300, 200)
direction = direction_from_components((1, 0))
analyzer = FrothVelocityAnalyzer(direction=direction, roi=roi)

result = analyzer.analyze_video("froth.mp4", fps_override=None)
print(result.mean_velocity, result.units())
```

## Running tests

The repository includes a small pytest suite that validates the ROI/direction
parsers and the velocity estimator against synthetic translations. Execute the
following command from the project root after installing development
dependencies:

```bash
pytest
```

## License

This project is provided for demonstration purposes. Review and adjust it to
match your deployment and licensing requirements before using it in production.
