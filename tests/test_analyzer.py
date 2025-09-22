import numpy as np
import pytest

from froth_velocity.analyzer import FrothVelocityAnalyzer
from froth_velocity.utils import RegionOfInterest, direction_from_components


def _generate_pattern(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    pattern = (
        np.sin(x / 5.0)
        + np.cos(y / 7.0)
        + np.sin((x + y) / 11.0)
        + np.cos((x - y) / 13.0)
    )
    pattern -= pattern.min()
    pattern /= pattern.max()
    return (pattern * 255).astype(np.uint8)


def _generate_frames(frame_count: int, shift: tuple[int, int]) -> list[np.ndarray]:
    base = _generate_pattern()
    frames = []
    current = base
    for _ in range(frame_count):
        frames.append(np.dstack([current] * 3))
        current = np.roll(current, shift[0], axis=1)
        current = np.roll(current, shift[1], axis=0)
    return frames


def test_analyze_frames_recovers_expected_velocity():
    frames = _generate_frames(8, (1, 0))
    direction = direction_from_components((1, 0))
    roi = RegionOfInterest(0, 0, frames[0].shape[1], frames[0].shape[0])
    analyzer = FrothVelocityAnalyzer(direction=direction, roi=roi)

    result = analyzer.analyze_frames(frames, fps=5.0)

    assert len(result.measurements) == len(frames) - 1
    expected_velocity = 5.0  # 1 pixel per frame at 5 FPS
    assert result.mean_velocity == pytest.approx(expected_velocity, rel=0.15, abs=0.5)
    assert result.median_velocity == pytest.approx(expected_velocity, rel=0.15, abs=0.5)


def test_analyze_frames_without_fps_reports_per_frame_velocity():
    frames = _generate_frames(5, (0, 1))
    direction = direction_from_components((0, 1))
    roi = RegionOfInterest(0, 0, frames[0].shape[1], frames[0].shape[0])
    analyzer = FrothVelocityAnalyzer(direction=direction, roi=roi)

    result = analyzer.analyze_frames(frames, fps=None)

    assert result.fps is None
    for measurement in result.measurements:
        assert measurement.mean_velocity == pytest.approx(
            measurement.mean_flow_per_frame, rel=1e-6
        )


def test_analyze_frames_roi_outside_frame_raises():
    frames = _generate_frames(3, (1, 0))
    direction = direction_from_components((1, 0))
    roi = RegionOfInterest(1000, 1000, 10, 10)
    analyzer = FrothVelocityAnalyzer(direction=direction, roi=roi)

    with pytest.raises(ValueError):
        analyzer.analyze_frames(frames, fps=5.0)
