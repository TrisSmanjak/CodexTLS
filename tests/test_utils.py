import math

import pytest

from froth_velocity.utils import (
    RegionOfInterest,
    clamp_roi_to_frame,
    direction_from_angle,
    direction_from_components,
    parse_roi,
)


def test_parse_roi_success():
    roi = parse_roi([10, 20, 30, 40])
    assert roi == RegionOfInterest(10, 20, 30, 40)


def test_parse_roi_invalid_dimensions():
    with pytest.raises(ValueError):
        parse_roi([0, 0, -5, 10])


def test_direction_from_components_normalizes():
    direction = direction_from_components([3, 4])
    assert math.isclose(direction.x ** 2 + direction.y ** 2, 1.0, rel_tol=1e-6)
    assert math.isclose(direction.x, 3 / 5, rel_tol=1e-6)
    assert math.isclose(direction.y, 4 / 5, rel_tol=1e-6)


def test_direction_from_angle_zero_degrees():
    direction = direction_from_angle(0)
    assert math.isclose(direction.x, 1.0, rel_tol=1e-6)
    assert math.isclose(direction.y, 0.0, rel_tol=1e-6)


def test_clamp_roi_to_frame_adjusts_boundaries():
    roi = RegionOfInterest(5, 5, 10, 10)
    clamped = clamp_roi_to_frame(roi, (12, 12))
    assert clamped == RegionOfInterest(5, 5, 7, 7)


def test_clamp_roi_to_frame_outside_raises():
    roi = RegionOfInterest(100, 100, 10, 10)
    with pytest.raises(ValueError):
        clamp_roi_to_frame(roi, (50, 50))
