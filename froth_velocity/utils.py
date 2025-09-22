"""Utility helpers for froth velocity analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RegionOfInterest:
    """Simple representation of a rectangular region of interest.

    Attributes
    ----------
    x, y : int
        Top-left coordinate of the ROI.
    width, height : int
        Size of the ROI in pixels.
    """

    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return the ROI as an ``(x, y, width, height)`` tuple."""

        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class Direction:
    """Represents a normalized 2D direction vector."""

    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


def parse_roi(values: Sequence[int]) -> RegionOfInterest:
    """Validate and convert a ROI specification.

    Parameters
    ----------
    values:
        Sequence with four integers ``(x, y, width, height)``.

    Returns
    -------
    RegionOfInterest
        Parsed ROI data structure.

    Raises
    ------
    ValueError
        If the ROI specification is invalid.
    """

    if len(values) != 4:
        raise ValueError("ROI must have four integers: x y width height")

    x, y, width, height = map(int, values)
    if x < 0 or y < 0:
        raise ValueError("ROI coordinates must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")

    return RegionOfInterest(x=x, y=y, width=width, height=height)


def _normalize_direction(dx: float, dy: float) -> Direction:
    vector = np.array([dx, dy], dtype=float)
    norm = np.linalg.norm(vector)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Direction vector must have a non-zero magnitude")
    normalized = vector / norm
    return Direction(x=float(normalized[0]), y=float(normalized[1]))


def direction_from_components(components: Iterable[float]) -> Direction:
    """Create a direction vector from Cartesian components."""

    values = list(components)
    if len(values) != 2:
        raise ValueError("Direction must have two components: dx dy")
    dx, dy = map(float, values)
    return _normalize_direction(dx, dy)


def direction_from_angle(angle_degrees: float) -> Direction:
    """Create a direction vector from an angle in degrees."""

    radians = math.radians(float(angle_degrees))
    dx = math.cos(radians)
    dy = math.sin(radians)
    return _normalize_direction(dx, dy)


def units_label(fps: Optional[float]) -> str:
    """Return the textual representation of velocity units."""

    if fps and fps > 0:
        return "px/s"
    return "px/frame"


def clamp_roi_to_frame(roi: RegionOfInterest, frame_shape: Tuple[int, int]) -> RegionOfInterest:
    """Ensure the ROI lies within the provided frame dimensions.

    Parameters
    ----------
    roi:
        Region of interest to clamp.
    frame_shape:
        Tuple ``(height, width)`` describing the frame size.

    Returns
    -------
    RegionOfInterest
        Potentially adjusted ROI to fit inside the frame.

    Raises
    ------
    ValueError
        If the ROI does not intersect the frame at all.
    """

    frame_height, frame_width = frame_shape
    if roi.x >= frame_width or roi.y >= frame_height:
        raise ValueError("ROI is completely outside of the frame")

    x = max(roi.x, 0)
    y = max(roi.y, 0)
    width = min(roi.width, frame_width - x)
    height = min(roi.height, frame_height - y)

    if width <= 0 or height <= 0:
        raise ValueError("ROI does not intersect the frame")

    return RegionOfInterest(x=x, y=y, width=width, height=height)
