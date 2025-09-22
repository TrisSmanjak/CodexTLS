"""Froth velocity measurement package.

This package provides tools for measuring the motion of froth in
flotation videos using OpenCV optical flow algorithms.
"""

from .analyzer import (
    FrameVelocity,
    FrothVelocityAnalyzer,
    FrothVelocityResult,
)

__all__ = [
    "FrameVelocity",
    "FrothVelocityAnalyzer",
    "FrothVelocityResult",
]
