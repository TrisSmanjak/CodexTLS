"""Core froth velocity analysis logic."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Callable, Iterable, Iterator, List, Optional

import cv2
import numpy as np

from .utils import (
    Direction,
    RegionOfInterest,
    clamp_roi_to_frame,
    units_label,
)


DEFAULT_FLOW_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 5,
    "poly_n": 5,
    "poly_sigma": 1.1,
    "flags": 0,
}


@dataclass
class FrameVelocity:
    """Velocity statistics for a single processed frame."""

    frame_index: int
    timestamp_s: Optional[float]
    mean_velocity: float
    median_velocity: float
    mean_flow_per_frame: float
    median_flow_per_frame: float


@dataclass
class FrothVelocityResult:
    """Aggregated velocity measurements for an entire video."""

    measurements: List[FrameVelocity]
    roi: RegionOfInterest
    direction: Direction
    fps: Optional[float]
    mean_velocity: float
    median_velocity: float

    def units(self) -> str:
        """Return the textual representation of the velocity units."""

        return units_label(self.fps)


FrameCallback = Callable[[int, np.ndarray, Optional[FrameVelocity], Optional[np.ndarray], Optional[float]], None]


class FrothVelocityAnalyzer:
    """Analyze froth motion within a video using optical flow."""

    def __init__(
        self,
        direction: Direction,
        roi: RegionOfInterest,
        flow_params: Optional[dict] = None,
    ) -> None:
        self.direction = direction
        self.roi = roi
        self.flow_params = dict(DEFAULT_FLOW_PARAMS)
        if flow_params:
            self.flow_params.update(flow_params)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError("Unsupported frame format: expected grayscale or BGR image")

    def _validate_roi(self, frame: np.ndarray) -> RegionOfInterest:
        gray = self._to_gray(frame)
        clamped = clamp_roi_to_frame(self.roi, (gray.shape[0], gray.shape[1]))
        if (clamped.width, clamped.height) != (self.roi.width, self.roi.height) or (
            clamped.x != self.roi.x or clamped.y != self.roi.y
        ):
            raise ValueError("ROI must lie within the frame dimensions")
        return clamped

    def analyze_video(
        self,
        video_path: str,
        fps_override: Optional[float] = None,
        max_frames: Optional[int] = None,
        frame_callback: Optional[FrameCallback] = None,
    ) -> FrothVelocityResult:
        """Analyze a video file located on disk."""

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        try:
            fps = fps_override
            if fps is None:
                fps = capture.get(cv2.CAP_PROP_FPS) or None
                if fps is not None and fps <= 0:
                    fps = None

            frame_iter = self._capture_frames(capture)
            result = self.analyze_frames(
                frame_iter,
                fps=fps,
                max_frames=max_frames,
                frame_callback=frame_callback,
            )
        finally:
            capture.release()
        return result

    @staticmethod
    def _capture_frames(capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield frame

    def analyze_frames(
        self,
        frames: Iterable[np.ndarray],
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        frame_callback: Optional[FrameCallback] = None,
    ) -> FrothVelocityResult:
        """Analyze an iterable of frames."""

        frame_iterator = iter(frames)
        try:
            first_frame = next(frame_iterator)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise ValueError("At least one frame is required") from exc

        roi = self._validate_roi(first_frame)
        direction_vector = np.array(self.direction.as_tuple(), dtype=np.float32)
        if frame_callback:
            frame_callback(0, first_frame, None, None, fps)

        prev_gray = self._to_gray(first_frame)
        prev_roi = prev_gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

        measurements: List[FrameVelocity] = []
        processed = 0

        for frame_index, frame in enumerate(frame_iterator, start=1):
            if max_frames is not None and processed >= max_frames:
                break

            gray = self._to_gray(frame)
            curr_roi = gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

            flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, **self.flow_params)
            directional_flow = flow[..., 0] * direction_vector[0] + flow[..., 1] * direction_vector[1]

            mean_flow = float(np.mean(directional_flow))
            median_flow = float(np.median(directional_flow))
            velocity_scale = fps if fps and fps > 0 else 1.0

            measurement = FrameVelocity(
                frame_index=frame_index,
                timestamp_s=(frame_index / fps) if fps and fps > 0 else None,
                mean_velocity=mean_flow * velocity_scale,
                median_velocity=median_flow * velocity_scale,
                mean_flow_per_frame=mean_flow,
                median_flow_per_frame=median_flow,
            )
            measurements.append(measurement)
            processed += 1

            if frame_callback:
                frame_callback(frame_index, frame, measurement, flow, fps)

            prev_roi = curr_roi

        aggregate_mean = mean([m.mean_velocity for m in measurements]) if measurements else 0.0
        aggregate_median = median([m.median_velocity for m in measurements]) if measurements else 0.0

        return FrothVelocityResult(
            measurements=measurements,
            roi=roi,
            direction=self.direction,
            fps=fps,
            mean_velocity=aggregate_mean,
            median_velocity=aggregate_median,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        measurement: Optional[FrameVelocity],
        fps: Optional[float],
        arrow_scale: float = 0.3,
    ) -> np.ndarray:
        """Draw ROI and velocity information onto a frame."""

        annotated = frame.copy()
        x, y, width, height = self.roi.as_tuple()
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)

        center = (int(x + width / 2), int(y + height / 2))
        magnitude = int(max(width, height) * arrow_scale)
        end_point = (
            int(center[0] + self.direction.x * magnitude),
            int(center[1] + self.direction.y * magnitude),
        )
        cv2.arrowedLine(annotated, center, end_point, (0, 0, 255), 2, tipLength=0.3)

        if measurement is not None:
            units = units_label(fps)
            base_y = max(30, y - 10)
            text_lines = [
                f"Frame {measurement.frame_index}",
                f"Mean: {measurement.mean_velocity:.2f} {units}",
                f"Median: {measurement.median_velocity:.2f} {units}",
            ]
            for idx, line in enumerate(text_lines):
                position = (x, base_y + idx * 20)
                cv2.putText(
                    annotated,
                    line,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        return annotated
