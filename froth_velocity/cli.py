"""Command line entry point for froth velocity analysis."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .analyzer import FrothVelocityAnalyzer
from .utils import direction_from_angle, direction_from_components, parse_roi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate froth velocity over time using optical flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        required=True,
        help="Rectangular region of interest specified as x y width height.",
    )

    direction_group = parser.add_mutually_exclusive_group(required=True)
    direction_group.add_argument(
        "--direction",
        nargs=2,
        type=float,
        metavar=("DX", "DY"),
        help="Flow direction vector components.",
    )
    direction_group.add_argument(
        "--direction-angle",
        type=float,
        help="Flow direction expressed as an angle in degrees (0° = +x axis).",
    )

    parser.add_argument(
        "--fps",
        type=float,
        help="Override the video FPS if metadata is missing or incorrect.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process at most this many frame transitions (pairs of frames).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save per-frame velocity measurements as CSV.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Optional path to write an annotated video (MP4 format recommended).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display an annotated preview window while processing the video.",
    )
    parser.add_argument(
        "--print-details",
        action="store_true",
        help="Print per-frame velocity statistics to stdout.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.3,
        help="Scale of the directional arrow relative to the ROI size.",
    )

    parser.add_argument(
        "--flow-pyr-scale",
        type=float,
        help="Override the Farneback pyramid scale parameter.",
    )
    parser.add_argument(
        "--flow-levels",
        type=int,
        help="Override the number of pyramid levels used by Farneback.",
    )
    parser.add_argument(
        "--flow-winsize",
        type=int,
        help="Override the window size parameter for Farneback.",
    )
    parser.add_argument(
        "--flow-iterations",
        type=int,
        help="Override the number of Farneback iterations per pyramid level.",
    )
    parser.add_argument(
        "--flow-poly-n",
        type=int,
        help="Override the neighborhood size used for polynomial expansion.",
    )
    parser.add_argument(
        "--flow-poly-sigma",
        type=float,
        help="Override the Gaussian sigma for polynomial expansion.",
    )

    return parser


def _flow_params_from_args(args: argparse.Namespace) -> Optional[dict]:
    params = {}
    if args.flow_pyr_scale is not None:
        params["pyr_scale"] = args.flow_pyr_scale
    if args.flow_levels is not None:
        params["levels"] = args.flow_levels
    if args.flow_winsize is not None:
        params["winsize"] = args.flow_winsize
    if args.flow_iterations is not None:
        params["iterations"] = args.flow_iterations
    if args.flow_poly_n is not None:
        params["poly_n"] = args.flow_poly_n
    if args.flow_poly_sigma is not None:
        params["poly_sigma"] = args.flow_poly_sigma
    return params or None


def _resolve_direction(args: argparse.Namespace):
    if args.direction is not None:
        return direction_from_components(args.direction)
    if args.direction_angle is not None:
        return direction_from_angle(args.direction_angle)
    raise ValueError("A direction specification is required")


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.fps is not None and args.fps <= 0:
        parser.error("FPS override must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")
    if args.arrow_scale <= 0:
        parser.error("arrow-scale must be positive")


def _write_csv(path: Path, result) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "timestamp_s",
                "mean_velocity",
                "median_velocity",
                "mean_flow_per_frame",
                "median_flow_per_frame",
            ]
        )
        for measurement in result.measurements:
            writer.writerow(
                [
                    measurement.frame_index,
                    "" if measurement.timestamp_s is None else f"{measurement.timestamp_s:.6f}",
                    f"{measurement.mean_velocity:.6f}",
                    f"{measurement.median_velocity:.6f}",
                    f"{measurement.mean_flow_per_frame:.6f}",
                    f"{measurement.median_flow_per_frame:.6f}",
                ]
            )


def _print_summary(result) -> None:
    units = result.units()
    print("\nSummary")
    print("-------")
    print(f"Frames processed: {len(result.measurements)}")
    if result.fps:
        print(f"Effective FPS: {result.fps:.3f}")
    print(f"Mean velocity: {result.mean_velocity:.3f} {units}")
    print(f"Median velocity: {result.median_velocity:.3f} {units}")


def _print_details(result) -> None:
    units = result.units()
    header = f"{'Frame':>8} {'Time (s)':>12} {'Mean':>14} {'Median':>14}"
    print("\nPer-frame measurements")
    print("----------------------")
    print(header)
    print("-" * len(header))
    for measurement in result.measurements:
        timestamp = "-" if measurement.timestamp_s is None else f"{measurement.timestamp_s:.3f}"
        print(
            f"{measurement.frame_index:8d} {timestamp:>12} "
            f"{measurement.mean_velocity:14.3f} {measurement.median_velocity:14.3f} {units}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    try:
        roi = parse_roi(args.roi)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        direction = _resolve_direction(args)
    except ValueError as exc:
        parser.error(str(exc))
    flow_params = _flow_params_from_args(args)

    analyzer = FrothVelocityAnalyzer(direction=direction, roi=roi, flow_params=flow_params)

    frame_callback = None
    window_name = "Froth Velocity"
    video_writer: Optional[cv2.VideoWriter] = None

    if args.visualize or args.output_video:
        def handle_frame(frame_index, frame, measurement, _flow, fps):
            nonlocal video_writer
            annotated = analyzer.annotate_frame(frame, measurement, fps, arrow_scale=args.arrow_scale)

            if args.visualize:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise KeyboardInterrupt

            if args.output_video:
                if video_writer is None:
                    if fps is None or fps <= 0:
                        raise RuntimeError(
                            "Cannot write an output video because the input FPS is unknown."
                        )
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    height, width = annotated.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(args.output_video),
                        fourcc,
                        fps,
                        (width, height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Unable to open output video file: {args.output_video}")
                video_writer.write(annotated)

        frame_callback = handle_frame

    try:
        result = analyzer.analyze_video(
            args.video,
            fps_override=args.fps,
            max_frames=args.max_frames,
            frame_callback=frame_callback,
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        result = None
    finally:
        if video_writer is not None:
            video_writer.release()
        if args.visualize:
            cv2.destroyAllWindows()

    if result is None:
        return 1

    _print_summary(result)
    if args.print_details:
        _print_details(result)

    if args.output_csv:
        _write_csv(args.output_csv, result)
        print(f"Per-frame measurements saved to {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
