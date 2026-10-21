"""Display MediaPipe Pose landmarks on a webcam or video file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render MediaPipe Pose landmarks on a webcam or video."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Video path or webcam index (default: 0).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror webcam frames.",
    )
    return parser.parse_args()


def open_capture(source_arg: str) -> tuple[cv2.VideoCapture, bool]:
    """Open a webcam index or a video path and report whether it is a webcam."""
    is_camera = source_arg.isdigit()
    source: int | str = int(source_arg) if is_camera else source_arg

    if not is_camera and not Path(source_arg).is_file():
        raise FileNotFoundError(f"Video file not found: {source_arg}")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {source_arg}")

    return capture, is_camera


def main() -> None:
    args = parse_args()
    capture, is_camera = open_capture(args.source)

    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_module = mp.solutions.pose
    previous_time = time.perf_counter()

    try:
        with pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break

                if is_camera and not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                # MediaPipe expects RGB; OpenCV captures frames in BGR order.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                result = pose.process(rgb_frame)

                if result.pose_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        pose_module.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                    )

                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("MediaPipe Pose - press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
