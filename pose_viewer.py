"""Display MediaPipe Pose landmarks on a webcam or video file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(first_landmark, middle_landmark, last_landmark) -> float:
    """Return the 0-180 degree angle at the middle MediaPipe landmark."""
    first = np.array([first_landmark.x, first_landmark.y], dtype=np.float64)
    middle = np.array([middle_landmark.x, middle_landmark.y], dtype=np.float64)
    last = np.array([last_landmark.x, last_landmark.y], dtype=np.float64)

    angle = np.degrees(
        np.arctan2(*(last - middle)[::-1])
        - np.arctan2(*(first - middle)[::-1])
    )
    angle = abs(angle)
    return float(360.0 - angle if angle > 180.0 else angle)


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

    if not hasattr(mp, "solutions"):
        capture.release()
        raise RuntimeError(
            "This viewer requires MediaPipe 0.10.21. Reinstall the project "
            "dependencies with: python -m pip install --force-reinstall "
            "-r requirements.txt"
        )

    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_module = mp.solutions.pose
    previous_time = time.perf_counter()
    rep_count = 0
    reached_curl_position = False
    curl_in_progress = False
    shoulder_baseline: float | None = None

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
                    landmarks = result.pose_landmarks.landmark
                    right_shoulder = landmarks[
                        pose_module.PoseLandmark.RIGHT_SHOULDER.value
                    ]
                    right_elbow = landmarks[pose_module.PoseLandmark.RIGHT_ELBOW.value]
                    right_wrist = landmarks[pose_module.PoseLandmark.RIGHT_WRIST.value]
                    right_hip = landmarks[pose_module.PoseLandmark.RIGHT_HIP.value]
                    elbow_angle = calculate_angle(
                        right_shoulder, right_elbow, right_wrist
                    )
                    shoulder_angle = calculate_angle(
                        right_hip, right_shoulder, right_elbow
                    )

                    # Arm the counter at the top of the curl, then count only
                    # after the arm returns to full extension.
                    if elbow_angle < 30.0:
                        reached_curl_position = True
                        curl_in_progress = True
                    elif elbow_angle > 160.0:
                        if reached_curl_position:
                            rep_count += 1
                        reached_curl_position = False
                        curl_in_progress = False
                        shoulder_baseline = shoulder_angle
                    elif shoulder_baseline is not None:
                        curl_in_progress = True

                    if shoulder_baseline is None:
                        shoulder_baseline = shoulder_angle

                    shoulder_movement = abs(shoulder_angle - shoulder_baseline)
                    bad_form = curl_in_progress and shoulder_movement > 20.0
                    form_text = "Keep upper arm still!" if bad_form else "Good form"
                    form_color = (0, 0, 255) if bad_form else (0, 255, 0)

                    drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        pose_module.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                    )

                    frame_height, frame_width = frame.shape[:2]
                    elbow_position = (
                        max(0, min(frame_width - 1, int(right_elbow.x * frame_width))),
                        max(30, min(frame_height - 1, int(right_elbow.y * frame_height))),
                    )
                    cv2.putText(
                        frame,
                        f"{elbow_angle:.0f} deg",
                        elbow_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Reps: {rep_count}",
                        (12, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        form_text,
                        (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        form_color,
                        2,
                        cv2.LINE_AA,
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
