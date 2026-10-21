"""Interactive Streamlit dashboard for video-based exercise form analysis."""

from __future__ import annotations

import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from pose_viewer import calculate_angle


@dataclass
class CurlState:
    reps: int = 0
    reached_curl: bool = False
    curl_in_progress: bool = False
    shoulder_baseline: float | None = None
    bad_form_frames: int = 0
    analyzed_frames: int = 0


def analyze_curl(
    landmarks,
    pose_module,
    state: CurlState,
    curled_threshold: float,
    extended_threshold: float,
    shoulder_tolerance: float,
) -> tuple[float, float, bool]:
    """Update curl state and return elbow angle, shoulder movement, and form."""
    right_shoulder = landmarks[pose_module.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[pose_module.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[pose_module.PoseLandmark.RIGHT_WRIST.value]
    right_hip = landmarks[pose_module.PoseLandmark.RIGHT_HIP.value]

    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

    if elbow_angle < curled_threshold:
        state.reached_curl = True
        state.curl_in_progress = True
    elif elbow_angle > extended_threshold:
        if state.reached_curl:
            state.reps += 1
        state.reached_curl = False
        state.curl_in_progress = False
        state.shoulder_baseline = shoulder_angle
    elif state.shoulder_baseline is not None:
        state.curl_in_progress = True

    if state.shoulder_baseline is None:
        state.shoulder_baseline = shoulder_angle

    shoulder_movement = abs(shoulder_angle - state.shoulder_baseline)
    bad_form = state.curl_in_progress and shoulder_movement > shoulder_tolerance
    state.analyzed_frames += 1
    if bad_form:
        state.bad_form_frames += 1

    return elbow_angle, shoulder_movement, bad_form


def draw_panel(
    frame,
    reps: int,
    elbow_angle: float | None,
    shoulder_movement: float | None,
    bad_form: bool,
) -> None:
    """Draw a readable form-feedback panel over a frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (390, 160), (12, 18, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(
        frame, f"REPS  {reps}", (34, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if elbow_angle is not None:
        cv2.putText(
            frame, f"ELBOW  {elbow_angle:.0f} deg", (34, 91),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA,
        )
    if shoulder_movement is not None:
        cv2.putText(
            frame, f"SHOULDER  {shoulder_movement:.0f} deg", (34, 122),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (205, 205, 205), 2, cv2.LINE_AA,
        )

    message = "KEEP UPPER ARM STILL!" if bad_form else "GOOD FORM"
    color = (45, 45, 245) if bad_form else (80, 220, 110)
    cv2.putText(
        frame, message, (34, 150), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, color, 2, cv2.LINE_AA,
    )


class LiveCurlProcessor(VideoProcessorBase):
    """Process browser webcam frames while preserving one curl session's state."""

    def __init__(
        self,
        curled_threshold: float,
        extended_threshold: float,
        shoulder_tolerance: float,
    ) -> None:
        self.curled_threshold = curled_threshold
        self.extended_threshold = extended_threshold
        self.shoulder_tolerance = shoulder_tolerance
        self.state = CurlState()
        self.pose_module = mp.solutions.pose
        self.pose = self.pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.lock = threading.Lock()

    def recv(self, video_frame: av.VideoFrame) -> av.VideoFrame:
        frame = video_frame.to_ndarray(format="bgr24")
        elbow_angle: float | None = None
        shoulder_movement: float | None = None
        bad_form = False

        with self.lock:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            result = self.pose.process(rgb_frame)

            if result.pose_landmarks:
                elbow_angle, shoulder_movement, bad_form = analyze_curl(
                    result.pose_landmarks.landmark,
                    self.pose_module,
                    self.state,
                    self.curled_threshold,
                    self.extended_threshold,
                    self.shoulder_tolerance,
                )
                self.drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    self.pose_module.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
                )

            draw_panel(
                frame,
                self.state.reps,
                elbow_angle,
                shoulder_movement,
                bad_form,
            )

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

    def __del__(self) -> None:
        pose = getattr(self, "pose", None)
        if pose is not None:
            pose.close()


def make_browser_video(raw_path: Path, browser_path: Path) -> Path:
    """Convert OpenCV's output to browser-friendly H.264 when possible."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw_path),
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(browser_path),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    return browser_path if result.returncode == 0 and browser_path.exists() else raw_path


def process_video(
    video_bytes: bytes,
    curled_threshold: float,
    extended_threshold: float,
    shoulder_tolerance: float,
) -> tuple[bytes, CurlState]:
    """Analyze an uploaded video and return annotated MP4 bytes and metrics."""
    pose_module = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    state = CurlState()

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "upload.mp4"
        raw_path = temp_dir / "annotated_raw.mp4"
        browser_path = temp_dir / "annotated.mp4"
        input_path.write_bytes(video_bytes)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("The uploaded file could not be opened as a video.")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create the processed video file.")

        progress = st.progress(0, text="Preparing analysis…")
        elbow_angle: float | None = None
        shoulder_movement: float | None = None
        bad_form = False
        frame_number = 0

        try:
            with pose_module.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    result = pose.process(rgb_frame)

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks.landmark
                        elbow_angle, shoulder_movement, bad_form = analyze_curl(
                            landmarks,
                            pose_module,
                            state,
                            curled_threshold,
                            extended_threshold,
                            shoulder_tolerance,
                        )
                        drawing.draw_landmarks(
                            frame,
                            result.pose_landmarks,
                            pose_module.POSE_CONNECTIONS,
                            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
                        )

                    draw_panel(
                        frame, state.reps, elbow_angle, shoulder_movement, bad_form
                    )
                    writer.write(frame)
                    frame_number += 1
                    if frame_number % 5 == 0 or frame_number == total_frames:
                        progress.progress(
                            min(frame_number / total_frames, 1.0),
                            text=f"Analyzing frame {frame_number:,} of {total_frames:,}",
                        )
        finally:
            capture.release()
            writer.release()

        progress.progress(1.0, text="Analysis complete")
        final_path = make_browser_video(raw_path, browser_path)
        return final_path.read_bytes(), state


def render_app() -> None:
    st.set_page_config(
        page_title="FormCheck Studio", page_icon="🏋️", layout="wide"
    )
    st.markdown(
        """
        <style>
        .stApp { background: #090d14; }
        [data-testid="stSidebar"] { background: #101722; }
        .hero {
            padding: 1.4rem 1.6rem; border: 1px solid #263143;
            border-radius: 18px; background: linear-gradient(135deg, #111b29, #0d121b);
            margin-bottom: 1.4rem;
        }
        .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -0.04em; }
        .hero p { color: #aeb9c9; margin: .45rem 0 0; }
        .tag {
            display: inline-block; padding: .28rem .65rem; margin: .25rem .2rem 0 0;
            border-radius: 999px; border: 1px solid #334155; color: #aeb9c9;
            font-size: .78rem;
        }
        </style>
        <div class="hero">
          <h1>FormCheck Studio</h1>
          <p>Upload a lift. See every rep, joint angle, and form cue directly on video.</p>
          <span class="tag">Pose tracking</span>
          <span class="tag">Rep counting</span>
          <span class="tag">Form feedback</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis setup")
        exercise = st.selectbox(
            "Exercise",
            ["Bicep Curl", "Squat — coming soon", "Bench Press — coming soon",
             "Running — coming soon", "Deadlift — coming soon"],
        )
        input_mode = st.radio(
            "Input source", ["Upload MP4", "Live camera"], horizontal=True
        )
        st.caption("Current tracker analyzes the anatomical right arm.")
        st.divider()
        st.subheader("Curl thresholds")
        curled_threshold = st.slider("Curled angle", 15, 60, 30, format="%d°")
        extended_threshold = st.slider(
            "Extended angle", 130, 175, 160, format="%d°"
        )
        shoulder_tolerance = st.slider(
            "Shoulder tolerance", 5, 40, 20, format="%d°"
        )
        st.info("Start the video with your arm extended to establish a form baseline.")

    if exercise != "Bicep Curl":
        st.info(
            "This analyzer is on the roadmap. Choose **Bicep Curl** to process a video now."
        )
        return

    if input_mode == "Live camera":
        st.subheader("Live bicep curl analysis")
        st.write(
            "Select **START**, allow browser camera access, and begin with your "
            "right arm extended. Your skeleton, rep count, and form cue appear "
            "directly on the live feed."
        )
        webrtc_streamer(
            key="formcheck-live-curl",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: LiveCurlProcessor(
                float(curled_threshold),
                float(extended_threshold),
                float(shoulder_tolerance),
            ),
            async_processing=True,
        )
        st.caption(
            "Stopping and restarting the camera begins a new session and resets the rep count."
        )
        return

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader("Upload workout video")
        uploaded = st.file_uploader(
            "MP4 video", type=["mp4"], help="A side-on view gives the clearest joint angles."
        )
        if uploaded:
            st.video(uploaded.getvalue())

    with right:
        st.subheader("How to record")
        st.markdown(
            "1. Keep your full right side visible.\n"
            "2. Begin with the arm fully extended.\n"
            "3. Keep the camera steady and well lit.\n"
            "4. Complete each curl below 30° and above 160°."
        )
        process_clicked = st.button(
            "Analyze video", type="primary", use_container_width=True,
            disabled=uploaded is None,
        )

    if process_clicked and uploaded is not None:
        try:
            output_bytes, state = process_video(
                uploaded.getvalue(),
                float(curled_threshold),
                float(extended_threshold),
                float(shoulder_tolerance),
            )
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        else:
            st.session_state["output_video"] = output_bytes
            st.session_state["curl_state"] = state

    if "output_video" in st.session_state:
        state = st.session_state["curl_state"]
        st.divider()
        st.subheader("Analysis results")
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Reps completed", state.reps)
        bad_form_rate = (
            100 * state.bad_form_frames / state.analyzed_frames
            if state.analyzed_frames else 0
        )
        metric_two.metric("Bad-form frames", f"{bad_form_rate:.1f}%")
        metric_three.metric("Frames analyzed", f"{state.analyzed_frames:,}")
        st.video(st.session_state["output_video"])
        st.download_button(
            "Download analyzed video",
            st.session_state["output_video"],
            file_name="formcheck_bicep_curl.mp4",
            mime="video/mp4",
        )


if __name__ == "__main__":
    render_app()
