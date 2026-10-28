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
from squat_analysis import SquatFrame, SquatState, analyze_squat


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



def draw_squat_panel(frame, state: SquatState, analysis: SquatFrame) -> None:
    """Render squat metrics and the most useful current cue over the video."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (470, 204), (12, 18, 28), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    lines = [
        (f"REPS  {state.reps}", 1.0, (255, 255, 255)),
        (f"PHASE  {analysis.phase}", 0.62, (225, 230, 238)),
    ]
    if analysis.knee_flexion is not None:
        lines.append((f"KNEE FLEXION  {analysis.knee_flexion:.0f} deg", 0.62, (225, 230, 238)))
    if analysis.torso_lean is not None:
        lines.append((f"FORWARD LEAN  {analysis.torso_lean:.0f} deg", 0.58, (225, 230, 238)))
    if analysis.knee_tracking_deviation is not None:
        lines.append((f"KNEE TRACKING  {analysis.knee_tracking_deviation * 100:.0f}% shoulder width",
                      0.52, (225, 230, 238)))
    cue = analysis.cues[0] if analysis.cues else (
        "Hold upright briefly to calibrate" if analysis.knee_tracking_deviation is None and
        "Front view needed to assess knee tracking" not in analysis.cues else "Good so far"
    )
    y = 52
    for label, scale, color in lines:
        cv2.putText(frame, label, (34, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)
        y += 32
    cv2.putText(frame, cue[:48], (34, 184), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (45, 190, 255) if analysis.cues else (80, 220, 110),
                2, cv2.LINE_AA)


class LiveSquatProcessor(VideoProcessorBase):
    """Process live camera frames with an independent squat session state."""

    def __init__(self, camera_view: str, depth_target: float,
                 torso_lean_limit: float, knee_tracking_tolerance: float,
                 center_drift_tolerance: float) -> None:
        self.camera_view = camera_view
        self.depth_target = depth_target
        self.torso_lean_limit = torso_lean_limit
        self.knee_tracking_tolerance = knee_tracking_tolerance
        self.center_drift_tolerance = center_drift_tolerance
        self.state = SquatState()
        self.pose_module = mp.solutions.pose
        self.pose = self.pose_module.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            enable_segmentation=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.lock = threading.Lock()
        self.latest = SquatFrame(None, None, None, None, "STANDING",
                                 ("Waiting for pose",), False)

    def recv(self, video_frame: av.VideoFrame) -> av.VideoFrame:
        frame = video_frame.to_ndarray(format="bgr24")
        with self.lock:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            result = self.pose.process(rgb_frame)
            if result.pose_landmarks:
                self.latest = analyze_squat(
                    result.pose_landmarks.landmark, self.pose_module, self.state,
                    self.camera_view, self.depth_target, self.torso_lean_limit,
                    self.knee_tracking_tolerance, self.center_drift_tolerance,
                )
                self.drawing.draw_landmarks(
                    frame, result.pose_landmarks, self.pose_module.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
                )
            else:
                self.latest = SquatFrame(None, None, None, None, self.state.phase,
                                         ("Move back until your full body is visible",), False)
            draw_squat_panel(frame, self.state, self.latest)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

    def __del__(self) -> None:
        pose = getattr(self, "pose", None)
        if pose is not None:
            pose.close()


def process_squat_video(
    video_bytes: bytes, camera_view: str, depth_target: float,
    torso_lean_limit: float, knee_tracking_tolerance: float,
    center_drift_tolerance: float,
) -> tuple[bytes, SquatState]:
    """Analyze an uploaded squat clip and return annotated MP4 bytes and state."""
    pose_module = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    state = SquatState()

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path, raw_path, browser_path = (
            temp_dir / "upload.mp4", temp_dir / "squat_raw.mp4",
            temp_dir / "squat_annotated.mp4",
        )
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

        progress = st.progress(0, text="Preparing squat analysis…")
        latest = SquatFrame(None, None, None, None, "STANDING",
                            ("Waiting for pose",), False)
        frame_number = 0
        try:
            with pose_module.Pose(
                static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                enable_segmentation=False, min_detection_confidence=0.5,
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
                        latest = analyze_squat(
                            result.pose_landmarks.landmark, pose_module, state,
                            camera_view, depth_target, torso_lean_limit,
                            knee_tracking_tolerance, center_drift_tolerance,
                        )
                        drawing.draw_landmarks(
                            frame, result.pose_landmarks, pose_module.POSE_CONNECTIONS,
                            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
                        )
                    else:
                        latest = SquatFrame(None, None, None, None, state.phase,
                                            ("Keep your whole body in frame",), False)
                    draw_squat_panel(frame, state, latest)
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
        progress.progress(1.0, text="Squat analysis complete")
        final_path = make_browser_video(raw_path, browser_path)
        return final_path.read_bytes(), state


def render_squat_results(state: SquatState, output_key: str, filename: str) -> None:
    """Show clip summary and the annotated output for a completed squat analysis."""
    st.divider()
    st.subheader("Squat analysis results")
    cols = st.columns(3)
    cols[0].metric("Completed reps", state.reps)
    average_depth = sum(state.rep_depths) / len(state.rep_depths) if state.rep_depths else 0
    cols[1].metric("Average bottom flexion", f"{average_depth:.0f}°" if state.rep_depths else "—")
    cols[2].metric("Frames analyzed", f"{state.analyzed_frames:,}")
    st.caption(
        "Flexion bands: 70–100° is the initial half-squat reference; above 100° "
        "is a deeper range. These are 2D camera estimates, not medical measurements."
    )
    if state.rep_depths:
        depth_rows = [
            f"Rep {index}: {depth:.0f}° ({'deep range' if depth > 100 else 'half-squat reference'})"
            for index, depth in enumerate(state.rep_depths, start=1)
        ]
        st.write("Peak knee flexion by completed rep")
        st.bar_chart({"Peak knee flexion (degrees)": state.rep_depths})
        st.caption(" · ".join(depth_rows))
    else:
        st.info("No complete rep reached the selected depth and returned to standing.")
    if state.issue_frames:
        st.write("Frames over selected cue thresholds")
        issue_labels = {
            "torso": "Torso lean",
            "knee_tracking": "Knee tracking",
            "stability": "Center drift (2D)",
        }
        st.write(" · ".join(
            f"{issue_labels[key]}: {count / max(state.analyzed_frames, 1) * 100:.1f}%"
            for key, count in state.issue_frames.items()
        ))
    st.video(st.session_state[output_key])
    st.download_button(
        "Download analyzed video", st.session_state[output_key],
        file_name=filename, mime="video/mp4",
    )


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
    st.set_page_config(page_title="FormCheck Studio", page_icon="🏋️", layout="wide")
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
          <p>Analyze reps, joint angles, and movement cues from video.</p>
          <span class="tag">Full-body pose tracking</span>
          <span class="tag">Rep counting</span>
          <span class="tag">Form feedback</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis setup")
        exercise = st.selectbox("Exercise", ["Bicep Curl", "Squat"])
        input_mode = st.radio("Input source", ["Upload MP4", "Live camera"], horizontal=True)
        if exercise == "Bicep Curl":
            st.caption("The pose model detects all landmarks; this analyzer uses the right shoulder, elbow, wrist, and hip.")
            st.divider()
            st.subheader("Curl thresholds")
            curled_threshold = st.slider("Curled angle", 15, 60, 30, format="%d°")
            extended_threshold = st.slider("Extended angle", 130, 175, 160, format="%d°")
            shoulder_tolerance = st.slider("Shoulder tolerance", 5, 40, 20, format="%d°")
            st.info("Start with your arm extended to establish a shoulder baseline.")
        else:
            st.caption("MediaPipe finds the full 33-joint pose first. Squat checks use only the joints needed for the selected camera view.")
            camera_view = st.radio("Camera view", ["Side profile", "Front view"])
            depth_target = st.slider("Minimum knee flexion to count a rep", 70, 120, 70, format="%d°")
            torso_lean_limit = st.slider("Forward torso-lean cue", 30, 70, 50, format="%d°")
            knee_tracking_tolerance = st.slider("Knee tracking change", 5, 25, 10, format="%d%%") / 100
            center_drift_tolerance = st.slider("Center drift cue", 5, 30, 20, format="%d%%") / 100
            if camera_view == "Side profile":
                st.info("Tracks shoulder, hip, knee, ankle, heel, and toe. Best for rep count, depth, and torso angle. Front-view knee tracking needs a separate front clip.")
            else:
                st.info("Tracks both shoulders, hips, knees, and ankles. Hold a front-facing standing position briefly to calibrate knee tracking and center drift.")

    if exercise == "Bicep Curl":
        if input_mode == "Live camera":
            st.subheader("Live bicep curl analysis")
            st.write("Start with your right arm extended. The complete pose skeleton, rep count, elbow angle, and shoulder cue appear on the feed.")
            webrtc_streamer(
                key="formcheck-live-curl",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=lambda: LiveCurlProcessor(
                    float(curled_threshold), float(extended_threshold),
                    float(shoulder_tolerance),
                ),
                async_processing=True,
            )
            st.caption("Stopping and restarting the camera begins a new session.")
            return

        left, right = st.columns([1.35, 1], gap="large")
        with left:
            st.subheader("Upload workout video")
            uploaded = st.file_uploader("MP4 video", type=["mp4"], key="curl_upload")
            if uploaded:
                st.video(uploaded.getvalue())
        with right:
            st.subheader("How to record")
            st.markdown("1. Keep your full right side visible.\n2. Begin with the arm extended.\n3. Keep the camera steady and well lit.\n4. Complete each curl below 30° and above 160°.")
            process_clicked = st.button(
                "Analyze video", type="primary", use_container_width=True,
                disabled=uploaded is None, key="analyze_curl",
            )
        if process_clicked and uploaded is not None:
            try:
                output_bytes, state = process_video(
                    uploaded.getvalue(), float(curled_threshold),
                    float(extended_threshold), float(shoulder_tolerance),
                )
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            else:
                st.session_state["output_video"] = output_bytes
                st.session_state["curl_state"] = state
        if "output_video" in st.session_state:
            state = st.session_state["curl_state"]
            st.divider()
            st.subheader("Curl analysis results")
            cols = st.columns(3)
            cols[0].metric("Reps completed", state.reps)
            bad_form_rate = 100 * state.bad_form_frames / state.analyzed_frames if state.analyzed_frames else 0
            cols[1].metric("Bad-form frames", f"{bad_form_rate:.1f}%")
            cols[2].metric("Frames analyzed", f"{state.analyzed_frames:,}")
            st.video(st.session_state["output_video"])
            st.download_button("Download analyzed video", st.session_state["output_video"],
                               file_name="formcheck_bicep_curl.mp4", mime="video/mp4")
        return

    if input_mode == "Live camera":
        st.subheader(f"Live squat analysis · {camera_view.lower()}")
        st.write("Place the phone far enough away to keep your full body visible. Use side profile for depth and torso lean, or front view for knee tracking and center drift.")
        webrtc_streamer(
            key=f"formcheck-live-squat-{camera_view.lower().replace(' ', '-')}",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: LiveSquatProcessor(
                camera_view, float(depth_target), float(torso_lean_limit),
                float(knee_tracking_tolerance), float(center_drift_tolerance),
            ),
            async_processing=True,
        )
        st.caption("Front-view knee calibration uses the first 12 detected frames; pause upright before squatting. The 2D center-drift cue is a rough screen-space estimate.")
        return

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader(f"Upload squat video · {camera_view.lower()}")
        uploaded = st.file_uploader(
            "MP4 video", type=["mp4"],
            help="Keep your full body and feet visible; place the phone still.",
            key=f"squat_upload_{camera_view.lower().replace(' ', '_')}",
        )
        if uploaded:
            st.video(uploaded.getvalue())
    with right:
        st.subheader("How to record")
        if camera_view == "Side profile":
            st.markdown("1. Place the phone level and still.\n2. Stand fully sideways to the camera.\n3. Keep your head, hips, knees, ankles, and feet visible.\n4. Start standing upright, squat, then return to standing.")
        else:
            st.markdown("1. Place the phone level and still.\n2. Face the camera with both legs visible.\n3. Start upright and hold for about half a second for calibration.\n4. Squat while keeping both feet and knees visible.")
        process_clicked = st.button(
            "Analyze squat", type="primary", use_container_width=True,
            disabled=uploaded is None, key=f"analyze_squat_{camera_view}",
        )
    if process_clicked and uploaded is not None:
        try:
            output_bytes, state = process_squat_video(
                uploaded.getvalue(), camera_view, float(depth_target),
                float(torso_lean_limit), float(knee_tracking_tolerance),
                float(center_drift_tolerance),
            )
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        else:
            output_key = f"squat_output_{camera_view}"
            state_key = f"squat_state_{camera_view}"
            st.session_state[output_key] = output_bytes
            st.session_state[state_key] = state
    output_key = f"squat_output_{camera_view}"
    state_key = f"squat_state_{camera_view}"
    if output_key in st.session_state:
        render_squat_results(
            st.session_state[state_key], output_key,
            f"formcheck_squat_{camera_view.lower().replace(' ', '_')}.mp4",
        )


if __name__ == "__main__":
    render_app()
