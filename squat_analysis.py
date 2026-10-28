"""Squat kinematics for the Streamlit pose-analysis pipeline.

MediaPipe detects the complete 33-landmark body pose. This module consumes only
the subset needed for a squat; front-view knee tracking is intentionally marked
as unavailable for side-view footage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from pose_viewer import calculate_angle


@dataclass
class SquatState:
    reps: int = 0
    phase: str = "STANDING"
    bottom_reached: bool = False
    previous_flexion: float | None = None
    current_flexion: float | None = None
    deepest_flexion: float = 0.0
    rep_depths: list[float] = field(default_factory=list)
    torso_angle: float | None = None
    knee_tracking_deviation: float | None = None
    center_drift: float | None = None
    hip_center_baseline: float | None = None
    standing_knee_offsets: list[float] = field(default_factory=list)
    knee_offsets_baseline: tuple[float, float] | None = None
    analyzed_frames: int = 0
    issue_frames: dict[str, int] = field(default_factory=dict)
    last_cues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SquatFrame:
    knee_flexion: float | None
    torso_lean: float | None
    knee_tracking_deviation: float | None
    center_drift: float | None
    phase: str
    cues: tuple[str, ...]
    joints_visible: bool


# Landmark names are kept in one place so a later gait analyzer can declare its
# own subset without changing the capture or drawing pipeline.
SQUAT_LANDMARKS = {
    "side": ("SHOULDER", "HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"),
    "front": ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
              "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"),
}


def _landmark(landmarks: Any, pose_module: Any, name: str) -> Any:
    return landmarks[getattr(pose_module.PoseLandmark, name).value]


def _visible(*points: Any, minimum: float = 0.5) -> bool:
    return all(getattr(point, "visibility", 0.0) >= minimum for point in points)


def _best_side(landmarks: Any, pose_module: Any) -> tuple[str, Any, Any, Any, Any, Any]:
    candidates = []
    for side in ("RIGHT", "LEFT"):
        shoulder, hip, knee, ankle, heel, toe = (
            _landmark(landmarks, pose_module, f"{side}_{joint}")
            for joint in ("SHOULDER", "HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX")
        )
        visibility = sum(getattr(p, "visibility", 0.0)
                         for p in (shoulder, hip, knee, ankle, heel, toe))
        candidates.append((visibility, side, shoulder, hip, knee, ankle, heel, toe))
    _, side, shoulder, hip, knee, ankle, heel, toe = max(candidates, key=lambda item: item[0])
    return side, shoulder, hip, knee, ankle, heel, toe


def analyze_squat(
    landmarks: Any,
    pose_module: Any,
    state: SquatState,
    camera_view: str = "Side profile",
    depth_target: float = 70.0,
    torso_lean_limit: float = 50.0,
    knee_tracking_tolerance: float = 0.10,
    center_drift_tolerance: float = 0.20,
) -> SquatFrame:
    """Update one frame of squat kinematics and return visible metrics/cues.

    Knee flexion is 180° minus the hip-knee-ankle angle. A rep counts only
    after reaching the chosen depth target and returning to standing.
    """
    cues: list[str] = []
    front_view = camera_view == "Front view"

    if front_view:
        left = tuple(_landmark(landmarks, pose_module, f"LEFT_{joint}")
                     for joint in ("SHOULDER", "HIP", "KNEE", "ANKLE"))
        right = tuple(_landmark(landmarks, pose_module, f"RIGHT_{joint}")
                      for joint in ("SHOULDER", "HIP", "KNEE", "ANKLE"))
        if not (_visible(*left) and _visible(*right)):
            state.last_cues = ["Keep both legs and hips visible"]
            return SquatFrame(None, None, None, None, state.phase,
                              tuple(state.last_cues), False)
        ls, lh, lk, la = left
        rs, rh, rk, ra = right
        shoulder_width = max(abs(ls.x - rs.x), 1e-4)
        hip_center = (lh.x + rh.x) / 2
        body_center = (ls.x + rs.x + lh.x + rh.x) / 4

        # Calibrate knee-to-ankle offsets during the first upright frames. The
        # positive direction is toward the body's midline on either leg.
        offsets = []
        for hip, knee, ankle in ((lh, lk, la), (rh, rk, ra)):
            inward = 1.0 if hip.x < hip_center else -1.0
            offsets.append((knee.x - ankle.x) * inward)
        knee_deviation = None
        if state.knee_offsets_baseline is None:
            state.standing_knee_offsets.append(offsets[0])
            if len(state.standing_knee_offsets) >= 12:
                # Use the left-side offset as a stable calibration marker;
                # right baseline is estimated from mirrored leg geometry.
                base = median(state.standing_knee_offsets)
                state.knee_offsets_baseline = (base, base)
        if state.knee_offsets_baseline is not None:
            deviations = [
                offsets[i] - state.knee_offsets_baseline[i] for i in range(2)
            ]
            knee_deviation = max(0.0, max(deviations)) / shoulder_width
        side, shoulder, hip, knee, ankle, heel, toe = _best_side(landmarks, pose_module)
        # Front-view knee angles are less reliable under foreshortening, but
        # remain useful for phase and rep segmentation; display angle only.
        knee_angle = calculate_angle(hip, knee, ankle)
        torso_lean = None
    else:
        side, shoulder, hip, knee, ankle, heel, toe = _best_side(landmarks, pose_module)
        selected = (shoulder, hip, knee, ankle, heel, toe)
        if not _visible(*selected):
            state.last_cues = ["Turn slightly sideways and keep your full body visible"]
            return SquatFrame(None, None, None, None, state.phase,
                              tuple(state.last_cues), False)
        knee_angle = calculate_angle(hip, knee, ankle)
        torso_lean = __import__("math").degrees(
            __import__("math").atan2(abs(shoulder.x - hip.x),
                                     max(abs(shoulder.y - hip.y), 1e-4))
        )
        knee_deviation = None
        body_center = hip.x
        shoulder_width = max(abs(shoulder.x - hip.x) * 2.0, 0.04)
        cues.append("Front view needed to assess knee tracking")

    flexion = max(0.0, min(180.0, 180.0 - knee_angle))
    state.current_flexion = flexion
    state.analyzed_frames += 1

    # Center drift is a 2D screen-space proxy. It is shown as a metric, not
    # presented as a clinical or force-plate balance measurement.
    if state.hip_center_baseline is None:
        state.hip_center_baseline = body_center
    center_drift = abs(body_center - state.hip_center_baseline) / shoulder_width
    state.center_drift = center_drift
    if center_drift > center_drift_tolerance:
        cues.append("Keep your center steadier")

    if front_view:
        if knee_deviation is not None and knee_deviation > knee_tracking_tolerance:
            cues.append("Keep knees tracking over your feet")
        cues.append("Side view gives more reliable depth and torso angles")
    elif torso_lean is not None and torso_lean > torso_lean_limit:
        cues.append("Reduce excessive forward torso lean")

    previous = state.previous_flexion
    if flexion < 35.0:
        if state.bottom_reached:
            state.reps += 1
            state.rep_depths.append(state.deepest_flexion)
        elif state.deepest_flexion > 0:
            cues.append("Go lower to complete the rep")
        state.phase = "STANDING"
        state.bottom_reached = False
        state.deepest_flexion = 0.0
    else:
        if flexion >= depth_target:
            state.bottom_reached = True
            state.deepest_flexion = max(state.deepest_flexion, flexion)
            state.phase = "BOTTOM"
        elif state.bottom_reached and previous is not None and flexion < previous - 2.0:
            state.phase = "ASCENDING"
        elif previous is not None and flexion > previous + 2.0:
            state.phase = "DESCENDING"
        elif state.phase == "STANDING":
            state.phase = "DESCENDING"
        if not state.bottom_reached and flexion >= 35.0:
            cues.append(f"Reach at least {depth_target:.0f}° knee flexion")
    state.previous_flexion = flexion

    if torso_lean is not None and torso_lean > torso_lean_limit:
        state.issue_frames["torso"] = state.issue_frames.get("torso", 0) + 1
    if knee_deviation is not None and knee_deviation > knee_tracking_tolerance:
        state.issue_frames["knee_tracking"] = state.issue_frames.get("knee_tracking", 0) + 1
    if center_drift > center_drift_tolerance:
        state.issue_frames["stability"] = state.issue_frames.get("stability", 0) + 1

    state.last_cues = cues
    return SquatFrame(flexion, torso_lean, knee_deviation, center_drift,
                      state.phase, tuple(cues), True)
