# FormCheck pose viewer

This first version opens a webcam or video, detects one person's pose, and draws
MediaPipe's pose landmarks and connections. It intentionally contains no
exercise or fitness-analysis logic.

## Install

MediaPipe currently works most reliably with a supported 64-bit CPython release
(Python 3.10-3.12 is a safe choice). Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project pins MediaPipe `0.10.21` because newer releases removed the legacy
Pose Solutions API used by this initial viewer. If another MediaPipe version is
already installed, replace it with the pinned version:

```powershell
python -m pip install --force-reinstall -r requirements.txt
```

On macOS/Linux, activate it with `source .venv/bin/activate` instead.

## Run

Use the default webcam:

```powershell
python pose_viewer.py
```

Use another webcam (for example, camera 1):

```powershell
python pose_viewer.py 1
```

Use a video file:

```powershell
python pose_viewer.py "path/to/workout.mp4"
```

Webcam output is mirrored by default. Add `--no-mirror` to disable that. Press
`Q` or Escape while the video window is focused to exit.

## Streamlit web app

Install the dependencies, then start the interactive dashboard:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Choose **Bicep Curl** or **Squat**, then upload an MP4 or use **Live camera**.
Both modes draw the full MediaPipe pose; each exercise analyzer uses a smaller
set of joints. Curl analysis uses the right shoulder, elbow, wrist, and hip.
Squat analysis supports:

- **Side profile:** rep counting, knee-flexion depth, and torso lean from the
  more visible side. Knee tracking toward/away from the midline needs a front
  view and is not scored from this view.
- **Front view:** rep-phase estimate, knee tracking compared with the upright
  starting position, and a 2D screen-space center-drift cue. Hold still and
  upright for the first 12 detected frames to calibrate. The front camera angle
  is less reliable for depth and torso lean.

Knee flexion is calculated as `180° - hip/knee/ankle angle`. The initial depth
reference is 70° minimum; 70–100° is labeled the half-squat range, and over
100° is a deeper range. These are approximate 2D camera measurements, not
medical or individualized coaching judgments.

### Use it from an iPhone

The app is a web app, so it can be opened in iPhone Safari after it is hosted at
an HTTPS URL. For the first iPhone trial, choose **Upload MP4** and select a
video from Photos or Files; analysis runs on the Streamlit server. Live camera
mode uses browser WebRTC permissions and network support, so test it separately
on the target iPhone and browser.

A straightforward host is Streamlit Community Cloud: connect the GitHub account,
choose this repository, select `streamlit_app.py` as the entry point, and
deploy. Keep `requirements.txt` at the repository root. The app can also run
on a computer on the same Wi-Fi network; open the computer's Streamlit URL from
Safari, with HTTPS required for camera access on most hosted browser contexts.
