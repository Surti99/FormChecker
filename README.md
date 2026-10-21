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

Install the dependencies, then start the interactive upload dashboard:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Streamlit opens the app in your browser. Choose **Bicep Curl**, upload an MP4,
adjust the tracking thresholds if needed, and select **Analyze video**. The
result includes the pose skeleton, rep count, elbow angle, shoulder movement,
and live form feedback. You can play or download the analyzed video afterward.

Choose **Live camera** under **Input source** to analyze your webcam in real
time. Select **START** in the camera panel and allow camera access when the
browser asks. The live session resets its rep count whenever the camera stream
is restarted.
