# BarBeCue — Live Posture Detection and Rep Counter

Bar = Barbell + Cue = reminding and coaching. A real-time computer-vision
fitness coach: watches one user through a standard laptop webcam, estimates
pose with a pre-trained model, converts joints into biomechanical angles,
checks posture against per-exercise rules and counts reps with a hysteresis
state machine. CPU-only, no GPU, no model training, all video stays on device.

## Layout

- `posture-coach/` — the app (`src/`, `test_camera.py`, `evaluate.py`,
  `plot_angle.py`, `bench.py`, `requirements.txt`, `data/`, `models/`)
- `BarBeCue_Technical_Report.docx` / `BarBeCue_Presentation (1).pptx` — GLBITM report + slides

## Setup (Python 3.11 recommended)

```powershell
cd posture-coach
py -3.11 -m venv venv; venv\Scripts\activate
pip install -r requirements.txt
curl.exe -L -o models/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

## Run

```powershell
python test_camera.py            # raw camera check (indices 0 -> 1 -> 2)
python src/app.py                # live coach (default: Auto posture)
python src/app.py --exercise Squat --lenient
python evaluate.py --labels data/labels.csv --backend mediapipe --output result.json
python plot_angle.py --video data/squat_p01_good.mp4 --backend mediapipe --csv data/squat_angles.csv
python bench.py --backends mediapipe --frames 60
```

`Auto posture` (default) detects sitting vs standing and coaches that scenario:
upright `Good form`, slouch/chin/shake hints, good-posture seconds. Press
`Calibrate` (or `C`) while in your best posture to zero thresholds to your
body + camera. Exercise sets count strict-form reps (bad-form full cycles show
`No rep - <hint>` instead of counting); `--lenient` restores depth-only counting.

## Notes

- Camera: hip height, 2–3m, full body visible; side view for squat/push-up/plank.
- Thresholds are heuristics — tune via `plot_angle.py` on own clips, never clinical.
- YOLO-Pose backend is AGPL-3.0 (copyleft); MediaPipe/RTMPose are Apache-2.0.
