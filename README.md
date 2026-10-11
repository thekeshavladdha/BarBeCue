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
python web/server.py             # web dashboard at http://127.0.0.1:5000
python web/server.py --exercise Squat --port 5000
python evaluate.py --labels data/labels.csv --backend mediapipe --output result.json
python plot_angle.py --video data/squat_p01_good.mp4 --backend mediapipe --csv data/squat_angles.csv
python bench.py --backends mediapipe --frames 60
```

Exercises live in muscle-group categories (same list in the desktop
dropdowns and the web sidebar): Legs (Squat, Lunge, Glute bridge), Chest
(Push-up, Knee push-up), Biceps (Bicep curl), Back (Bent-over row, Superman
hold), Shoulders (Shoulder press), Abs (Plank hold, Hollow hold), Posture
(Auto posture, Desk posture).

`Auto posture` (default) detects sitting vs standing and coaches that scenario:
upright `Good form`, slouch/chin/shake hints, good-posture seconds. Press
`Calibrate` (or `C`) while in your best posture to zero thresholds to your
body + camera. Exercise sets count strict-form reps (bad-form full cycles show
`No rep - <hint>` and add to the BAD tally instead of counting); `--lenient`
restores depth-only counting. Squat/Lunge offer Beginner/Pro profiles
(Pro = deeper + stricter knees); `M` toggles spoken rep cues.

## How counting works

Rep state machine (hysteresis — separate start/end thresholds, no single line):

```text
REST --angle leaves rest zone--> MOVING --angle reaches end zone--> PEAK
 ^                                                              |
 +----------------angle returns to rest, min 0.5s since last rep-+
         PEAK -> REST emits: rep (clean) | bad_rep (sev>=2 fault
         persisted >= 0.3s mid-cycle -> counted in BAD tally) |
         MOVING -> REST without PEAK (excursion > 20 deg) emits: partial
         ("Go lower" / "Full range" flash)
```

Form faults of severity 2–3 must persist 0.3s (not one frame) to kill a rep;
severity-1 tips never block. Back-lean rules only fire on the way down
(descent gating). Side-view lifts also check camera alignment
(`Turn sideways to the camera`) and long absences show
`Paused - no person detected` instead of silently idling.

| Exercise | Rep angle | Start -> End | Key form rules |
|----------|-----------|--------------|----------------|
| Squat | knee | 160 -> 100 (Pro: 90) | back straight (sev 3, descending), knees over toes ≤ 0.30 (Pro: 0.25) |
| Lunge | knee | 160 -> 100 (Pro: 90) | same as squat |
| Push-up | elbow | 155 -> 90 | hips in line, body_line ≥ 160 (sev 3) |
| Bicep curl | elbow | 150 -> 50 | no swing (back_lean ≤ 15), elbows tucked ≤ 0.25 |
| Shoulder press | elbow | 100 -> 160 | no arch (back_lean ≤ 15) |

## Notes

- Camera: hip height, 2–3m, full body visible; side view for squat/push-up/plank.
- Thresholds are heuristics — tune via `plot_angle.py` on own clips, never clinical.
- YOLO-Pose backend is AGPL-3.0 (copyleft); MediaPipe/RTMPose are Apache-2.0.
