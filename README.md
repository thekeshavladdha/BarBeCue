# BarBeCue — Live Posture Detection and Rep Counter

Bar = Barbell + Cue = reminding and coaching. A real-time computer-vision
fitness coach: watches one user through a standard laptop webcam, estimates
pose with a pre-trained model, converts joints into biomechanical angles,
checks posture against per-exercise rules and counts reps with a hysteresis
state machine. CPU-only, no GPU, no model training, offline-first, all video
stays on device.

## Layout

- `posture-coach/src/` — app code: `app.py` (Tkinter loop + `CoachState`),
  `geometry.py`, `exercises.py`, `rep_counter.py`, `feedback.py`,
  `posture.py`, `smoothing.py`, `pose_estimator.py`
- `posture-coach/web/` — browser dashboard (`server.py` Flask + MJPEG,
  `templates/`, `static/`); same `CoachState` pipeline, sidebar by muscle
  group, session logbook
- `posture-coach/` root scripts: `test_camera.py`, `evaluate.py`,
  `plot_angle.py`, `bench.py`, `sampler.py`, `requirements.txt`
- `posture-coach/data/labels.csv` — labelled-clip metadata (currently
  header-only, no clips)
- `posture-coach/models/pose_landmarker_full.task` — MediaPipe model asset
  (git-ignored, must exist locally)

## Setup (Python 3.11 recommended, 3.10–3.12 ok)

```powershell
cd posture-coach
py -3.11 -m venv venv; venv\Scripts\activate
pip install -r requirements.txt
python download_model.py
```

### Model file (do this once per fresh clone)

The pose model (`models/pose_landmarker_full.task`, ~9 MB) is git-ignored,
so it never comes with `git clone` — every setup needs it once. Without it
the app exits with `pose model not found`.

- Recommended (Windows / macOS / Linux, no curl knowledge needed):
  ```powershell
  cd posture-coach
  python download_model.py
  ```
  Expected output ends with `saved (... MB): models/pose_landmarker_full.task`.
  Re-running is a no-op unless you pass `--force`.
- Alternative with curl (PowerShell must use `curl.exe`, not `curl`):
  ```powershell
  cd posture-coach
  curl.exe -L -o models/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
  ```
- Manual fallback (browser): open the URL above, save the file as
  `posture-coach/models/pose_landmarker_full.task` (exact name, inside the
  existing `models/` folder — create it if missing).

Verify it worked:

```powershell
dir models\pose_landmarker_full.task   # ~9,398,198 bytes
python download_model.py               # prints "exists ... nothing to do"
```

Troubleshooting:

- `pose model not found` → you skipped this section, or ran the app from the
  wrong folder (paths assume you run from `posture-coach/`).
- `curl: not recognized` → PowerShell: use `curl.exe ...`; or just use
  `python download_model.py` instead.
- Truncated / 0-byte file → delete it and re-run with
  `python download_model.py --force`.
- Just want a smoke test without the model: `python src/app.py --backend static`.

Tkinter ships with Python (not in requirements). `matplotlib` is only needed
for `plot_angle.py` PNG output. If the camera fails, close Zoom/Meet first.
Headless smoke test (no camera/GUI/model): `--backend static`.

## Run

```powershell
python test_camera.py            # raw camera check (indices 0 -> 1 -> 2), q to quit
python src/app.py                # live coach, default Auto posture
python src/app.py --exercise Squat --lenient
python src/app.py --exercise Squat --profile Pro --no-audio
python src/app.py --video data/squat.mp4 --backend mediapipe
python web/server.py             # web dashboard at http://127.0.0.1:5000
python web/server.py --exercise Squat --port 5000 --profile Beginner
python sampler.py --label upright --seconds 3   # live threshold sampler (run when app.py is NOT holding the camera)
python evaluate.py --labels data/labels.csv --backend mediapipe --output result.json
python plot_angle.py --video data/squat.mp4 --backend mediapipe --csv data/squat_angles.csv
python bench.py --backends mediapipe --frames 60
```

Exercises live in muscle-group categories (same list in the desktop
dropdowns and the web sidebar): Legs (Squat, Lunge, Glute bridge), Chest
(Push-up, Knee push-up), Biceps (Bicep curl), Back (Bent-over row, Superman
hold), Shoulders (Shoulder press), Abs (Plank hold, Hollow hold), Posture
(Auto posture, Desk posture).

`Auto posture` (default) detects sitting vs standing and coaches that
scenario: upright `Good form`, slouch/chin/shake hints, good-posture seconds.
`Desk posture` is the hold-mode version (sitting only, good-form timer pauses
on fault). Press `Calibrate` (or `C`) while in your best posture to zero
back-lean/neck-tilt thresholds to your body + camera. Exercise sets count
strict-form reps (bad-form full cycles show `No rep - <hint>` and add to the
BAD tally instead of counting); `--lenient` restores depth-only counting.
Squat/Lunge offer Beginner/Pro profiles (Pro = deeper + stricter knees).
`M` toggles spoken rep cues (`--no-audio` disables, `pyttsx3` optional).
Category + Profile dropdowns and `Reset` never restart the camera loop.

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
severity-1 tips never block. Back-lean rules on squat/lunge only fire on the
way down (descent gating). Side-view lifts also check camera alignment
(`Turn sideways to the camera`), missing joints show
`Step back so your whole body is visible`, and long absences show
`Paused - no person detected` (timers frozen, never reset).

| Exercise | Mode | Rep angle | Start -> End | Key form rules |
|----------|------|-----------|--------------|----------------|
| Squat | reps | knee | 160 -> 100 (Pro: 90) | back straight > 55 sev 3 descending, knees over toes ≤ 0.30 (Pro: 0.25), chest up sev 1 |
| Lunge | reps | knee | 160 -> 100 (Pro: 90) | same as squat |
| Glute bridge | reps | hip | 100 -> 160 | depth only, no lean rules |
| Push-up | reps | elbow | 155 -> 90 | hips in line, body_line ≥ 160 (sev 3) |
| Knee push-up | reps | elbow | 155 -> 110 | hips in line, body_line ≥ 160 (sev 3) |
| Bicep curl | reps | elbow | 150 -> 50 | no swing back_lean ≤ 15, elbows tucked ≤ 0.25 |
| Bent-over row | reps | elbow | 150 -> 60 | bend over at hips, back_lean ≥ 30 when pulled in |
| Superman hold | hold | — | good-form timer | lift chest + legs, body_line ≤ 170 (sev 3) |
| Shoulder press | reps | elbow | 100 -> 160 | no arch, back_lean ≤ 15 |
| Plank hold | hold | — | good-form timer | hips level ± 0.06 + body_line ≥ 160 (sev 3) |
| Hollow hold | hold | — | good-form timer | banana 125–170, curl tighter / open up (sev 2) |
| Desk posture | hold | — | good-form timer | sit straight ≤ 20, chin ≤ 25, uncurl back ≥ 0.28 |
| Auto posture | auto | — | good-posture secs | sitting: lean ≤ 20 / chin ≤ 15 / back ≥ 0.28; standing: lean ≤ 15 / chin ≤ 25 / back ≥ 0.36 |

Feedback is debounced (~0.6s) + held (~1.5s), sorted by severity, max 2
messages. Live angle overlay draws the measured joint angle.

## Notes

- Camera: hip height, 2–3m, full body visible; side view for squat/push-up/plank.
- Thresholds are heuristics — tune via `plot_angle.py` / `sampler.py` on own clips, never clinical.
- `data/labels.csv` is header-only, so no accuracy claims; bench on black frames is smoke-test only.
- YOLO-Pose backend is AGPL-3.0 (copyleft); MediaPipe/RTMPose are Apache-2.0.
