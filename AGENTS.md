# AGENTS.md — BarBeCue (posture-coach)

Live posture detection + rep counter. Python + OpenCV + MediaPipe, CPU-only, offline-first, all video stays on device.

## Layout

- `posture-coach/src/` — app code: `app.py` (Tkinter loop + `CoachState`), `geometry.py`, `exercises.py`, `rep_counter.py`, `feedback.py`, `posture.py`, `smoothing.py`, `pose_estimator.py`
- `posture-coach/` root scripts: `test_camera.py`, `evaluate.py`, `plot_angle.py`, `bench.py`, `requirements.txt`
- `posture-coach/data/labels.csv` — labelled-clip metadata (currently header-only, no clips)
- `posture-coach/models/pose_landmarker_full.task` — MediaPipe model asset (git-ignored, must exist locally)
- Root `README.md` — user setup/run; `posture-coach/explainer.md` — detailed design (git-ignored)

## Setup (Python 3.11 recommended, 3.10–3.12 ok)

```powershell
cd posture-coach
py -3.11 -m venv venv; venv\Scripts\activate
pip install -r requirements.txt
curl.exe -L -o models/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

MediaPipe wheels lag newest Python — prefer 3.11. Tkinter ships with Python (not in requirements). `matplotlib` only needed for `plot_angle.py` PNG output.

## Commands

```powershell
python test_camera.py            # raw camera check (indices 0 -> 1 -> 2), q to quit
python src/app.py                # live coach, default Auto posture
python src/app.py --exercise Squat --lenient
python src/app.py --video data/squat.mp4 --backend mediapipe
python evaluate.py --labels data/labels.csv --backend mediapipe --output result.json
python plot_angle.py --video data/squat.mp4 --backend mediapipe --csv data/squat_angles.csv
python bench.py --backends mediapipe --frames 60
```

Headless smoke tests (no camera/GUI/model): use `--backend static` or `static,dummy`.

## Architecture / pipeline

Per-frame (`CoachState.step()`, GUI-independent and unit-testable):
`read+mirror (cv2.flip) -> resize 960x540 -> detect() -> EMA smooth x/y -> pick_side -> visibility gate -> compute_metrics -> AngleSmoother -> counter/hold/auto-posture -> Feedback -> draw skeleton + HUD`

- `CoachState` in `src/app.py` is the shared live + offline path. `evaluate.py` reuses it. Keep it GUI-free.
- Backends (`pose_estimator.py`) all return `{name: (x_px, y_px, visibility)}` in **pixels, not normalized**. MediaPipe = default (33-pt); YOLO-Pose = optional 17-pt COCO, largest bbox wins; RTMPose = guarded stub (not wired); `DummyPose` (no person) / `StaticStandingPose` (synthetic) for tests.
- `geometry.py`: pure functions, no camera/model/UI state. Missing landmarks → safe defaults, but callers must gate on visibility first.
- `exercises.py`: single config center (`EXERCISES` dict). Add exercises there, never with conditionals in `app.py`.
- `rep_counter.py`: hysteresis state machine (REST/MOVING/PEAK, separate start/end thresholds, 0.5s min interval, partial-rep events).
- `feedback.py`: `raw_violations()` (instant, used for strict gating) vs `Feedback.update()` (debounce ~0.6s, hold ~1.5s, sort by severity, max 2 msgs). Both accept `phase`; rules with `phase="descending"` fire mid-rep on the way down only.
- Strict-form counting is default ON: severity >= 2 faults persisting >= 0.3s
  (FAULT_LATCH, time-based) during a cycle → `bad_rep`, tallied in `bad_reps`
  with `last_bad_reason` (HUD: `REPS n BAD m (<reason>)`). Sev 1 advisory only.
  `--lenient` restores depth-only counting. `--profile Beginner/Pro` (Squat/Lunge)
  tightens depth + knee-over-toe; `CoachState.set_profile()` resets set like `select()`.

## Conventions (follow these)

- Pixel space: always convert normalized coords to pixels (`x*W, y*H`) before `angle()` — 960x540 is non-square, angles distort otherwise.
- Visibility gate: `VIS_TH = 0.5` on exercise-required joints; on fail show `Step back so your whole body is visible`, skip counting/feedback for that frame. `foot_index` is optional for 17-pt backends.
- Side selection: per-frame, sum vis of shoulder+hip+knee+ankle, pick more visible side. Don't hardcode left/right.
- Camera-alignment guard (side-view exercises only): `face_offset_angle()` (nose vs shoulder midpoint) < ALIGN_MIN (20.0) → `bad_view` status (`Turn sideways to the camera`), counting skipped, drawing kept.
- Missing person: `no_person`/`step_back` past PAUSE_AFTER (15s) → `paused` status (timers/tallies frozen, never reset).
- Smoothing defaults: keypoint `EMA(0.5)`, angle `AngleSmoother(0.6)`. Reset all stateful filters/counter/feedback/timers/shake buffer on exercise change.
- Skeleton: draw links only when both endpoints vis > 0.5.
- Severity: 3 = safety, 2 = alignment/technique (blocks reps), 1 = style tip. `when` gates (e.g. squat back-lean only when flexed) avoid nagging at rest.
- Calibration (`C` / button): 3s, ≥10 samples, median becomes back_lean/neck_tilt baseline. Relative offsets only, never clinical.
- Missing pose = explicit gap/status (NaN in CSV, paused hold timer — not reset), never invented advice.
- Imports in `src/`: bare `from geometry import ...` (no `src.` prefix) + `from __future__ import annotations`. Keep that style.
- Timestamps: video-file time from frame#/FPS, strictly increasing for MediaPipe video mode.

## Don't

- Don't auto-download YOLO weights in `evaluate.py`/`bench.py` — weights path must be explicit. YOLO backend is AGPL-3.0 copyleft; MediaPipe/RTMPose are Apache-2.0.
- Don't put thresholds in `app.py`/HUD code — they belong in `EXERCISES`.
- Don't add camera/GUI deps to `geometry.py`, `rep_counter.py`, `smoothing.py`, `exercises.py`.
- Don't claim accuracy without clips: `data/labels.csv` is empty, bench on black frames is smoke-test only.
- Don't commit: `*.task`, `*.mp4`, `*.png`, `*.log`, `venv/`, `__pycache__/`, `result.json`, `samples.csv`, `report.pdf`, `SIDE_BY_SIDE.md`, report `.docx`/`.pptx`, `posture-coach/explainer.md` (all in single root `.gitignore` — no nested ignore files).

## Notes

- Camera: hip height, 2–3m, full body visible; side view for squat/push-up/plank. If camera fails, close Zoom/Meet first.
- Thresholds are heuristics — tune via `plot_angle.py` on own clips, never clinical.
