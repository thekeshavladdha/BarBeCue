"""BarBeCue evaluate.py — offline repeatable harness (Ch.5, Appx C/G).

Drives an EXISTING CoachState (counter+smoothing owned by the caller) over a
clip in video-time (deterministic), at app resolution:
  read -> stride filter -> resize 960x540 -> detect(ts=source idx/fps)
  -> state.step(kp, t=source idx/fps)
Source timestamps (not processed counts) keep rep tempo honest under stride.
Fresh CoachState per clip in main() = independent runs. Missing clips and
inference errors are recorded per-clip (main continues); capture is always
released. Labels: data/labels.csv(file,true_reps,fault,exercise), aliases
filename/reps/label/movement accepted, split by person for classifier.

Run: python evaluate.py --labels data/labels.csv --backend static --output result.json
Targets (Report Table 5.1): exact-match >90%, mean abs err <0.5.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2

from pose_estimator import create_backend

WORK_W, WORK_H = 960, 540
REPS_EXERCISES = {"Squat", "Lunge", "Push-up", "Bicep curl", "Shoulder press"}

__all__ = [
    "evaluate_clip", "_result_for_label", "_summary", "read_labels",
    "main", "_make_backend",
]


def _make_backend(kind: str, model: Path | None = None, yolo_weights: str | None = None):
    """Shared guard: never auto-download YOLO weights."""
    lowered = kind.lower()
    if lowered in {"yolo", "yolo-pose", "ultralytics"}:
        if not yolo_weights:
            raise RuntimeError("no --yolo-weights supplied; skipped to avoid an implicit download")
        weights = Path(yolo_weights).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        return create_backend(kind, weights=str(weights.resolve()))
    if lowered in {"mediapipe", "mp", "full", "lite", "heavy"}:
        mp = Path(model) if model else (ROOT / "models" / "pose_landmarker_full.task")
        return create_backend(kind, model_path=str(mp))
    return create_backend(kind)


def _finite_fps(raw) -> float:
    try:
        fps = float(raw)
    except Exception:
        return 30.0
    return fps if math.isfinite(fps) and fps > 0 else 30.0


def evaluate_clip(pose, state, video, stride: int = 1, max_frames: int | None = None) -> dict:
    """Run one clip through `state`. Returns frames/predicted_reps/hold/flags.

    `stride`: process every Nth source frame (timestamps stay in source time).
    `max_frames`: stop after N source reads (row marked truncated).
    Raises inference errors after releasing capture (caller records per-clip).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {video}")
    fps = _finite_fps(cap.get(cv2.CAP_PROP_FPS))
    processed, saw_pose, saw_msg = 0, False, False
    truncated = False
    try:
        idx = 0
        while True:
            if max_frames is not None and idx >= max_frames:
                truncated = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(int(stride), 1) != 0:
                idx += 1
                continue
            frame = cv2.resize(frame, (WORK_W, WORK_H))
            ts_ms = int(idx * 1000.0 / fps)
            kp = pose.detect(frame, ts_ms)  # may raise -> finally releases
            t = idx / fps
            out = state.step(kp, t)
            processed += 1
            if kp:
                saw_pose = True
            if out.get("msgs"):
                saw_msg = True
            idx += 1
    finally:
        cap.release()
    reps = getattr(state.counter, "reps", 0) if state.cfg.get("mode") == "reps" else 0
    return {
        "status": "ok", "frames": processed, "predicted_reps": int(reps),
        "hold_seconds": round(float(getattr(state, "hold_secs", 0.0)), 2),
        "saw_pose": saw_pose, "saw_msg": saw_msg, "truncated": truncated,
    }


def _result_for_label(label: dict, run: dict, _thresholds=None) -> dict:
    f = str(label.get("fault", "good") or "good")
    true_reps = int(label.get("true_reps", 0))
    row = {
        "file": label.get("file", ""), "exercise": label.get("exercise", ""),
        "true_reps": true_reps, "predicted_reps": run.get("predicted_reps", 0),
        "hold_seconds": run.get("hold_seconds", 0),
        "status": run.get("status", "ok"), "truncated": bool(run.get("truncated", False)),
        "reps_correct": "", "abs_rep_error": "", "fault_match": "",
    }
    if row["truncated"]:
        row["status"] = "truncated"
        return row
    if not run.get("saw_pose", True):
        row["status"] = "no_usable_pose"
        return row
    if not f or label.get("exercise") not in REPS_EXERCISES:
        return row  # hold mode / blank fault: reps + fault unscored, hold kept
    row["reps_correct"] = row["predicted_reps"] == true_reps
    row["abs_rep_error"] = abs(row["predicted_reps"] - true_reps)
    row["fault_match"] = (f == "good") == (not run.get("saw_msg", False))
    return row


def _summary(rows: list) -> dict:
    rep_rows = [r for r in rows if r.get("abs_rep_error") not in ("", None)]
    fault_rows = [r for r in rows if r.get("fault_match") not in ("", None)]
    return {
        "clips": len(rows),
        "rep_scored_clips": len(rep_rows),
        "fault_scored_clips": len(fault_rows),
        "rep_exact": sum(1 for r in rep_rows if r.get("reps_correct")),
        "rep_mean_abs_err": (sum(r["abs_rep_error"] for r in rep_rows) / len(rep_rows)) if rep_rows else 0.0,
        "fault_match": sum(1 for r in fault_rows if r.get("fault_match")),
    }


_ALIASES = {"filename": "file", "reps": "true_reps", "label": "fault", "movement": "exercise"}


def read_labels(path: Path | str) -> list:
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [(_ALIASES.get(h.strip().lower(), h.strip().lower())) for h in (reader.fieldnames or [])]
        out = []
        for i, raw in enumerate(reader, start=2):
            row = {f: (raw[h] or "").strip() for f, h in zip(fields, (reader.fieldnames or []))}
            row.setdefault("fault", "good")
            try:
                row["true_reps"] = int(float(row.get("true_reps", 0)))
            except ValueError:
                raise ValueError(f"row {i}: bad true_reps {row.get('true_reps')!r}")
            if row["true_reps"] < 0:
                raise ValueError(f"row {i}: bad true_reps {row['true_reps']}")
            if not row.get("file") or not row.get("exercise"):
                raise ValueError(f"row {i}: file + exercise required")
            out.append(row)
    return out


def main(argv=None) -> int:
    sys.path.insert(0, str(SRC))
    from app import CoachState
    ap = argparse.ArgumentParser(description="Score BarBeCue clips against labels (headless).")
    ap.add_argument("--labels", type=Path, default=ROOT / "data" / "labels.csv")
    ap.add_argument("--backend", default="static")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "pose_landmarker_full.task")
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    labels = read_labels(args.labels)
    clips, failed = [], 0
    for lab in labels:
        base = Path(args.labels).parent
        cand = Path(lab["file"])
        video = cand if cand.is_absolute() else (base / cand)
        try:
            from pathlib import Path as _P
            pose = _make_backend(args.backend, args.model, args.yolo_weights)
            state = CoachState(lab["exercise"])
            run = evaluate_clip(pose, state, video, stride=args.stride, max_frames=args.max_frames)
            try:
                pose.close()
            except Exception:
                pass
            row = _result_for_label(lab, run)
            row["backend"] = args.backend
            clips.append(row)
            print(f"[evaluate] {lab['file']}: true={lab['true_reps']} pred={row['predicted_reps']} status={row['status']}")
        except Exception as exc:
            failed += 1
            clips.append({"file": lab["file"], "exercise": lab.get("exercise", ""), "status": "failed", "error": str(exc)})
            print(f"[evaluate] {lab['file']}: FAILED ({exc})")
    summary = _summary([c for c in clips if c.get("status") not in ("failed",)])
    summary.update({"completed": len(clips) - failed, "failed": failed})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "clips": clips}, indent=2), encoding="utf-8")
        print(f"[evaluate] wrote {args.output}")
    s = _summary([c for c in clips if c.get("abs_rep_error") not in ("", None)])
    if s["rep_scored_clips"]:
        print(f"[evaluate] exact {s['rep_exact']}/{s['rep_scored_clips']} mean-err {s['rep_mean_abs_err']:.2f} (targets >90%, <0.5)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
