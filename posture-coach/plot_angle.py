"""BarBeCue plot_angle.py — metric extraction + threshold tuning plot (Fig 5.1, Table B.1).

Two modes share one module:
1. Sample: `--video clip --backend static --csv samples.csv [--csv-only]`
   decodes at source fps, runs detect() per frame at app resolution, and
   writes t + metric columns. Frames with no pose or gated joints become NaN
   gaps (never guessed zeros) so plots show dropouts honestly.
2. Plot: `--input samples.csv --metric knee --start 160 --end 100 --out plot.png`
   draws curve + start/end lines; proper reps must clear both lines while
   hover noise stays inside the >=30deg deadband. No --input writes a synthetic
   3-rep squat demo to prove the workflow.
"""
from __future__ import annotations
import argparse, csv, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2

from pose_estimator import create_backend

WORK_W, WORK_H = 960, 540
ALL_METRICS = ["knee", "hip", "elbow", "body_line", "back_lean", "neck_tilt",
               "knee_over_toe", "elbow_drift", "hip_sag"]

__all__ = ["_metric_available", "extract_metrics", "main", "_make_backend"]


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


def _vis(kp: dict, key: str) -> float:
    v = kp.get(key)
    return float(v[2]) if v is not None else -1.0


def _metric_available(kp: dict, side: str, metric: str, thr: float = 0.5) -> bool:
    """Visibility-gated computability (plot gaps, never guessed zeros).

    Ear rule: side ear key wins when present (even if dim) with nose fallback.
    """
    ok = lambda *keys: all(_vis(kp, k) >= thr for k in keys)
    s = lambda j: f"{side}_{j}"
    if metric == "knee":
        return ok(s("hip"), s("knee"), s("ankle"))
    if metric == "hip":
        return ok(s("shoulder"), s("hip"), s("knee"))
    if metric == "elbow":
        return ok(s("shoulder"), s("elbow"), s("wrist"))
    if metric == "body_line":
        return ok(s("shoulder"), s("hip"), s("ankle"))
    if metric == "back_lean":
        return ok(s("shoulder"), s("hip"))
    if metric == "neck_tilt":
        head = s("ear") if s("ear") in kp else "nose"
        return ok(s("shoulder")) and _vis(kp, head) >= thr
    if metric == "knee_over_toe":
        foot = s("foot_index") if s("foot_index") in kp else (s("heel") if s("heel") in kp else None)
        return foot is not None and ok(s("knee"), s("ankle"), foot)
    if metric == "elbow_drift":
        return ok(s("shoulder"), s("elbow"))
    if metric == "hip_sag":
        return ok(s("shoulder"), s("hip"), s("ankle"))
    return False


def extract_metrics(video, backend, metrics: list) -> list:
    """Per-frame t + metric values (NaN on no-pose/gated). Releases capture."""
    from geometry import pick_side, compute_metrics
    if isinstance(backend, str):
        backend = _make_backend(backend)
        owned = True
    else:
        owned = False
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = float(fps)
        except Exception:
            fps = 30.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        rows, idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (WORK_W, WORK_H))
            kp = backend.detect(frame, int(idx * 1000.0 / fps))
            row = {"t": round(idx / fps, 3)}
            if kp:
                from geometry import pick_side as _pick
                side = _pick(kp)
                m = compute_metrics(kp, side)
                for metric in metrics:
                    row[metric] = m[metric] if _metric_available(kp, side, metric) else float("nan")
            else:
                for metric in metrics:
                    row[metric] = float("nan")
            rows.append(row)
            idx += 1
    finally:
        cap.release()
        if owned:
            try:
                backend.close()
            except Exception:
                pass
    return rows


def _demo_rows():
    t, y, idx = [], [], [0]
    def push(v):
        t.append(round(idx[0] * 0.1, 3)); y.append(v); idx[0] += 1
    for _ in range(3):
        for i in range(12):
            push(165 - 70 * i / 11)
        for i in range(12):
            push(95 + 70 * i / 11)
        for _ in range(5):
            push(165)
    return t, y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sample pose metrics and/or plot tuning curves.")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--backend", default="static")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "pose_landmarker_full.task")
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--metrics", default="knee", help="comma-separated metric names")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--metric", default="knee")
    ap.add_argument("--start", type=float, default=160)
    ap.add_argument("--end", type=float, default=100)
    ap.add_argument("--out", type=Path, default=Path("plot.png"))
    args = ap.parse_args(argv)

    rows = None
    if args.video:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or ["knee"]
        backend = _make_backend(args.backend, args.model, args.yolo_weights)
        try:
            rows = extract_metrics(args.video, backend, metrics)
        finally:
            try:
                backend.close()
            except Exception:
                pass
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["t"] + metrics)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[plot] wrote {args.csv} ({len(rows)} frames)")
        if args.csv_only:
            return 0

    if rows is None:
        if args.input and Path(args.input).exists():
            t, y = [], []
            with open(args.input, newline="", encoding="utf-8") as handle:
                for r in csv.DictReader(handle):
                    t.append(float(r["t"])); y.append(float(r[args.metric]))
            demo, metric = False, args.metric
        else:
            t, y = _demo_rows()
            demo, metric = True, args.metric
            rows = [{"t": tt, metric: vv} for tt, vv in zip(t, y)]
    else:
        metric = args.metric
        t = [r["t"] for r in rows]
        y = [r.get(metric, float("nan")) for r in rows]
        demo = False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t, y, label=f"{metric} angle")
    plt.axhline(args.start, linestyle="--", label=f"start {args.start}")
    plt.axhline(args.end, linestyle="--", label=f"end {args.end}")
    plt.xlabel("time (s)")
    plt.ylabel("angle (deg)")
    plt.title(f"{'Demo ' if demo else ''}{metric} with start/end thresholds")
    plt.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
