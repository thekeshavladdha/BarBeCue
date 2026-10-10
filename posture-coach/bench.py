"""Headless benchmark for the pose backends exposed by BarBeCue.

Optional backends are attempted independently.  Missing packages, models, and
the current RTMPose stub are recorded as ``unavailable`` instead of aborting
the whole comparison.  YOLO is never allowed to auto-download weights: pass
``--yolo-weights`` explicitly if it should be measured.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from pose_estimator import create_backend

WORK_W, WORK_H = 960, 540


def _make_backend(kind: str, model: Path, yolo_weights: str | None):
    lowered = kind.lower()
    if lowered in {"mediapipe", "mp", "full", "lite", "heavy"}:
        return create_backend(kind, model_path=str(model))
    if lowered in {"yolo", "yolo-pose", "ultralytics"}:
        if not yolo_weights:
            raise RuntimeError("no --yolo-weights supplied; skipped to avoid an implicit download")
        weights = Path(yolo_weights).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        return create_backend(kind, weights=str(weights.resolve()))
    return create_backend(kind)


def _load_frames(video: Path | None, count: int) -> tuple[list[np.ndarray], float, str]:
    if video is None:
        return [np.zeros((WORK_H, WORK_W, 3), dtype=np.uint8)] * count, 30.0, "synthetic black frames (smoke only; not representative of person tracking)"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    try:
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frames.append(cv2.resize(frame, (WORK_W, WORK_H)))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"video contains no readable frames: {video}")
    return frames, fps if math.isfinite(fps) and fps > 0 else 30.0, str(video)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def benchmark_backend(kind: str, frames: list[np.ndarray], fps: float, warmup: int, model: Path, yolo_weights: str | None) -> dict:
    started = time.perf_counter()
    try:
        backend = _make_backend(kind, model, yolo_weights)
    except Exception as exc:
        return {"backend": kind, "status": "unavailable", "error": str(exc), "init_ms": round((time.perf_counter() - started) * 1000, 3)}
    init_ms = (time.perf_counter() - started) * 1000
    latencies = []
    detections = 0
    errors = []
    warmup_count = max(warmup, 0)
    try:
        for index in range(warmup_count):
            try:
                backend.detect(frames[index % len(frames)].copy(), int(index * 1000.0 / fps))
            except Exception as exc:
                errors.append(str(exc))
        for index, frame in enumerate(frames, start=warmup_count):
            frame = frame.copy()  # keep the same cached input for every backend
            started_frame = time.perf_counter()
            try:
                keypoints = backend.detect(frame, int(index * 1000.0 / fps))
                detections += bool(keypoints)
            except Exception as exc:
                errors.append(str(exc))
            latencies.append((time.perf_counter() - started_frame) * 1000)
    finally:
        backend.close()
    elapsed = sum(latencies) / 1000.0
    return {
        "backend": kind,
        "status": "ok" if not errors else "error",
        "error": "; ".join(dict.fromkeys(errors)),
        "init_ms": round(init_ms, 3),
        "frames": len(latencies),
        "warmup_frames": warmup_count,
        "detections": int(detections),
        "detection_rate": round(detections / len(latencies), 4) if latencies else 0.0,
        "mean_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "p95_ms": round(_p95(latencies), 3),
        "fps": round(len(latencies) / elapsed, 3) if elapsed > 0 else 0.0,
    }


def _write_output(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark BarBeCue pose backends without opening a GUI.", epilog="Measures detect() only on identical cached 960x540 mirrored frames; excludes decode, preprocessing, coach logic and rendering. Initialization is reported separately. Use representative video for comparisons; dummy/static and blank-frame results are smoke checks only. RTMPose is currently a stub and will be reported unavailable.")
    parser.add_argument("--video", type=Path, help="video used for the comparison; omit for synthetic frames")
    parser.add_argument("--backends", default="mediapipe,yolo,rtmpose", help="comma-separated backend names (static/dummy also available for smoke checks)")
    parser.add_argument("--frames", type=int, default=60, help="number of frames to benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warm-up frames")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "pose_landmarker_full.task")
    parser.add_argument("--yolo-weights", help="explicit YOLO weights path; no implicit download is attempted")
    parser.add_argument("--output", type=Path, help="CSV or JSON result path")
    args = parser.parse_args(argv)
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup cannot be negative")
    video = args.video if not args.video or args.video.is_absolute() else Path.cwd() / args.video
    if video and not video.is_file():
        print(f"[BarBeCue/bench] video not found: {video}", file=sys.stderr)
        return 2
    try:
        frames, fps, source = _load_frames(video, args.frames)
    except (OSError, RuntimeError) as exc:
        print(f"[BarBeCue/bench] {exc}", file=sys.stderr)
        return 2
    names = [name.strip() for name in args.backends.split(",") if name.strip()]
    if not names:
        parser.error("--backends must contain at least one name")
    print(f"[BarBeCue/bench] detect() timing only; source={source} frames={len(frames)} source_fps={fps:.2f}")
    rows = []
    for name in names:
        row = benchmark_backend(name, frames, fps, args.warmup, args.model, args.yolo_weights)
        rows.append(row)
        if row["status"] == "ok":
            print(f"[BarBeCue/bench] {name}: {row['fps']:.2f} FPS, mean={row['mean_ms']:.2f} ms, detections={row['detections']}/{row['frames']}")
        else:
            print(f"[BarBeCue/bench] {name}: {row['status']} ({row['error']})")
    if args.output:
        _write_output(args.output, rows)
        print(f"[BarBeCue/bench] wrote {args.output}")
    return 0 if any(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
