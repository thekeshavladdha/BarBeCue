"""Live posture sampler: grabs camera frames, runs MediaPipe, prints metrics.

Usage: python sampler.py --label upright --seconds 3
Run when src/app.py is NOT holding the camera. Prints per-metric mean/min/max
over the window so thresholds can be set from live postures.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from pose_estimator import create_backend
from geometry import pick_side, visible_ok, compute_metrics

WORK_W, WORK_H = 960, 540
VIS_TH = 0.5
JOINTS = ["shoulder", "hip"]  # desk-style gating


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="sample")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--save", default=None, help="save last frame PNG here")
    args = ap.parse_args()

    pose = create_backend("mediapipe")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("cannot open camera (is src/app.py still running?)")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(5):  # warm up exposure/autofocus
        cap.read()
        time.sleep(0.1)

    series = {m: [] for m in ("head_hold", "back_lean", "neck_tilt")}
    n_frames, n_pose, frame = 0, 0, None
    t_end = time.time() + args.seconds
    i = 0
    WIN = f"Sampler - {args.label} (hold the posture)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WORK_W, WORK_H))
        left = t_end - time.time()
        cv2.putText(frame, f"{args.label}: {left:.1f}s", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)
        kp = pose.detect(frame, int(time.time() * 1000))
        n_frames += 1
        if not kp:
            continue
        side = pick_side(kp)
        if not visible_ok(kp, side, JOINTS, VIS_TH):
            continue
        n_pose += 1
        m = compute_metrics(kp, side)
        for k in series:
            series[k].append(m[k])
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    try:
        pose.close()
    except Exception:
        pass

    print(f"[{args.label}] frames={n_frames} usable={n_pose}")
    for k, vals in series.items():
        if vals:
            print(f"  {k}: mean={sum(vals)/len(vals):.3f} "
                  f"min={min(vals):.3f} max={max(vals):.3f}")
        else:
            print(f"  {k}: NO USABLE FRAMES (step back / check light)")
    if args.save and frame is not None:
        cv2.imwrite(args.save, frame)
        print(f"  saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
