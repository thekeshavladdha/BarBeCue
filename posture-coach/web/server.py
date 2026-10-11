"""BarBeCue web dashboard — browser UI over the same CoachState pipeline.

Run on the training machine (it owns the webcam), open in a browser:
    python web/server.py [--exercise Squat] [--port 5000]
    -> http://127.0.0.1:5000

One camera thread owns capture + CoachState; browsers are thin: an MJPEG
<img> for video plus 300 ms JSON polls for numbers. No build step, no CDN,
no external assets. Session logbook lives in memory (cleared on restart).
"""
from __future__ import annotations
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from flask import Flask, jsonify, render_template, request, Response

from app import (CoachState, open_capture, draw_skeleton, draw_angle_overlay,
                 draw_hud, WORK_W, WORK_H)
from pose_estimator import create_backend
from exercises import EXERCISES, list_categories, exercises_in

app = Flask(__name__, template_folder=str(HERE / "templates"),
            static_folder=str(HERE / "static"))

_lock = threading.Lock()
_pose = None
_state: CoachState | None = None
_cap = None
_offline = False
_file_fps = 30.0
_frame_jpg: bytes | None = None
_last_out: dict = {}
_fps = 30.0
_log: list = []


def _snapshot() -> dict:
    assert _state is not None
    out = dict(_last_out)
    return {
        "exercise": _state.name,
        "category": EXERCISES[_state.name].get("category", "Posture"),
        "profile": _state.profile,
        "profiles": _state.profiles(),
        "status": out.get("status", "starting"),
        "msgs": out.get("msgs", []),
        "stage": out.get("stage", "REST"),
        "reps": out.get("reps", 0),
        "bad_reps": out.get("bad_reps", 0),
        "bad_reason": out.get("bad_reason", ""),
        "hold": round(float(out.get("hold", 0.0)), 1),
        "event": out.get("event"),
        "scenario": out.get("scenario"),
        "view": _state.cfg.get("view", "side"),
        "fps": round(_fps, 0),
    }


def _camera_loop():
    global _frame_jpg, _last_out, _fps
    assert _pose is not None and _state is not None and _cap is not None
    n, fps, t_prev = 0, 30.0, time.time()
    while True:
        ok, frame = _cap.read()
        if not ok:
            if _offline:
                _cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WORK_W, WORK_H))
        now = n / _file_fps if _offline else time.time()
        kp_raw = _pose.detect(frame, int(now * 1000))
        with _lock:
            out = _state.step(kp_raw, now)
            drawn = _state.ema.state if _state.ema.state else (kp_raw or {})
            draw_skeleton(frame, drawn)
            if out.get("angle") is not None and out.get("side") \
                    and _state.cfg.get("metric"):
                draw_angle_overlay(frame, drawn, out["side"],
                                   _state.cfg["metric"], out["angle"])
            draw_hud(frame, _state, out, fps, _state.cfg.get("view", "side"))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _frame_jpg = buf.tobytes()
            _last_out = out
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        _fps = fps
        n += 1


def _mjpeg():
    while True:
        with _lock:
            jpg = _frame_jpg
        if jpg is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.033)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/exercises")
def api_exercises():
    return jsonify({"categories": {c: exercises_in(c) for c in list_categories()}})


@app.get("/api/state")
def api_state():
    with _lock:
        return jsonify(_snapshot())


@app.post("/api/select")
def api_select():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("exercise")
    if name not in EXERCISES:
        return jsonify({"error": f"unknown exercise {name!r}"}), 400
    with _lock:
        assert _state is not None
        _state.select(name)
        return jsonify(_snapshot())


@app.post("/api/profile")
def api_profile():
    data = request.get_json(force=True, silent=True) or {}
    with _lock:
        assert _state is not None
        name = data.get("profile", "")
        if name not in _state.profiles():
            return jsonify({"error": f"unknown profile {name!r}"}), 400
        _state.set_profile(name)
        return jsonify(_snapshot())


@app.post("/api/reset")
def api_reset():
    with _lock:
        assert _state is not None
        _state.reset_set()
        return jsonify(_snapshot())


@app.post("/api/calibrate")
def api_calibrate():
    with _lock:
        assert _state is not None
        _state.start_calibration(time.time())
        return jsonify({"status": "calibrating"})


@app.get("/api/log")
def api_log():
    return jsonify({"sets": list(_log)})


@app.post("/api/log_set")
def api_log_set():
    with _lock:
        assert _state is not None
        snap = _snapshot()
        _log.append({
            "at": time.strftime("%H:%M:%S"),
            "exercise": snap["exercise"],
            "reps": snap["reps"],
            "bad_reps": snap["bad_reps"],
            "hold": snap["hold"],
        })
        _state.reset_set()
        return jsonify({"sets": list(_log)})


def main(argv=None) -> None:
    global _pose, _state, _cap, _offline, _file_fps
    ap = argparse.ArgumentParser(description="BarBeCue web dashboard.")
    ap.add_argument("--exercise", default="Auto posture")
    ap.add_argument("--backend", default="mediapipe")
    ap.add_argument("--video", default=None)
    ap.add_argument("--lenient", action="store_true")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args(argv)

    _pose = create_backend(args.backend) \
        if args.backend not in ("static", "dummy") else create_backend("static")
    _state = CoachState(args.exercise, strict=not args.lenient, profile=args.profile)
    _cap, _offline, _file_fps = open_capture(args.video)

    threading.Thread(target=_camera_loop, daemon=True).start()
    print(f"[BarBeCue] web dashboard at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
