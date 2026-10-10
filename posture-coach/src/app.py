"""BarBeCue app.py — Tkinter + OpenCV live coach (Day 4).

Run: python app.py [--video data/squat.mp4] [--backend mediapipe] [--exercise Squat]
Default: webcam 0->1->2, 1280x720 request, work at 960x540, mirror with flip.
Pipeline per frame (Alg B.1): read+mirror -> detect(ts mono) -> smooth ->
pick_side -> draw skeleton (vis>0.5 amber lines/red dots) -> vis gate ->
metrics -> angle smooth (0.6/0.4) -> counter + feedback -> merge partial
flash (1.5s precedence) -> hold timer (pause, not reset, on fault) -> HUD.

Dropdown txn: EXERCISES[name] lookup, fresh RepCounter, Feedback.reset(),
smoothing.reset(), hold=0, view hint update. Camera loop never restarts.
"""
from __future__ import annotations
import argparse
import time
import cv2

from pose_estimator import create_backend
from smoothing import EMA, AngleSmoother
from geometry import pick_side, visible_ok, compute_metrics
from exercises import EXERCISES, list_exercises
from rep_counter import RepCounter
from feedback import Feedback, raw_violations
from posture import ScenarioClassifier, LegShakeDetector, SHAKE_MSG

WORK_W, WORK_H = 960, 540
VIS_TH = 0.5
PARTIAL_HOLD = 1.5

# skeleton links drawn only when both endpoints vis>0.5
SKELETON = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_ankle", "l_foot_index"), ("r_ankle", "r_foot_index"),
]

HUD_AMBER = (0, 165, 255)
HUD_RED = (0, 0, 255)
HUD_GREEN = (0, 200, 0)


class CoachState:
    """GUI-independent pipeline state — unit-testable without Tk/camera.

    Strict-form counting (default ON): raw faults of severity>=strict_sev
    latched at ANY point of a rep cycle invalidate it ('bad_rep', not
    counted). Sev 1 style tips never block. Pass strict=False for the old
    lenient behaviour (depth-only counting, hints advisory only).
    """

    def __init__(self, exercise: str = "Auto posture", strict: bool = True, strict_sev: int = 2):
        self.fb = Feedback(debounce=0.6, hold=1.5, max_msgs=2)
        self.ema = EMA(0.5)
        self.ang_sm = AngleSmoother(0.6)
        self.strict = strict
        self.strict_sev = int(strict_sev)
        self.scenario_cls = ScenarioClassifier()
        self.shake = LegShakeDetector()
        self.scenario = "standing"
        self.baseline = None  # {"back_lean":x, "neck_tilt":y}: upright zero
        self._calib_until = None
        self._calib_samples = []
        self.hold_secs = 0.0
        self._last_t = None
        self.partial_until = -1e9
        self.partial_msg = ""
        self.bad_until = -1e9
        self.bad_msg = ""
        self._rep_dirty = False
        self._rep_bad_msg = ""
        self.select(exercise)

    def select(self, name: str):
        self.name = name
        self.cfg = EXERCISES[name]
        self.counter = RepCounter(self.cfg)
        self.fb.reset()
        self.ema.reset()
        self.ang_sm.reset()
        self.hold_secs = 0.0
        self._last_t = None
        self.partial_until = -1e9
        self.bad_until = -1e9
        self._rep_dirty = False
        self._rep_bad_msg = ""
        self.scenario_cls.reset()
        self.shake.reset()
        self.scenario = "standing"
        return self.cfg.get("view", "side")

    def reset_set(self):
        self.select(self.name)

    def start_calibration(self, now: float, duration: float = 3.0):
        """Capture current upright as zero. Sit/stand naturally until done."""
        self._calib_until = float(now) + float(duration)
        self._calib_samples = []
        return float(duration)

    def _finalize_calibration(self):
        samples = self._calib_samples
        self._calib_until = None
        self._calib_samples = []
        if len(samples) < 10:
            print(f"[BarBeCue] calibration discarded ({len(samples)} frames, need 10+)")
            return
        def median(vals):
            s = sorted(vals)
            n = len(s)
            return (s[n // 2] + s[(n - 1) // 2]) / 2.0
        self.baseline = {
            "back_lean": median([s[0] for s in samples]),
            "neck_tilt": median([s[1] for s in samples]),
        }
        print(f"[BarBeCue] calibrated upright: {self.baseline}")

    def _calibrated_metrics(self, m: dict) -> dict:
        # Rules then mean "degrees WORSE than your upright", absorbing body
        # proportions, camera tilt and consistent hip-guess offset at once.
        if not self.baseline:
            return m
        eff = dict(m)
        for k in ("back_lean", "neck_tilt"):
            if k in eff:
                eff[k] = max(0.0, float(eff[k]) - float(self.baseline.get(k, 0.0)))
        return eff

    def step(self, kp_raw: dict, now: float):
        """One logic tick. Returns dict for render. Pure apart from state."""
        cfg = self.cfg
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if not kp_raw:
            return {"status": "no_person", "msgs": [], "reps": self.counter.reps,
                    "stage": self.counter.stage, "hold": self.hold_secs, "metrics": {}}
        kp = self.ema(kp_raw)
        side = pick_side(kp)
        if not visible_ok(kp, side, cfg["joints"], VIS_TH):
            return {"status": "step_back", "msgs": [], "side": side,
                    "reps": self.counter.reps, "stage": self.counter.stage,
                    "hold": self.hold_secs, "metrics": {}}
        m = compute_metrics(kp, side)
        if self._calib_until is not None:
            if now < self._calib_until:
                self._calib_samples.append(
                    (m.get("back_lean", 0.0), m.get("neck_tilt", 0.0)))
                return {"status": "calibrating", "msgs": [], "side": side,
                        "reps": self.counter.reps, "stage": self.counter.stage,
                        "hold": self.hold_secs, "metrics": m,
                        "remaining": round(self._calib_until - now, 1),
                        "scenario": getattr(self, "scenario", None)}
            self._finalize_calibration()
        meff = self._calibrated_metrics(m)
        event = None
        scenario = None
        if cfg["mode"] == "auto":
            ang = None
            self.scenario = self.scenario_cls.update(kp, side, m.get("knee", 0.0), now)
            scenario = self.scenario
            scfg = cfg["scenarios"][self.scenario]
            ax = kp.get(f"{side}_ankle")
            if ax is not None and float(ax[2]) >= VIS_TH:
                foot_x = float(ax[0])
            else:
                kx = kp.get(f"{side}_knee")
                foot_x = float(kx[0]) if kx is not None and float(kx[2]) >= VIS_TH else None
            shaking = self.shake.update(foot_x, now)
            msgs = self.fb.update(scfg["rules"], meff, now)
            if self.scenario == "sitting" and shaking:
                msgs = [SHAKE_MSG] + msgs
                msgs = msgs[:2]
            if not msgs:
                self.hold_secs += dt
            # pause (not reset) on fault, like hold mode
        elif cfg["mode"] == "reps":
            raw = m.get(cfg["metric"], 0.0)
            ang = self.ang_sm(raw)
            if self.strict:
                blocking = [r for r in raw_violations(cfg["rules"], meff)
                            if r.get("severity", 1) >= self.strict_sev]
                if blocking:
                    blocking.sort(key=lambda r: -r.get("severity", 1))
                    self._rep_dirty = True
                    self._rep_bad_msg = blocking[0]["msg"]
                elif not self.counter.moving:
                    self._rep_dirty = False
                    self._rep_bad_msg = ""
            event = self.counter.update(ang, now, valid=not self._rep_dirty)
            if event in ("rep", "bad_rep"):
                self._rep_dirty = False
            if event == "partial":
                self.partial_until = now + PARTIAL_HOLD
                self.partial_msg = cfg.get("partial_msg", "Go lower")
            elif event == "bad_rep":
                self.bad_until = now + PARTIAL_HOLD
                self.bad_msg = f"No rep - {self._rep_bad_msg}" if self._rep_bad_msg else "No rep - bad form"
                self._rep_bad_msg = ""
            msgs = self.fb.update(cfg["rules"], meff, now)
        else:  # hold mode (auto branch above sets its own msgs)
            ang = None
            msgs = self.fb.update(cfg["rules"], meff, now)
            if not msgs:
                self.hold_secs += dt
            # pause (not reset) on fault by design
        if now <= self.bad_until and self.bad_msg:
            msgs = [self.bad_msg]
        elif now <= self.partial_until and self.partial_msg:
            msgs = [self.partial_msg]
        return {"status": "ok", "msgs": msgs, "side": side, "metrics": m,
                "reps": self.counter.reps, "stage": self.counter.stage,
                "hold": self.hold_secs, "event": event, "angle": ang,
                "scenario": scenario}


def draw_skeleton(frame, kp: dict):
    for a, b in SKELETON:
        pa, pb = kp.get(a), kp.get(b)
        if pa is None or pb is None:
            continue
        if pa[2] < VIS_TH or pb[2] < VIS_TH:
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), HUD_AMBER, 2)
    for _, (x, y, v) in kp.items():
        if v >= VIS_TH:
            cv2.circle(frame, (int(x), int(y)), 4, HUD_RED, -1)


def draw_hud(frame, state: CoachState, out: dict, fps: float, view_hint: str,
             voice_status: str = ""):
    cfg = state.cfg
    if cfg.get("mode") == "auto":
        title = f"{state.name} - {getattr(state, 'scenario', 'standing')}"
    else:
        title = f"{state.name} ({view_hint} view)"
    cv2.putText(frame, title, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if cfg["mode"] == "reps":
        cv2.putText(frame, f"REPS {out.get('reps',0)}", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"{out.get('stage','REST')}", (12, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"HOLD {out.get('hold',0.0):.1f}s", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1]-110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    if voice_status:
        cv2.putText(frame, f"Voice {voice_status} (M)", (frame.shape[1]-205, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    if getattr(state, "baseline", None):
        cv2.putText(frame, "Calibrated", (frame.shape[1]-135, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if out.get("status") == "calibrating":
        cv2.putText(frame, f"Sit/stand naturally... {out.get('remaining', 0)}s", (12, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_AMBER, 2)
    elif out.get("status") == "no_person":
        cv2.putText(frame, "No person detected", (12, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_AMBER, 2)
    elif out.get("status") == "step_back":
        cv2.putText(frame, "Step back so your whole body is visible", (12, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_AMBER, 2)
    else:
        msgs = out.get("msgs", [])
        if not msgs:
            cv2.putText(frame, "Good form", (12, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, HUD_GREEN, 2)
        else:
            y = 150
            for msg in msgs[:2]:
                cv2.putText(frame, msg, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
                y += 36


def open_capture(video: str | None):
    if video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video file {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return cap, True, fps
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[BarBeCue] camera index {idx}")
            return cap, False, 30.0
    raise RuntimeError("no camera 0/1/2. Close video-call apps or use --video data/file.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--backend", default="mediapipe")
    ap.add_argument("--exercise", default="Auto posture", choices=list_exercises())
    ap.add_argument("--lenient", action="store_true",
                    help="count on depth alone (old behaviour); default is strict-form: "
                         "severity>=2 faults invalidate the rep")
    args = ap.parse_args()

    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    pose = create_backend(args.backend) if args.backend not in ("static", "dummy") else create_backend("static")
    # MediaPipePose needs explicit model path default; create_backend handles it
    state = CoachState(args.exercise, strict=not args.lenient)
    cap, offline, file_fps = open_capture(args.video)

    root = tk.Tk()
    root.title("BarBeCue")
    top = ttk.Frame(root, padding=6)
    top.pack(fill="x")
    ttk.Label(top, text="Exercise:").pack(side="left")
    combo = ttk.Combobox(top, values=list_exercises(), state="readonly", width=16)
    combo.set(state.name)
    combo.pack(side="left", padx=6)
    def hint_for(cfg):
        if cfg.get("mode") == "auto":
            return "Sit or stand naturally, upper body visible"
        return f"Stand {cfg.get('view','side')} view, 2-3m, full body"

    hint = ttk.Label(top, text=hint_for(state.cfg))
    hint.pack(side="left", padx=6)
    ttk.Button(top, text="Reset", command=state.reset_set).pack(side="right")
    ttk.Button(top, text="Calibrate",
               command=lambda: [state.start_calibration(time.time()),
                                print("[BarBeCue] calibrating upright 3s: sit/stand naturally")]).pack(side="right", padx=6)
    video_label = ttk.Label(root)
    video_label.pack()
    root.bind("c", lambda _e: state.start_calibration(time.time()))
    root.bind("C", lambda _e: state.start_calibration(time.time()))

    n = 0
    t_prev = time.time()
    fps = 30.0
    last_img = None

    def on_select(_e=None):
        view = state.select(combo.get())
        hint.config(text=hint_for(state.cfg))

    combo.bind("<<ComboboxSelected>>", on_select)

    def tick():
        nonlocal n, t_prev, fps, last_img
        ok, frame = cap.read()
        if not ok:
            if offline:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return root.after(10, tick)
            root.after(10, tick)
            return
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WORK_W, WORK_H))
        if offline:
            now = n / file_fps
            ts_ms = int(now * 1000)
        else:
            now = time.time()
            ts_ms = int(now * 1000)
        kp_raw = pose.detect(frame, ts_ms)
        # smooth for drawing: reuse state's ema output shape via second pass? draw raw filtered:
        out = state.step(kp_raw, now)
        # draw uses ema state for stability
        draw_skeleton(frame, state.ema.state if state.ema.state else (kp_raw or {}))
        t_now = time.time()
        inst = 1.0 / max(t_now - t_prev, 1e-6)
        fps = 0.9 * fps + 0.1 * inst
        t_prev = t_now
        draw_hud(frame, state, out, fps, state.cfg.get("view", "side"))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        last_img = img
        video_label.configure(image=img)
        n += 1
        root.after(10, tick)

    def on_close():
        try:
            cap.release()
        except Exception:
            pass
        try:
            pose.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(10, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
