"""BarBeCue app.py — Tkinter + OpenCV live coach (Day 4).

Run: python app.py [--video data/squat.mp4] [--backend mediapipe] [--exercise Squat]
Default: webcam 0->1->2, 1280x720 request, work at 960x540, mirror with flip.
Pipeline per frame (Alg B.1): read+mirror -> detect(ts mono) -> smooth ->
pick_side -> draw skeleton (vis>0.5 amber lines/red dots) -> vis gate ->
side-view alignment guard (bad_view) -> metrics -> angle smooth (0.6/0.4) ->
phase track -> counter + feedback (phase-gated, 0.3s fault latch) -> merge
partial/bad flash (1.5s precedence, bad tallied) -> hold timer (pause, not
reset, on fault) -> HUD (tallies, angle overlay, audio on M).

Dropdown txn: EXERCISES[name] lookup, fresh RepCounter, Feedback.reset(),
smoothing.reset(), hold=0, view hint update. Camera loop never restarts.
"""
from __future__ import annotations
import argparse
import time
import cv2

from pose_estimator import create_backend
from smoothing import EMA, AngleSmoother
from geometry import pick_side, visible_ok, face_offset_angle, compute_metrics
from exercises import EXERCISES, list_exercises
from rep_counter import RepCounter
from feedback import Feedback, raw_violations
from posture import ScenarioClassifier, LegShakeDetector, SHAKE_MSG

WORK_W, WORK_H = 960, 540
VIS_TH = 0.5
PARTIAL_HOLD = 1.5
# Side-view guard: face_offset_angle() below this = too frontal to trust
# side-view joint angles (mirrors LearnOpenCV's camera-alignment check).
ALIGN_MIN = 20.0
# No usable person for this long -> explicit "paused" status (timers frozen).
PAUSE_AFTER = 15.0
# Strict-form latch delay: a sev>=strict_sev fault must persist this long
# (time-based, FPS-independent) before it invalidates the rep. Ignores
# single-frame jitter without weakening real-form enforcement.
FAULT_LATCH = 0.3
# Minimum gap between spoken cues so audio never spams.
AUDIO_COOLDOWN = 3.0

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
    persisting >= FAULT_LATCH (0.3s) at ANY point of a rep cycle invalidate
    it ('bad_rep', tallied in bad_reps, not counted). Sev 1 style tips
    never block. Pass strict=False for the old lenient behaviour
    (depth-only counting, hints advisory only). Bad reps keep
    last_bad_reason so the HUD shows `REPS n (m bad - <reason>)`.

    Profiles: exercises may ship Beginner/Pro threshold sets
    (EXERCISES[name]["profiles"]); set_profile() applies one without
    restarting the camera loop. Phase-gated rules (phase="descending")
    only fire mid-rep on the way down.
    """

    def __init__(self, exercise: str = "Auto posture", strict: bool = True, strict_sev: int = 2,
                 profile: str | None = None):
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
        self._fault_since = None  # when current blocking fault first seen
        self.bad_reps = 0
        self.last_bad_reason = ""
        self.phase = None  # "descending" | "ascending" | None
        self._prev_ang = None
        self._last_usable = None  # last time a person was measurably present
        self.profile = None
        self.select(exercise, profile=profile)

    @staticmethod
    def _resolve_cfg(name: str, profile: str | None) -> tuple:
        """Return (cfg copy with profile overrides applied, profile name).

        Profiles only override start_thr/end_thr and rule values by msg;
        "Beginner" (empty) == base behaviour, so headless/evaluate runs
        without a profile are exact regressions of old behaviour.
        """
        import copy
        base = EXERCISES[name]
        profiles = base.get("profiles") or {}
        if profile is None:
            profile = next(iter(profiles), "Standard") if profiles else "Standard"
        cfg = copy.deepcopy({k: v for k, v in base.items() if k != "profiles"})
        over = profiles.get(profile, {})
        if "start_thr" in over:
            cfg["start_thr"] = over["start_thr"]
        if "end_thr" in over:
            cfg["end_thr"] = over["end_thr"]
        for msg, val in (over.get("rule_values") or {}).items():
            for r in cfg.get("rules", []):
                if r.get("msg") == msg:
                    r["value"] = val
        return cfg, profile

    def profiles(self) -> list:
        base = EXERCISES.get(self.name, {})
        profs = base.get("profiles")
        return list(profs) if profs else ["Standard"]

    def set_profile(self, profile: str):
        """Switch difficulty profile mid-session; resets set like select()."""
        self.select(self.name, profile=profile, keep_tallies=False)

    def select(self, name: str, profile: str | None = None, keep_tallies: bool = False):
        self.name = name
        self.cfg, self.profile = self._resolve_cfg(name, profile)
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
        self._fault_since = None
        self.phase = None
        self._prev_ang = None
        if not keep_tallies:
            self.bad_reps = 0
            self.last_bad_reason = ""
        self.scenario_cls.reset()
        self.shake.reset()
        self.scenario = "standing"
        return self.cfg.get("view", "side")

    def reset_set(self):
        self.select(self.name, profile=self.profile)

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

    def _idle_out(self, now: float, base: dict) -> dict:
        """Paused status after PAUSE_AFTER of no usable person (timers frozen).

        Unlike the reference project's reset-on-inactivity, tallies and hold
        time are preserved — the session just explicitly pauses instead of
        silently idling.
        """
        if self._last_usable is None:
            self._last_usable = now  # anchor session start
        if now - self._last_usable >= PAUSE_AFTER:
            out = dict(base)
            out["status"] = "paused"
            out["msgs"] = ["Paused - no person detected"]
            return out
        return base

    def _reps_tally(self) -> dict:
        return {"reps": self.counter.reps, "bad_reps": self.bad_reps,
                "bad_reason": self.last_bad_reason}

    def step(self, kp_raw: dict, now: float):
        """One logic tick. Returns dict for render. Pure apart from state."""
        cfg = self.cfg
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if not kp_raw:
            return self._idle_out(now, {"status": "no_person", "msgs": [],
                                        "stage": self.counter.stage, "hold": self.hold_secs,
                                        "metrics": {}, **self._reps_tally()})
        kp = self.ema(kp_raw)
        side = pick_side(kp)
        if not visible_ok(kp, side, cfg["joints"], VIS_TH):
            return self._idle_out(now, {"status": "step_back", "msgs": [], "side": side,
                                        "stage": self.counter.stage,
                                        "hold": self.hold_secs, "metrics": {},
                                        **self._reps_tally()})
        if cfg.get("view") == "side":
            # Camera-alignment guard: all joints visible but frontal means
            # side-view angles are meaningless — skip counting, keep drawing.
            view_score = face_offset_angle(kp)
            if view_score is not None and view_score < ALIGN_MIN:
                self._last_usable = now
                return {"status": "bad_view", "msgs": ["Turn sideways to the camera"],
                        "side": side, "stage": self.counter.stage,
                        "hold": self.hold_secs, "metrics": {},
                        "view_score": round(view_score, 1), **self._reps_tally()}
        self._last_usable = now
        m = compute_metrics(kp, side)
        if self._calib_until is not None:
            if now < self._calib_until:
                self._calib_samples.append(
                    (m.get("back_lean", 0.0), m.get("neck_tilt", 0.0)))
                return {"status": "calibrating", "msgs": [], "side": side,
                        "stage": self.counter.stage,
                        "hold": self.hold_secs, "metrics": m,
                        "remaining": round(self._calib_until - now, 1),
                        "scenario": getattr(self, "scenario", None),
                        **self._reps_tally()}
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
            # Rep-cycle phase from smoothed-angle direction (jitter-gated):
            # phase-gated rules (phase="descending") only fire on the way
            # down, never at rest or on the way up.
            dec = (cfg.get("start_thr") or 0) > (cfg.get("end_thr") or 0)
            self.phase = None
            if self._prev_ang is not None and self.counter.moving:
                diff = ang - self._prev_ang
                if abs(diff) > 0.5:
                    down = diff < 0 if dec else diff > 0
                    self.phase = "descending" if down else "ascending"
            self._prev_ang = ang
            if self.strict:
                blocking = [r for r in raw_violations(cfg["rules"], meff, self.phase)
                            if r.get("severity", 1) >= self.strict_sev]
                if blocking:
                    blocking.sort(key=lambda r: -r.get("severity", 1))
                    self._rep_bad_msg = blocking[0]["msg"]
                    # Latch only after FAULT_LATCH of continuous fault, so a
                    # single noisy frame can't kill an otherwise good rep.
                    if self._fault_since is None:
                        self._fault_since = now
                    if now - self._fault_since >= FAULT_LATCH:
                        self._rep_dirty = True
                else:
                    self._fault_since = None
                    if not self.counter.moving:
                        self._rep_dirty = False
                        self._rep_bad_msg = ""
            event = self.counter.update(ang, now, valid=not self._rep_dirty)
            if event in ("rep", "bad_rep"):
                self._rep_dirty = False
                self._fault_since = None
            if event == "partial":
                self.partial_until = now + PARTIAL_HOLD
                self.partial_msg = cfg.get("partial_msg", "Go lower")
            elif event == "bad_rep":
                self.bad_reps += 1
                self.last_bad_reason = self._rep_bad_msg
                self.bad_until = now + PARTIAL_HOLD
                self.bad_msg = f"No rep - {self._rep_bad_msg}" if self._rep_bad_msg else "No rep - bad form"
                self._rep_bad_msg = ""
            msgs = self.fb.update(cfg["rules"], meff, now, self.phase)
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
                "stage": self.counter.stage,
                "hold": self.hold_secs, "event": event, "angle": ang,
                "phase": self.phase, "profile": self.profile,
                "scenario": scenario, **self._reps_tally()}


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


ANGLE_JOINTS = {
    # rep metric -> (jointA, jointB-at-vertex, jointC) for the overlay
    "knee": ("hip", "knee", "ankle"),
    "elbow": ("shoulder", "elbow", "wrist"),
}


def draw_angle_overlay(frame, kp: dict, side: str, metric: str, value: float):
    """Draw the measured rep angle at its joint: rays + numeric value.

    Makes thresholds transparent live (same idea as the reference
    project's ellipse arcs) and explains every bad_rep. No-op when any
    of the three joints is missing/hidden.
    """
    trip = ANGLE_JOINTS.get(metric)
    if trip is None or value is None:
        return
    pts = [kp.get(f"{side}_{j}") for j in trip]
    if any(p is None or p[2] < VIS_TH for p in pts):
        return
    (ax, ay), (bx, by), (cx, cy) = [(float(p[0]), float(p[1])) for p in pts]
    import math as _math
    r = 34.0
    for px, py in ((ax, ay), (cx, cy)):
        dx, dy = px - bx, py - by
        n = _math.hypot(dx, dy) or 1.0
        ex, ey = int(bx + dx / n * r), int(by + dy / n * r)
        cv2.line(frame, (int(bx), int(by)), (ex, ey), HUD_GREEN, 2)
    cv2.circle(frame, (int(bx), int(by)), 5, HUD_GREEN, -1)
    cv2.putText(frame, f"{float(value):.0f}", (int(bx) + 10, int(by) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_GREEN, 2)


def draw_hud(frame, state: CoachState, out: dict, fps: float, view_hint: str,
             voice_status: str = ""):
    cfg = state.cfg
    if cfg.get("mode") == "auto":
        title = f"{state.name} - {getattr(state, 'scenario', 'standing')}"
    else:
        title = f"{state.name} ({view_hint} view)"
    if getattr(state, "profile", "Standard") not in (None, "Standard"):
        title += f" [{state.profile}]"
    cv2.putText(frame, title, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if cfg["mode"] == "reps":
        tally = f"REPS {out.get('reps',0)}"
        if out.get("bad_reps"):
            tally += f"  BAD {out['bad_reps']}"
            if out.get("bad_reason"):
                tally += f" ({out['bad_reason']})"
        cv2.putText(frame, tally, (12, 70),
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
    elif out.get("status") == "bad_view":
        cv2.putText(frame, "Turn sideways to the camera", (12, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_AMBER, 2)
    elif out.get("status") == "paused":
        cv2.putText(frame, "Paused - no person detected", (12, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_AMBER, 2)
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


class AudioCues:
    """Optional spoken cues (rep counts, severe faults). Silent no-op when
    pyttsx3 is missing or disabled. Speech runs on a daemon thread so the
    camera loop never blocks; AUDIO_COOLDOWN gaps utterances."""

    def __init__(self, enabled: bool = True):
        import threading as _th
        self._th = _th
        self.cooldown = AUDIO_COOLDOWN
        self._last = -1e9
        self._engine = None
        self.available = False
        self.enabled = False
        if enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self.available = True
                self.enabled = True
            except Exception as exc:
                print(f"[BarBeCue] audio unavailable ({exc}); run silent")

    @property
    def status(self) -> str:
        if not self.available:
            return "missing"
        return "on" if self.enabled else "off"

    def toggle(self):
        if self.available:
            self.enabled = not self.enabled
        return self.status

    def say(self, text: str, now: float):
        if not self.available or not self.enabled or not text:
            return
        if now - self._last < self.cooldown:
            return
        self._last = now

        def _speak():
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass

        self._th.Thread(target=_speak, daemon=True).start()


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
    ap.add_argument("--profile", default=None,
                    help="difficulty profile (e.g. Beginner/Pro where offered); "
                         "default is the exercise base profile")
    ap.add_argument("--no-audio", action="store_true",
                    help="disable spoken rep/fault cues (pyttsx3 optional)")
    args = ap.parse_args()

    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    pose = create_backend(args.backend) if args.backend not in ("static", "dummy") else create_backend("static")
    # MediaPipePose needs explicit model path default; create_backend handles it
    state = CoachState(args.exercise, strict=not args.lenient, profile=args.profile)
    audio = AudioCues(enabled=not args.no_audio)
    cap, offline, file_fps = open_capture(args.video)

    root = tk.Tk()
    root.title("BarBeCue")
    top = ttk.Frame(root, padding=6)
    top.pack(fill="x")
    ttk.Label(top, text="Exercise:").pack(side="left")
    combo = ttk.Combobox(top, values=list_exercises(), state="readonly", width=16)
    combo.set(state.name)
    combo.pack(side="left", padx=6)
    ttk.Label(top, text="Profile:").pack(side="left")
    prof_combo = ttk.Combobox(top, values=state.profiles(), state="readonly", width=10)
    prof_combo.set(state.profile)
    prof_combo.pack(side="left", padx=6)
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
    root.bind("m", lambda _e: print(f"[BarBeCue] audio {audio.toggle()}"))
    root.bind("M", lambda _e: print(f"[BarBeCue] audio {audio.toggle()}"))

    n = 0
    t_prev = time.time()
    fps = 30.0
    last_img = None

    def on_select(_e=None):
        view = state.select(combo.get())
        hint.config(text=hint_for(state.cfg))
        prof_combo.config(values=state.profiles())
        prof_combo.set(state.profile)

    def on_profile(_e=None):
        state.set_profile(prof_combo.get())
        hint.config(text=hint_for(state.cfg))

    combo.bind("<<ComboboxSelected>>", on_select)
    prof_combo.bind("<<ComboboxSelected>>", on_profile)

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
        drawn = state.ema.state if state.ema.state else (kp_raw or {})
        draw_skeleton(frame, drawn)
        if out.get("angle") is not None and out.get("side") and state.cfg.get("metric"):
            draw_angle_overlay(frame, drawn, out["side"], state.cfg["metric"], out["angle"])
        # spoken cues: rep counts + bad reps only (severe live faults stay
        # visual to avoid mid-set chatter); cooldown-gated in AudioCues.
        if out.get("event") == "rep":
            audio.say(f"{out.get('reps', 0)}", now)
        elif out.get("event") == "bad_rep":
            audio.say("incorrect", now)
        t_now = time.time()
        inst = 1.0 / max(t_now - t_prev, 1e-6)
        fps = 0.9 * fps + 0.1 * inst
        t_prev = t_now
        draw_hud(frame, state, out, fps, state.cfg.get("view", "side"),
                 voice_status=audio.status)
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
