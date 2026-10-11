"""BarBeCue geometry.py — pixel-space joint angles and posture metrics.

LLD: pure functions, no camera/model/UI state. All inputs in pixels.
Must convert normalized x,y to pixels (x*W, y*H) BEFORE calling angle()
— non-square frames (960x540) distort angles otherwise.

Keypoint dict contract (from pose_estimator.detect):
    {name: (x_px, y_px, visibility)}
Names: nose, l_ear/r_ear, l_shoulder/r_shoulder, l_elbow/r_elbow,
       l_wrist/r_wrist, l_hip/r_hip, l_knee/r_knee, l_ankle/r_ankle,
       l_heel/r_heel, l_foot_index/r_foot_index
17-pt backends (YOLO/RTMPose) have no heel/foot_index -> knee_over_toe = 0.
"""
from __future__ import annotations
import math
import numpy as np

__all__ = [
    "angle",
    "tilt_from_vertical",
    "midpoint",
    "pick_side",
    "visible_ok",
    "face_offset_angle",
    "compute_metrics",
]

_EPS = 1e-9


def angle(a, b, c) -> float:
    """Joint angle at B from points A,B,C in degrees.

    Straight leg ~170-180, deep squat ~70-90.
    Eq 4.1: theta = arccos(BA.BC / |BA||BC|).
    """
    a = np.array(a, dtype=float)[:2]
    b = np.array(b, dtype=float)[:2]
    c = np.array(c, dtype=float)[:2]
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) + _EPS
    cosv = float(np.dot(ba, bc) / denom)
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))


def tilt_from_vertical(top, bottom) -> float:
    """Absolute tilt from vertical in degrees. 0 = perfectly upright.

    Eq 4.2: tilt = |arctan2(dx, dy)|, dx = top.x-bottom.x,
    dy = bottom.y-top.y (image y grows downward so sign flipped).
    """
    dx = float(top[0]) - float(bottom[0])
    dy = float(bottom[1]) - float(top[1])
    return float(abs(np.degrees(np.arctan2(dx, dy))))


def midpoint(p, q):
    return ((float(p[0]) + float(q[0])) / 2.0, (float(p[1]) + float(q[1])) / 2.0)


def _dist(p, q) -> float:
    return float(math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1])))


def pick_side(kp: dict) -> str:
    """Select visible side per frame by visibility sum. Never hardcode l/r.

    Mirror-safe: cv2.flip swaps image left/right, so hardcoded side fails
    when user turns. Sums shoulder+hip+knee+ankle visibility each side.
    """
    def score(s: str) -> float:
        total = 0.0
        for j in ("shoulder", "hip", "knee", "ankle"):
            v = kp.get(f"{s}_{j}")
            if v is not None:
                try:
                    total += float(v[2])
                except Exception:
                    pass
        return total

    return "l" if score("l") >= score("r") else "r"


def visible_ok(kp: dict, side: str, joints, thr: float = 0.5) -> bool:
    """True if every required joint on `side` has visibility >= thr.

    foot_index is optional (17-pt backends lack it) — skipped if absent.
    Caller shows 'Step back so your whole body is visible' when False
    and must skip rules/counting for that frame, never guess.
    """
    for j in joints:
        key = f"{side}_{j}"
        if key not in kp:
            if j == "foot_index":
                continue
            return False
        try:
            if float(kp[key][2]) < thr:
                return False
        except Exception:
            return False
    return True


def face_offset_angle(kp: dict) -> float | None:
    """Profile-view score in degrees, or None if unmeasurable.

    Side-view exercises (squat/lunge/push-up/plank) assume the camera sees
    the body from the side. When the user faces the camera the nose sits
    near the shoulder midpoint (score ~0); in profile the nose is far to
    one side relative to the (foreshortened) shoulder width::

        score = atan(|nose.x - shoulder_mid.x| / shoulder_width)

    Large value = good profile view; small value = too frontal. Caller
    warns when score < ALIGN_MIN (side-view exercises only). This mirrors
    LearnOpenCV's camera-alignment check (their OFFSET_THRESH frontal
    guard) adapted to pixel space. Threshold is a heuristic — tune with
    plot_angle.py on own clips. Returns None when nose or both shoulders
    are missing.
    """
    n = kp.get("nose")
    ls, rs = kp.get("l_shoulder"), kp.get("r_shoulder")
    if n is None or ls is None or rs is None:
        return None
    try:
        mid_x = (float(ls[0]) + float(rs[0])) / 2.0
        width = abs(float(ls[0]) - float(rs[0])) + _EPS
        dx = abs(float(n[0]) - mid_x) / width
        return float(np.degrees(np.arctan(dx)))
    except Exception:
        return None


def _get(kp: dict, side: str, joint: str):
    """Return (x, y) for side joint or None if missing."""
    v = kp.get(f"{side}_{joint}")
    if v is None:
        return None
    return (float(v[0]), float(v[1]))


def _hip_sag(shoulder, hip, ankle) -> float:
    """Signed hip deviation from shoulder-ankle line, normalised by length.

    sag = cross / len^2 = distance / len. Plank fault when |sag| > 0.06
    (~6% of torso-line length). Sign indicates above/below line.
    """
    sx, sy = float(shoulder[0]), float(shoulder[1])
    hx, hy = float(hip[0]), float(hip[1])
    ax, ay = float(ankle[0]), float(ankle[1])
    line_len = math.hypot(ax - sx, ay - sy) + _EPS
    cross = (ax - sx) * (hy - sy) - (ay - sy) * (hx - sx)
    return float(cross / (line_len * line_len + _EPS))


def compute_metrics(kp: dict, side: str) -> dict:
    """Compute all posture metrics for `side`. Missing joints -> 0 for that metric.

    knee: angle(hip,knee,ankle) — squat/lunge depth + rep metric
    hip: angle(shoulder,hip,knee)
    elbow: angle(shoulder,elbow,wrist) — push-up/curl/press rep metric
    body_line: angle(shoulder,hip,ankle) at hip — push-up/plank straightness
    back_lean: tilt(shoulder over hip) — 0 upright
    neck_tilt: tilt(ear/nose over shoulder)
    knee_over_toe: |knee.x-foot.x|/shin_len — 0 if foot_index absent (17-pt)
    elbow_drift: |elbow.x-shoulder.x|/upper_arm_len — curl isolation
    hip_sag: signed hip-to-line distance / line_len — plank
    head_hold: (shoulder.y-ear.y)/torso_len — chin-drop detector for
      desk/sitting. No pose model ships spine joints, so mid-back curvature
      is unobservable directly; but a slouch always drags the head down
      toward the chest, shrinking this gap even on a frontal laptop camera
      where the bend itself hides along the view axis. Upright ~0.45+ on a
      side/desk-recording setup, slouched below ~0.36 there (tuned on real
      desk footage, 2026-10-11: upright plateaus 0.42–0.46, slouch valleys
      0.30–0.33). NOTE: on a frontal laptop camera the ratio is near
      scale-invariant — leaning in grows gap and torso together — so live
      laptop sampling (same day) showed upright and slouch both ~0.30–0.38
      and the metric cannot separate them there. Sitting/Desk therefore use
      it only as a gross-collapse backstop (0.28) and rely on neck_tilt,
      which separated cleanly live (upright ~10, slouch ~23). Neutral 1.0
      when ear/shoulder/hip missing so an unseen head never counts as
      a fault.
    """
    shoulder = _get(kp, side, "shoulder")
    hip = _get(kp, side, "hip")
    knee = _get(kp, side, "knee")
    ankle = _get(kp, side, "ankle")
    elbow = _get(kp, side, "elbow")
    wrist = _get(kp, side, "wrist")

    # ear: prefer side ear, fallback to nose (central)
    ear = _get(kp, side, "ear")
    if ear is None:
        n = kp.get("nose")
        ear = (float(n[0]), float(n[1])) if n is not None else None

    foot = _get(kp, side, "foot_index")
    if foot is None:  # 17-pt backends: heel/toe absent
        foot = _get(kp, side, "heel")

    m = {
        "knee": 0.0, "hip": 0.0, "elbow": 0.0, "body_line": 0.0,
        "back_lean": 0.0, "neck_tilt": 0.0,
        "knee_over_toe": 0.0, "elbow_drift": 0.0, "hip_sag": 0.0,
        "head_hold": 1.0,  # neutral when unmeasurable: never punish unseen head
    }

    try:
        if hip and knee and ankle:
            m["knee"] = angle(hip, knee, ankle)
    except Exception:
        pass
    try:
        if shoulder and hip and knee:
            m["hip"] = angle(shoulder, hip, knee)
    except Exception:
        pass
    try:
        if shoulder and elbow and wrist:
            m["elbow"] = angle(shoulder, elbow, wrist)
    except Exception:
        pass
    try:
        if shoulder and hip and ankle:
            m["body_line"] = angle(shoulder, hip, ankle)
    except Exception:
        pass
    try:
        if shoulder and hip:
            m["back_lean"] = tilt_from_vertical(shoulder, hip)
    except Exception:
        pass
    try:
        if ear is not None and shoulder is not None:
            m["neck_tilt"] = tilt_from_vertical(ear, shoulder)
    except Exception:
        pass
    try:
        if knee is not None and foot is not None and ankle is not None:
            # foot_index present only on 33-pt; else stays 0
            if f"{side}_foot_index" in kp or f"{side}_heel" in kp:
                shin = _dist(knee, ankle) + _EPS
                m["knee_over_toe"] = abs(float(knee[0]) - float(foot[0])) / shin
    except Exception:
        pass
    try:
        if shoulder and elbow:
            upper = _dist(shoulder, elbow) + _EPS
            m["elbow_drift"] = abs(float(elbow[0]) - float(shoulder[0])) / upper
    except Exception:
        pass
    try:
        if shoulder and hip and ankle:
            m["hip_sag"] = _hip_sag(shoulder, hip, ankle)
    except Exception:
        pass
    try:
        # Chin-drop: slouching collapses the ear-shoulder vertical gap even
        # when the spinal bend itself is hidden along the camera axis.
        if ear is not None and shoulder is not None and hip is not None:
            torso = _dist(shoulder, hip) + _EPS
            m["head_hold"] = (float(shoulder[1]) - float(ear[1])) / torso
    except Exception:
        pass
    return m
