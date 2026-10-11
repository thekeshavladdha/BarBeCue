"""BarBeCue posture.py — ambient posture scenario + leg-shake detectors.

Used by the default `Auto posture` exercise: no counting, just ongoing cues
for whatever the body is doing right now (desk sitting vs standing).
- ScenarioClassifier: knee angle with hysteresis + 0.5s latch. Sitting when
  knees stay bent (chair), standing when legs stay straight. Legs hidden
  (desk/chair occlusion) assumes sitting — the common occluded case.
- LegShakeDetector: ring buffer of ankle x; flags rhythmic fidgeting
  (high zero-crossing rate + spread, low net drift) so still legs pass.

Rounded shoulders are NOT directly claimed: no pose model ships spine
joints, and forward-shoulder depth needs side-view 3D/world landmarks
(future work). Slouch is instead caught two ways: back_lean + neck_tilt
for visible lean, and the head_hold chin-drop gap (geometry.py) for
mid-back rounding whose bend hides along a frontal camera axis.
Thresholds are heuristics — retune on own clips, never clinical.
"""
from __future__ import annotations
from collections import deque

__all__ = ["ScenarioClassifier", "LegShakeDetector", "SHAKE_MSG"]

SHAKE_MSG = "Keep your legs still"

SIT_ENTER = 125.0
STAND_EXIT = 145.0
LATCH_S = 0.5
VIS_TH = 0.5


class ScenarioClassifier:
    def __init__(self, sit_enter: float = SIT_ENTER, stand_exit: float = STAND_EXIT,
                 latch_s: float = LATCH_S):
        self.sit_enter = float(sit_enter)
        self.stand_exit = float(stand_exit)
        self.latch_s = float(latch_s)
        self.scenario = "standing"
        self._pending = None
        self._since = 0.0

    def reset(self):
        self.scenario = "standing"
        self._pending = None
        self._since = 0.0

    def _legs_visible(self, kp: dict, side: str) -> bool:
        for j in ("knee", "ankle"):
            v = kp.get(f"{side}_{j}")
            if v is None or float(v[2]) < VIS_TH:
                return False
        return True

    def update(self, kp: dict, side: str, knee_angle: float, now: float) -> str:
        if not self._legs_visible(kp, side):
            target = "sitting"  # desk/chair occlusion is the common case
        elif knee_angle < self.sit_enter:
            target = "sitting"
        elif knee_angle > self.stand_exit:
            target = "standing"
        else:
            target = self.scenario  # deadband: hold current, no flicker
        if target == self.scenario:
            self._pending = None
            return self.scenario
        if self._pending != target:
            self._pending, self._since = target, float(now)
        if float(now) - self._since >= self.latch_s:
            self.scenario = target
            self._pending = None
        return self.scenario


class LegShakeDetector:
    """Rhythmic leg fidget (e.g. desk bouncing) vs intentional movement.

    stadig legs: low spread. Walking/shifting: large net drift. Shaking:
    spread WITHOUT drift + rapid direction changes. Buffer spans ~2s at 30fps.
    """

    def __init__(self, window: int = 60, min_std: float = 4.5,
                 min_crossings: int = 8, max_drift: float = 30.0):
        self.window = int(window)
        self.min_std = float(min_std)
        self.min_crossings = int(min_crossings)
        self.max_drift = float(max_drift)
        self._xs = deque(maxlen=self.window)

    def reset(self):
        self._xs.clear()

    def update(self, x: float | None, now: float = 0.0) -> bool:
        if x is None:
            return False
        self._xs.append(float(x))
        if len(self._xs) < self.window:
            return False
        xs = list(self._xs)
        mean = sum(xs) / len(xs)
        var = sum((v - mean) ** 2 for v in xs) / len(xs)
        if var ** 0.5 < self.min_std:
            return False
        if abs(xs[-1] - xs[0]) > self.max_drift:
            return False  # travelled somewhere: shifting, not fidgeting
        crossings = 0
        prev = xs[1] - xs[0]
        for a, b in zip(xs[1:-1], xs[2:]):
            d = b - a
            if prev != 0 and d != 0 and ((d > 0) != (prev > 0)) and abs(d) >= 1.0:
                crossings += 1
            if d != 0:
                prev = d
        return crossings >= self.min_crossings
