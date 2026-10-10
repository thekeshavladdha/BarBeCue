"""BarBeCue smoothing.py — jitter filtering for keypoints and angles.

LLD: stateful filters with reset() discipline. Dropdown switch must call
reset() so old limb history never leaks into the new exercise.
Default: EMA alpha=0.5 on keypoints + 0.6*raw+0.4*prev on angles.
Upgrade to OneEuro when static-hold flicker persists (plank benefits most).
"""
from __future__ import annotations
import math

__all__ = ["EMA", "AngleSmoother", "smooth_angle", "OneEuroFilter"]


class EMA:
    """Exponential moving average over keypoint dict.

    smoothed = alpha*new + (1-alpha)*prev, per joint x,y.
    Visibility passes through unfiltered. Roughly halves ankle jitter
    (4.2px raw -> 2.1px @0.5 in static stand).
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.state: dict = {}

    def __call__(self, kp: dict) -> dict:
        out = {}
        for name, (x, y, v) in kp.items():
            x, y = float(x), float(y)
            if name in self.state:
                px, py, _ = self.state[name]
                x = self.alpha * x + (1.0 - self.alpha) * px
                y = self.alpha * y + (1.0 - self.alpha) * py
            out[name] = (x, y, v)
        self.state = out
        return out

    def reset(self):
        self.state = {}


def smooth_angle(prev: float | None, raw: float, alpha: float = 0.6) -> float:
    """Angle-level smoothing: alpha*raw + (1-alpha)*prev. None-safe."""
    if prev is None:
        return float(raw)
    return float(alpha * float(raw) + (1.0 - alpha) * float(prev))


class AngleSmoother:
    """Single-value EMA for the tracked rep metric (0.6/0.4 default)."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = float(alpha)
        self.prev: float | None = None

    def __call__(self, raw: float) -> float:
        self.prev = smooth_angle(self.prev, raw, self.alpha)
        return self.prev

    def reset(self):
        self.prev = None


class _LowPass:
    """Single 1-D low-pass used by OneEuroFilter."""

    def __init__(self):
        self.hat = None

    def __call__(self, x: float, cutoff: float, freq: float, dt: float = None) -> float:
        if dt is None:
            dt = 1.0 / float(freq)
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        alpha = 1.0 / (1.0 + tau / max(dt, 1e-6))
        if self.hat is None:
            self.hat = float(x)
        else:
            self.hat = alpha * float(x) + (1.0 - alpha) * self.hat
        return self.hat


class OneEuroFilter:
    """Speed-adaptive filter (Casiez et al. CHI 2012).

    Smooths strongly during slow holds, follows quickly during fast reps.
    Tune per camera FPS, not per exercise. Drop-in replacement for EMA
    when EMA(alpha=0.5) still flickers.
    """

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x: dict = {}   # name -> {'fx': LowPass, 'dx': LowPass, 'last': x}
        self._y: dict = {}

    def _filter_axis(self, store: dict, name: str, val: float) -> float:
        st = store.get(name)
        if st is None:
            st = {"fx": _LowPass(), "dx": _LowPass(), "last": float(val)}
            store[name] = st
        dt = 1.0 / self.freq
        d = (float(val) - st["last"]) / dt
        d_hat = st["dx"](d, self.dcutoff, self.freq, dt)
        cutoff = self.mincutoff + self.beta * abs(d_hat)
        out = st["fx"](val, cutoff, self.freq, dt)
        st["last"] = float(val)
        return out

    def __call__(self, kp: dict) -> dict:
        out = {}
        for name, (x, y, v) in kp.items():
            fx = self._filter_axis(self._x, name, float(x))
            fy = self._filter_axis(self._y, name, float(y))
            out[name] = (fx, fy, v)
        return out

    def reset(self):
        self._x = {}
        self._y = {}
