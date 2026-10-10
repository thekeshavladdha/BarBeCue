"""BarBeCue rep_counter.py — hysteresis rep counter (Alg B.2, Fig 4.3).

LLD: State pattern. States REST/MOVING/PEAK + `moving` flag + `extreme`
tracking + 0.5s minimum interval. Single threshold would double/triple
count on hover noise; deadband >=30deg (from EXERCISES) + smoothing + min
interval are three independent defences.

Direction: dec = start_thr > end_thr (squat/curl decreasing) else press
increasing. Hold-mode configs (start_thr None) return None — app uses a
good-form timer instead.
Events: 'rep' (full down+up with clean form), 'bad_rep' (full cycle but a
severity>=threshold fault was latched during the rep -> not counted),
'partial' (clear excursion >20deg from start without reaching PEAK ->
'Go lower' flash), None otherwise.
`valid=False` marks the in-progress rep dirty; the state machine still
tracks depth normally so depth-first debugging (plot_angle.py) is unaffected.
"""
from __future__ import annotations

__all__ = ["RepCounter"]

MIN_REP_INTERVAL = 0.5
PARTIAL_EXCURSION = 20.0


class RepCounter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.reps = 0
        self.stage = "REST"
        self.moving = False
        self.extreme = None
        self.last_rep = -1e9

    def update(self, a: float, now: float, valid: bool = True):
        """Feed smoothed angle `a` at time `now` (seconds). Returns event.

        `valid` is the strict-form latch from CoachState: raw faults of
        severity>=threshold seen at any point of this rep cycle. Invalid
        full cycles emit 'bad_rep' and are NOT counted.
        """
        c = self.cfg
        if c.get("start_thr") is None or c.get("end_thr") is None:
            return None  # hold mode: no counting
        dec = c["start_thr"] > c["end_thr"]
        a = float(a)
        if dec:
            at_rest = a >= c["start_thr"]
            at_peak = a <= c["end_thr"]
        else:
            at_rest = a <= c["start_thr"]
            at_peak = a >= c["end_thr"]
        event = None
        if not at_rest:
            if not self.moving:
                self.moving, self.extreme = True, a
            else:
                self.extreme = min(self.extreme, a) if dec else max(self.extreme, a)
            if at_peak:
                self.stage = "PEAK"
        else:
            if self.moving:
                if self.stage == "PEAK":
                    if now - self.last_rep >= MIN_REP_INTERVAL:
                        self.last_rep = now
                        event = "rep" if valid else "bad_rep"
                        if valid:
                            self.reps += 1
                elif abs(self.extreme - c["start_thr"]) > PARTIAL_EXCURSION:
                    event = "partial"
                self.moving, self.stage = False, "REST"
        return event
