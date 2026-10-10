"""BarBeCue feedback.py — debounced, prioritised coaching hints (Alg B.3).

LLD: rule evaluation is stateless; timing state (first_seen/last_active)
makes single noisy frames never flash and readable hints persist 1.5s
after the fault clears (Fig 4.4). At most `max_msgs` by severity; caller
merges 'partial' flash (depth failure) over rule hints and shows
'Good form' when the returned list is empty. Visibility gating happens in
app.py before update() — never call with guessed joints.
Rule shape: {metric, op, value, msg, severity 1-3, when=(m,op,v)?}.
"""
from __future__ import annotations
import operator

__all__ = ["Feedback", "OPS", "raw_violations"]

OPS = {">": operator.gt, "<": operator.lt,
       ">=": operator.ge, "<=": operator.le, "==": operator.eq}


def raw_violations(rules: list, m: dict) -> list:
    """Instant (no debounce/hold) rule evaluation. Used for rep gating.

    Debounced msgs lag ~0.6s, so a fault flashing only at the bottom of a
    rep would miss the count decision. Gating uses raw state instead.
    """
    violated = []
    for r in rules:
        gate = r.get("when")
        if gate is not None:
            gm, gop, gv = gate
            if gm not in m or not OPS[gop](m[gm], gv):
                continue
        if r["metric"] not in m:
            continue
        if OPS[r["op"]](m[r["metric"]], r["value"]):
            violated.append(r)
    return violated


class Feedback:
    def __init__(self, debounce: float = 0.6, hold: float = 1.5, max_msgs: int = 2):
        self.debounce, self.hold, self.max_msgs = float(debounce), float(hold), int(max_msgs)
        self.first_seen: dict = {}
        self.last_active: dict = {}

    def update(self, rules: list, m: dict, now: float) -> list:
        violated = raw_violations(rules, m)
        active = {r["msg"] for r in violated}
        for msg in list(self.first_seen):
            if msg not in active:
                del self.first_seen[msg]
        for r in sorted(violated, key=lambda r: -r.get("severity", 1)):
            t0 = self.first_seen.setdefault(r["msg"], now)
            if now - t0 >= self.debounce:
                self.last_active[r["msg"]] = now
        out = [msg for msg, t in self.last_active.items() if now - t <= self.hold]
        sev = {r["msg"]: r.get("severity", 1) for r in rules}
        out.sort(key=lambda s: -sev.get(s, 1))
        return out[: self.max_msgs]

    def reset(self):
        self.first_seen.clear()
        self.last_active.clear()
