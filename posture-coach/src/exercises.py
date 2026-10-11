"""BarBeCue exercises.py — dropdown source. Adding an exercise = one dict entry.

LLD: config-as-code (Factory). app.py never hardcodes thresholds; Day 3-4
logic reads metric/start_thr/end_thr/rules/view/joints from here.
- reps mode: dual thresholds, deadband >=30deg, direction from start vs end
  (dec=start>end: squat/curl decreasing, press increasing).
- hold mode (plank/desk): no thresholds, accumulates good-form seconds;
  timer pauses (not resets) on fault.
- auto mode (Auto posture): no counting; per-frame sitting/standing scenario
  from posture.py classifiers, scenario rule set applied, good-posture seconds
  accumulated like hold. Leg shake surfaced while sitting.
Severity: 3 injury-linked (back/hips sag), 2 depth/alignment, 1 style.
Gate `when=(metric,op,val)`: e.g. back_lean only when knee<140 so standing
rest never warns. Gate `phase="descending"`: rule fires only during the
down-phase of a rep (descent-only gating, fewer false nags on ascent).
Thresholds are heuristics — tune via plot_angle.py (Ch.5).
Profiles: exercises may add `profiles={name: {end_thr?, start_thr?,
rule_values?}}`; "Beginner" (empty = base behaviour) is default,
"Pro" tightens depth/alignment. Applied by CoachState.set_profile().
"""
from __future__ import annotations

__all__ = ["EXERCISES", "list_exercises", "get_exercise", "list_categories", "exercises_in"]

CATEGORY_ORDER = ["Legs", "Chest", "Biceps", "Back", "Shoulders", "Abs", "Posture"]

EXERCISES = {
    "Auto posture": dict(
        mode="auto", metric="auto", start_thr=None, end_thr=None,
        partial_msg="", view="auto", category="Posture",
        joints=["shoulder", "hip"],  # upper body only: desk occlusion tolerant
        rules=[],  # unused: scenarios below own their rules
        scenarios={
            "sitting": dict(
                joints=["shoulder", "hip"],
                rules=[
                    dict(metric="back_lean", op=">", value=20,
                         msg="Sit straight", severity=2),
                    dict(metric="neck_tilt", op=">", value=15,
                         msg="Lift your chin", severity=2),
                    dict(metric="head_hold", op="<", value=0.28,
                         msg="Uncurl your upper back", severity=2),
                ],
            ),
            "standing": dict(
                joints=["shoulder", "hip"],
                rules=[
                    dict(metric="back_lean", op=">", value=15,
                         msg="Stand tall", severity=2),
                    dict(metric="neck_tilt", op=">", value=25,
                         msg="Lift your chin", severity=2),
                    dict(metric="head_hold", op="<", value=0.36,
                         msg="Uncurl your upper back", severity=2),
                ],
            ),
        },
    ),
    "Squat": dict(
        mode="reps", metric="knee", start_thr=160, end_thr=100,
        partial_msg="Go lower", view="side", category="Legs",
        joints=["shoulder", "hip", "knee", "ankle", "foot_index"],
        rules=[
            dict(metric="back_lean", op=">", value=55,
                 msg="Keep back straight", severity=3, when=("knee", "<", 140),
                 phase="descending"),
            dict(metric="knee_over_toe", op=">", value=0.30,
                 msg="Knees too far forward", severity=2),
            dict(metric="neck_tilt", op=">", value=40,
                 msg="Keep chest up", severity=1),
        ],
        # Difficulty profiles (Beginner = base behaviour, Pro = deeper +
        # stricter). CoachState.set_profile() applies rule_values by msg and
        # end_thr/start_thr overrides. HUD + --profile switch mid-session.
        profiles={
            "Beginner": dict(),
            "Pro": dict(end_thr=90, rule_values={"Knees too far forward": 0.25}),
        },
    ),
    "Lunge": dict(
        mode="reps", metric="knee", start_thr=160, end_thr=100,
        partial_msg="Go lower", view="side", category="Legs",
        joints=["shoulder", "hip", "knee", "ankle", "foot_index"],
        rules=[
            dict(metric="back_lean", op=">", value=55,
                 msg="Keep back straight", severity=3, when=("knee", "<", 140),
                 phase="descending"),
            dict(metric="knee_over_toe", op=">", value=0.30,
                 msg="Knees too far forward", severity=2),
        ],
        profiles={
            "Beginner": dict(),
            "Pro": dict(end_thr=90, rule_values={"Knees too far forward": 0.25}),
        },
    ),
    "Glute bridge": dict(
        mode="reps", metric="hip", start_thr=100, end_thr=160,
        partial_msg="Full extension", view="side", category="Legs",
        joints=["shoulder", "hip", "knee", "ankle"],
        # Lying down: shoulder-over-hip tilt is ~90 by construction, so no
        # lean rules — depth alone counts. Thresholds heuristic, tune first.
        rules=[],
    ),
    "Push-up": dict(
        mode="reps", metric="elbow", start_thr=155, end_thr=90,
        partial_msg="Go lower", view="side", category="Chest",
        joints=["shoulder", "elbow", "wrist", "hip", "ankle"],
        rules=[
            dict(metric="body_line", op="<", value=160,
                 msg="Keep hips in line", severity=3),
        ],
    ),
    "Knee push-up": dict(
        mode="reps", metric="elbow", start_thr=155, end_thr=110,
        partial_msg="Go lower", view="side", category="Chest",
        joints=["shoulder", "elbow", "wrist", "hip", "ankle"],
        rules=[
            dict(metric="body_line", op="<", value=160,
                 msg="Keep hips in line", severity=3),
        ],
    ),
    "Bicep curl": dict(
        mode="reps", metric="elbow", start_thr=150, end_thr=50,
        partial_msg="Full range", view="front/side", category="Biceps",
        joints=["shoulder", "elbow", "wrist", "hip"],
        rules=[
            dict(metric="back_lean", op=">", value=15,
                 msg="Do not swing back", severity=2),
            dict(metric="elbow_drift", op=">", value=0.25,
                 msg="Keep elbows tucked", severity=2),
        ],
    ),
    "Bent-over row": dict(
        mode="reps", metric="elbow", start_thr=150, end_thr=60,
        partial_msg="Full pull", view="side", category="Back",
        joints=["shoulder", "elbow", "wrist", "hip"],
        rules=[
            dict(metric="back_lean", op="<", value=30,
                 msg="Bend over at the hips", severity=2,
                 when=("elbow", "<", 120)),
        ],
    ),
    "Superman hold": dict(
        mode="hold", metric="hold", start_thr=None, end_thr=None,
        partial_msg="", view="side", category="Back",
        joints=["shoulder", "hip", "ankle"],
        rules=[
            dict(metric="body_line", op=">", value=170,
                 msg="Lift your chest and legs", severity=3),
        ],
    ),
    "Shoulder press": dict(
        mode="reps", metric="elbow", start_thr=100, end_thr=160,
        partial_msg="Full press", view="front/side", category="Shoulders",
        joints=["shoulder", "elbow", "wrist", "hip"],
        rules=[
            dict(metric="back_lean", op=">", value=15,
                 msg="Do not arch back", severity=2),
        ],
    ),
    "Plank hold": dict(
        mode="hold", metric="hold", start_thr=None, end_thr=None,
        partial_msg="", view="side", category="Abs",
        joints=["shoulder", "hip", "ankle"],
        rules=[
            dict(metric="hip_sag", op=">", value=0.06,
                 msg="Raise or lower hips", severity=3),
            dict(metric="hip_sag", op="<", value=-0.06,
                 msg="Raise or lower hips", severity=3),
            dict(metric="body_line", op="<", value=160,
                 msg="Keep hips in line", severity=3),
        ],
    ),
    "Hollow hold": dict(
        mode="hold", metric="hold", start_thr=None, end_thr=None,
        partial_msg="", view="side", category="Abs",
        joints=["shoulder", "hip", "ankle"],
        # Banana shape reads ~140-165 on body_line. Too flat or too tucked
        # both break the hold. Thresholds heuristic, tune first.
        rules=[
            dict(metric="body_line", op=">", value=170,
                 msg="Curl tighter", severity=2),
            dict(metric="body_line", op="<", value=125,
                 msg="Open up a little", severity=2),
        ],
    ),
    "Desk posture": dict(
        mode="hold", metric="hold", start_thr=None, end_thr=None,
        partial_msg="", view="side", category="Posture",
        joints=["shoulder", "hip"],
        rules=[
            dict(metric="back_lean", op=">", value=20,
                 msg="Sit straight", severity=2),
            dict(metric="neck_tilt", op=">", value=25,
                 msg="Lift your chin", severity=2),
            dict(metric="head_hold", op="<", value=0.28,
                 msg="Uncurl your upper back", severity=2),
        ],
    ),
}


def list_exercises() -> list:
    return list(EXERCISES.keys())


def list_categories() -> list:
    seen = [EXERCISES[n].get("category", "Posture") for n in EXERCISES]
    ordered = [c for c in CATEGORY_ORDER if c in seen]
    return ordered + [c for c in seen if c not in ordered]


def exercises_in(category: str) -> list:
    return [n for n in EXERCISES if EXERCISES[n].get("category") == category]


def get_exercise(name: str) -> dict:
    try:
        return EXERCISES[name]
    except KeyError:
        raise KeyError(
            f"unknown exercise {name!r}. Choose from {list_exercises()}"
        ) from None
