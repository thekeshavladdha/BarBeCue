"""BarBeCue pose_estimator.py — swappable pose backend behind one detect().

LLD: Dependency Inversion. app.py depends on PoseBackend, never on
MediaPipe/YOLO/RTMPose directly. All backends return the same contract:
    {name: (x_px, y_px, visibility)}
Pixels (not normalized) so geometry.angle() is aspect-correct on 960x540.

Backends:
- MediaPipePose (default, Apache-2.0): 33 pts incl heel/foot_index + visibility.
  Model: models/pose_landmarker_full.task (lite/heavy same URL pattern).
  VIDEO mode, 1 pose, conf 0.5, ts_ms strictly increasing.
- YoloPose (AGPL-3.0 copyleft — note if publishing): 17 COCO, no heel/toe,
  knee_over_toe stays 0 by design. Largest box = single user.
- RTMPose (Apache-2.0): 17 body via rtmlib ONNX balanced mode. Stub here,
  full wiring Day 5 comparison (needs onnxruntime + rtmlib).
- DummyPose: no model/camera, for Day 1-2 unit tests on Python 3.13
  where MediaPipe wheels are unavailable (needs Python 3.11).

MediaPipe landmark subset (Table 4.1):
 0 nose, 7 l_ear, 8 r_ear, 11 l_shoulder, 12 r_shoulder,
 13 l_elbow, 14 r_elbow, 15 l_wrist, 16 r_wrist,
 23 l_hip, 24 r_hip, 25 l_knee, 26 r_knee, 27 l_ankle, 28 r_ankle,
 29 l_heel, 30 r_heel, 31 l_foot_index, 32 r_foot_index
COCO 17 (YOLO/RTMPose): 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist,
 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
"""
from __future__ import annotations
import os

MODEL_URL_FULL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_URL_LITE = MODEL_URL_FULL.replace("pose_landmarker_full", "pose_landmarker_lite")
MODEL_URL_HEAVY = MODEL_URL_FULL.replace("pose_landmarker_full", "pose_landmarker_heavy")
DEFAULT_MODEL_PATH = os.path.join("models", "pose_landmarker_full.task")

# MediaPipe index -> BarBeCue name
MP_INDEX = {
    0: "nose",
    7: "l_ear", 8: "r_ear",
    11: "l_shoulder", 12: "r_shoulder",
    13: "l_elbow", 14: "r_elbow",
    15: "l_wrist", 16: "r_wrist",
    23: "l_hip", 24: "r_hip",
    25: "l_knee", 26: "r_knee",
    27: "l_ankle", 28: "r_ankle",
    29: "l_heel", 30: "r_heel",
    31: "l_foot_index", 32: "r_foot_index",
}

# COCO index -> BarBeCue name (no heel/foot_index)
COCO_INDEX = {
    0: "nose",
    3: "l_ear", 4: "r_ear",
    5: "l_shoulder", 6: "r_shoulder",
    7: "l_elbow", 8: "r_elbow",
    9: "l_wrist", 10: "r_wrist",
    11: "l_hip", 12: "r_hip",
    13: "l_knee", 14: "r_knee",
    15: "l_ankle", 16: "r_ankle",
}

__all__ = [
    "PoseBackend", "MediaPipePose", "YoloPose", "RTMPose", "DummyPose",
    "StaticStandingPose", "create_backend",
    "MP_INDEX", "COCO_INDEX",
    "DEFAULT_MODEL_PATH", "MODEL_URL_FULL",
]


class PoseBackend:
    """Interface: detect(BGR frame, ts_ms) -> kp dict. Monotonic ts enforced."""

    def __init__(self):
        self._last_ts = -1

    def _mono_ts(self, ts_ms: int) -> int:
        ts = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts
        return ts

    def detect(self, frame_bgr, ts_ms: int = 0) -> dict:
        raise NotImplementedError

    def close(self):
        pass


class MediaPipePose(PoseBackend):
    """Default backend. Requires Python 3.11 + mediapipe + task file."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[BarBeCue] pose model not found: {model_path}\n"
                f"Download with:\n"
                f'  curl -L -o {model_path} "{MODEL_URL_FULL}"\n'
                f"(lite/heavy share the same URL pattern)"
            )
        try:
            import mediapipe as mp  # noqa: F401
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ImportError(
                "[BarBeCue] mediapipe not installed. Use Python 3.11 "
                "(wheels lag newest Python):\n"
                "  py -3.11 -m venv venv & venv\\Scripts\\activate & "
                "pip install -r requirements.txt"
            ) from e
        BaseOptions = mp_python.BaseOptions
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._vision = vision
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr, ts_ms: int = 0) -> dict:
        import cv2
        ts = self._mono_ts(ts_ms)
        h, w = frame_bgr.shape[0], frame_bgr.shape[1]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        res = self._landmarker.detect_for_video(mp_image, ts)
        if not res.pose_landmarks:
            return {}
        lm = res.pose_landmarks[0]
        # largest-box not needed: num_poses=1 already single user
        out = {}
        for idx, name in MP_INDEX.items():
            if idx >= len(lm):
                continue
            p = lm[idx]
            vis = float(getattr(p, "visibility", 1.0) or 0.0)
            # presence fallback keeps tracking stable under brief occlusion
            out[name] = (float(p.x) * w, float(p.y) * h, vis)
        return out

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


class YoloPose(PoseBackend):
    """Comparison backend (AGPL-3.0). 17 COCO pts, largest box, conf as vis."""

    def __init__(self, weights: str = "yolo11n-pose.pt"):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "[BarBeCue] ultralytics not installed: pip install ultralytics "
                "(YOLO-Pose is AGPL-3.0 copyleft — note in report if publishing)"
            ) from e
        self._model = YOLO(weights)

    def detect(self, frame_bgr, ts_ms: int = 0) -> dict:
        import numpy as np
        self._mono_ts(ts_ms)
        res = self._model.predict(frame_bgr, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return {}
        # largest box = single-user framing (BarBeCue scope: 1 person)
        boxes = res.boxes
        kpts = res.keypoints.xy  # (N,17,2) pixels
        conf = res.keypoints.conf  # (N,17)
        n = len(kpts)
        if boxes is not None and len(boxes) > 0:
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (
                boxes.xyxy[:, 3] - boxes.xyxy[:, 1]
            )
            best = int(areas.argmax().item())
        else:
            best = 0
        xy = kpts[best].cpu().numpy() if hasattr(kpts[best], "cpu") else np.asarray(kpts[best])
        cf = conf[best].cpu().numpy() if hasattr(conf[best], "cpu") else np.asarray(conf[best])
        out = {}
        for idx, name in COCO_INDEX.items():
            if idx >= len(xy):
                continue
            out[name] = (float(xy[idx][0]), float(xy[idx][1]), float(cf[idx]))
        return out


class RTMPose(PoseBackend):
    """Comparison backend (Apache-2.0) via rtmlib ONNX. Stub until Day 5 bench."""

    def __init__(self, mode: str = "balanced"):
        super().__init__()
        self.mode = mode
        try:
            import rtmlib  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "[BarBeCue] RTMPose needs rtmlib + onnxruntime: "
                "pip install rtmlib onnxruntime (see rtmlib README — API drifts fast)"
            ) from e
        raise NotImplementedError(
            "[BarBeCue] RTMPose wiring lands Day 5 backend bench "
            "(balanced mode, COCO map, conf as vis, no heel/toe)."
        )

    def detect(self, frame_bgr, ts_ms: int = 0) -> dict:
        raise NotImplementedError


class DummyPose(PoseBackend):
    """No-person dummy for unit tests without model/camera."""

    def detect(self, frame_bgr, ts_ms: int = 0) -> dict:
        self._mono_ts(ts_ms)
        return {}


class StaticStandingPose(PoseBackend):
    """Synthetic standing person (pixels for 640x480) for Day 2 integration.

    Lets geometry/smoothing/exercises run on Python 3.13 with no model.
    """

    def detect(self, frame_bgr, ts_ms: int = 0) -> dict:
        self._mono_ts(ts_ms)
        return {
            "nose": (320, 80, 0.95), "l_ear": (315, 80, 0.9), "r_ear": (325, 80, 0.9),
            "l_shoulder": (300, 150, 0.95), "r_shoulder": (340, 150, 0.95),
            "l_elbow": (295, 220, 0.9), "r_elbow": (345, 220, 0.9),
            "l_wrist": (290, 280, 0.9), "r_wrist": (350, 280, 0.9),
            "l_hip": (300, 300, 0.95), "r_hip": (340, 300, 0.95),
            "l_knee": (300, 400, 0.95), "r_knee": (340, 400, 0.95),
            "l_ankle": (300, 470, 0.9), "r_ankle": (340, 470, 0.9),
            "l_heel": (295, 475, 0.85), "r_heel": (335, 475, 0.85),
            "l_foot_index": (310, 475, 0.85), "r_foot_index": (345, 475, 0.85),
        }


def create_backend(kind: str = "mediapipe", **kwargs) -> PoseBackend:
    kind = kind.lower()
    if kind in ("mediapipe", "mp", "full", "lite", "heavy"):
        return MediaPipePose(**kwargs)
    if kind in ("yolo", "yolo-pose", "ultralytics"):
        return YoloPose(**kwargs)
    if kind in ("rtmpose", "rtm", "onnx"):
        return RTMPose(**kwargs)
    if kind == "dummy":
        return DummyPose()
    if kind in ("static", "synthetic"):
        return StaticStandingPose()
    raise ValueError(f"unknown backend kind={kind!r}")
