"""
robot_vision.py  --  TEMPLATE
The report describes this module (NCNN-optimised YOLOv8 + homography) but its
source was not included in the listing. This is a minimal stand-in with the
interface main.py expects: classify_fruit(frame) -> dict.
Replace it with your team's real robot_vision.py.
"""
import cv2
import numpy as np
from ultralytics import YOLO

CONF_THRESHOLD = 0.521  # operating point from the F1-confidence curve

# 4+ pixel <-> world calibration pairs (pixels -> cm from robot base).
# Re-measure whenever the camera or lighting setup moves.
PIXEL_PTS = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], dtype=np.float32)
WORLD_PTS = np.array([[-20, 40], [20, 40], [20, 20], [-20, 20]], dtype=np.float32)


class RobotVision:
    def __init__(self, model_path="best_ncnn_model"):
        self.model = YOLO(model_path, task="detect")
        self.H, _ = cv2.findHomography(PIXEL_PTS, WORLD_PTS)

    def pixel_to_world(self, cx, cy):
        pt = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), self.H)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def classify_fruit(self, frame):
        res = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        if len(res.boxes) == 0:
            return {}
        best = max(res.boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        wx, wy = self.pixel_to_world(cx, cy)
        return {
            "label": res.names[int(best.cls)],   # e.g. "rotten banana"
            "conf": float(best.conf),
            "pixel": (cx, cy),
            "world": (wx, wy),
            # Bounding-box aspect angle, used to align the wrist with the fruit
            "angle": float(np.degrees(np.arctan2(y2 - y1, x2 - x1))),
        }
