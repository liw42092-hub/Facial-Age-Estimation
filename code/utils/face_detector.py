from __future__ import annotations

from collections import deque

import cv2


class HaarFaceDetector:
    def __init__(self, min_face_size: int = 40, padding: float = 0.15):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load OpenCV Haar cascade: {cascade_path}")
        self.min_face_size = min_face_size
        self.padding = padding

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        height, width = frame.shape[:2]
        boxes = []
        for x, y, w, h in faces:
            pad_x = int(w * self.padding)
            pad_y = int(h * self.padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)
            boxes.append((x1, y1, x2, y2))
        return boxes


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


class SimpleFaceTracker:
    """Greedy IoU tracking used only to keep per-face age smoothing queues."""

    def __init__(self, history_size: int = 10, iou_threshold: float = 0.25, max_missed: int = 5):
        self.history_size = history_size
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes, ages):
        unmatched_track_ids = set(self.tracks)
        assignments = []
        for box, age in zip(boxes, ages):
            candidates = [
                (box_iou(box, self.tracks[track_id]["box"]), track_id)
                for track_id in unmatched_track_ids
            ]
            best_iou, best_id = max(candidates, default=(0.0, None))
            if best_id is None or best_iou < self.iou_threshold:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "box": box,
                    "ages": deque(maxlen=self.history_size),
                    "missed": 0,
                }
            else:
                unmatched_track_ids.remove(best_id)

            track = self.tracks[best_id]
            track["box"] = box
            track["ages"].append(float(age))
            track["missed"] = 0
            assignments.append(best_id)

        for track_id in unmatched_track_ids:
            self.tracks[track_id]["missed"] += 1
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missed"] <= self.max_missed
        }
        return assignments

    def active_tracks(self):
        import numpy as np

        output = []
        for track_id, track in self.tracks.items():
            if track["ages"]:
                output.append((track_id, track["box"], float(np.median(track["ages"]))))
        return output
