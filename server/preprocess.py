"""VolleySense CalibMap preprocessing sidecar service.

This module exposes a lightweight Flask application that accepts volleyball videos,
optionally applies Ultralytics YOLO detections + a SORT-like tracker, and returns
per-frame hints to feed into the LLM. The implementation is intentionally
self-contained and falls back to naive heuristics when GPU/weights are absent.
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required to run preprocess.py") from exc

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:  # pragma: no cover - optional dependency
    from filterpy.kalman import KalmanFilter
except Exception:  # pragma: no cover
    KalmanFilter = None

app = Flask(__name__)

BALL_CLASS_IDS = {32, 37}  # sports ball (COCO)
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    track_id: int
    cls: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, y2

    @property
    def radius_hint(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(1.0, (x2 - x1 + y2 - y1) / 4)


class SimpleTracker:
    """A permissive SORT-like tracker using a constant velocity Kalman filter."""

    def __init__(self):
        self.next_id = 1
        self.tracks: Dict[int, "TrackState"] = {}

    def update(self, detections: List[Tuple[float, float]]) -> List[int]:
        for track in self.tracks.values():
            track.predict()
        assigned_ids: List[Optional[int]] = [None] * len(detections)
        used_tracks = set()
        for idx, det in enumerate(detections):
            best_id = None
            best_dist = 1e9
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                dist = track.distance(det)
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id
            if best_id is not None and best_dist < 160:
                self.tracks[best_id].update(det)
                used_tracks.add(best_id)
                assigned_ids[idx] = best_id
        for idx, det in enumerate(detections):
            if assigned_ids[idx] is None:
                track = TrackState(self.next_id, det)
                self.tracks[self.next_id] = track
                assigned_ids[idx] = self.next_id
                self.next_id += 1
        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks and track_id not in assigned_ids:
                track.missed += 1
            else:
                track.missed = 0
            if track.missed > 30:
                del self.tracks[track_id]
        return [track_id if track_id is not None else -1 for track_id in assigned_ids]


class TrackState:
    def __init__(self, track_id: int, detection: Tuple[float, float]):
        self.track_id = track_id
        self.state = np.array([detection[0], detection[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 500
        self.missed = 0

    def predict(self):
        dt = 1.0
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        Q = np.eye(4) * 5
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: Tuple[float, float]):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * 20
        z = np.array(measurement, dtype=float)
        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.missed = 0

    def distance(self, measurement: Tuple[float, float]) -> float:
        pred = self.state[:2]
        meas = np.array(measurement)
        return float(np.linalg.norm(pred - meas))


def load_yolo_model():
    if YOLO is None:
        return None
    try:
        return YOLO('yolov8n.pt')
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Failed to load YOLO: {exc}")
        return None


yolo_model = load_yolo_model()


@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Main entry point for the preprocessing sidecar."""

    if 'file' in request.files:
        file_storage = request.files['file']
        temp_dir = tempfile.mkdtemp(prefix='volleysense_')
        video_path = os.path.join(temp_dir, file_storage.filename or 'upload.mp4')
        file_storage.save(video_path)
    else:
        video_path = request.json.get('video_path') if request.is_json else None
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'video_path not found and no file uploaded'}), 400

    params = request.form if 'file' in request.files else (request.json or {})
    fps = float(params.get('fps', 6))
    resize_w = int(params.get('resize_w', 640))
    resize_h = int(params.get('resize_h', 360))
    H = params.get('H')
    if isinstance(H, str):
        try:
            H = json.loads(H)
        except json.JSONDecodeError:
            H = None
    H = np.array(H, dtype=float) if H is not None else None

    frames = process_video(video_path, fps=fps, resize=(resize_w, resize_h), H=H)

    return jsonify({'fps': fps, 'frames': frames})


def process_video(video_path: str, fps: float, resize: Tuple[int, int], H: Optional[np.ndarray]):
    """Decode frames, run detections, and build JSON serialisable hints."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')

    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(round(native_fps / max(fps, 1))))
    frames = []
    frame_idx = -1
    sample_idx = -1
    tracker = SimpleTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        sample_idx += 1
        resized = cv2.resize(frame, resize)
        detections = run_yolo(resized)
        player_points = [det.bottom_center for det in detections if det.cls == 'person']
        tracked_ids = tracker.update(player_points)
        frame_info = {
            'idx': sample_idx,
            't': float(sample_idx / fps),
            'players': [],
            'ball': None,
        }
        person_index = 0
        for det in detections:
            if det.cls == 'person':
                bottom = det.bottom_center
                overlay = project_point(bottom, H) if H is not None else None
                frame_info['players'].append(
                    {
                        'track_id': int(tracked_ids[person_index] if person_index < len(tracked_ids) else det.track_id),
                        'img': [float(bottom[0]), float(bottom[1])],
                        'ov': [float(overlay[0]), float(overlay[1])] if overlay is not None else None,
                        'conf': float(det.confidence),
                    }
                )
                person_index += 1
            elif det.cls == 'ball' and frame_info['ball'] is None:
                center = det.center
                overlay = project_point(center, H) if H is not None else None
                frame_info['ball'] = {
                    'img': [float(center[0]), float(center[1])],
                    'ov': [float(overlay[0]), float(overlay[1])] if overlay is not None else None,
                    'r_px': float(det.radius_hint),
                }
        frames.append(frame_info)
    cap.release()
    return frames


def run_yolo(image: np.ndarray) -> List[Detection]:
    """Execute YOLO if available, else return heuristic detections."""

    if yolo_model is None:
        h, w = image.shape[:2]
        return [
            Detection(track_id=1, cls='person', confidence=0.1, bbox=(w * 0.4, h * 0.3, w * 0.6, h * 0.8))
        ]

    results = yolo_model(image, verbose=False)[0]
    detections: List[Detection] = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        if cls_id == PERSON_CLASS_ID:
            detections.append(Detection(track_id=len(detections) + 1, cls='person', confidence=conf, bbox=(x1, y1, x2, y2)))
        elif cls_id in BALL_CLASS_IDS:
            detections.append(Detection(track_id=0, cls='ball', confidence=conf, bbox=(x1, y1, x2, y2)))
    return detections


def project_point(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    x, y = point
    vec = np.array([x, y, 1.0], dtype=float)
    proj = H @ vec
    proj /= proj[2] or 1e-6
    return float(proj[0]), float(proj[1])


if __name__ == '__main__':  # pragma: no cover
    app.run(host='0.0.0.0', port=5001, debug=True)
