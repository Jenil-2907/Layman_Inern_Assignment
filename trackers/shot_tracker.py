"""
Shot Tracker Module
Processes video frames with pose data to detect and log shot events.
Uses the ShotClassifier for per-frame predictions and handles
deduplication of consecutive identical predictions.
"""

import json
import cv2
import pandas as pd
from models.shot_classifier import ShotClassifier


class ShotTracker:
    """Tracks shot events across video frames using pose keypoints."""

    def __init__(self, model_path: str):
        """
        Initialize the shot tracker.

        Args:
            model_path: Path to the trained .pkl classifier model.
        """
        self.classifier = ShotClassifier(model_path)
        self.shot_log = []
        self.prev_shots = {}

    def process_frame(self, frame_id: int, fps: float, players_keypoints: list):
        """
        Process a single frame: predict shot types for all detected players.

        Args:
            frame_id: Current frame number.
            fps: Video frames per second (for timestamp calculation).
            players_keypoints: List of keypoint lists, one per detected player.

        Returns:
            List of prediction dicts for this frame.
        """
        timestamp_sec = round(frame_id / fps, 3)
        mm = int(timestamp_sec // 60)
        ss = timestamp_sec % 60
        predictions = []

        for player_idx, kp in enumerate(players_keypoints):
            if not isinstance(kp, list) or len(kp) < 3:
                continue

            result = self.classifier.predict(kp)
            player_key = f"player_{player_idx + 1}"
            result["player"] = player_key

            # Log only when shot type changes (avoids flooding the log)
            if self.prev_shots.get(player_key) != result["label"]:
                self.shot_log.append({
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp_sec,
                    "timestamp": f"{mm:02d}:{ss:05.2f}",
                    "player": player_key,
                    "shot_type": result["label"],
                    "confidence": result["confidence"],
                })
                self.prev_shots[player_key] = result["label"]

            predictions.append(result)

        return predictions

    def draw_annotations(self, frame, bboxes: list, predictions: list):
        """
        Draw bounding boxes and shot labels on a video frame.

        Args:
            frame: OpenCV image (numpy array).
            bboxes: List of [x, y, w, h] bounding boxes per player.
            predictions: List of prediction dicts from process_frame().

        Returns:
            labelled frame.
        """
        for bbox, pred in zip(bboxes, predictions):
            if bbox is None:
                continue
            x, y, w, h = [int(v) for v in bbox]
            color = pred["color"]
            label = f"{pred['label']} ({pred['confidence']:.0%})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return frame

    def export_csv(self, output_path: str):
        """Save shot events to a CSV file."""
        df = pd.DataFrame(self.shot_log)
        df.to_csv(output_path, index=False)
        return df

    def export_json(self, output_path: str, video_path: str = "",
                    total_frames: int = 0, fps: float = 0):
        """Save shot events and metadata to a JSON file."""
        df = pd.DataFrame(self.shot_log)
        summary = df["shot_type"].value_counts().to_dict() if len(df) > 0 else {}

        output = {
            "video": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
            "total_shot_events": len(self.shot_log),
            "shot_summary": summary,
            "events": self.shot_log,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return output
