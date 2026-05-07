"""
Shot Classifier Module
Wraps the trained Random Forest / XGBoost model for shot type prediction.
Supports loading from .pkl files exported from the training notebook.
"""

import pickle
import numpy as np


class ShotClassifier:
    """Classifies padel shot types from player pose keypoints."""

    SHOT_COLORS = {
        "Forehand": (0, 255, 0),
        "Backhand": (255, 165, 0),
        "Smash": (0, 0, 255),
        "Serve": (255, 255, 0),
        "Dropshot": (255, 0, 255),
        "Other": (200, 200, 200),
    }

    def __init__(self, model_path: str):
        """
        Load a trained shot classifier from a .pkl file.

        Args:
            model_path: Path to the .pkl file containing 'model' and 'label_encoder'.
        """
        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        self.model = saved["model"]
        self.label_encoder = saved["label_encoder"]
        self.classes = self.label_encoder.classes_.tolist()

    def extract_features(self, keypoints: list) -> np.ndarray:
        """
        Convert COCO keypoints [x1, y1, v1, x2, y2, v2, ...] into
        a flat feature vector of [x1, y1, x2, y2, ...] (dropping visibility).

        Args:
            keypoints: Raw COCO-format keypoints list.

        Returns:
            1D numpy array of (x, y) coordinate pairs.
        """
        coords = []
        for i in range(0, len(keypoints), 3):
            coords.append(keypoints[i])      # x
            coords.append(keypoints[i + 1])  # y
        return np.array(coords, dtype=np.float32)

    def predict(self, keypoints: list) -> dict:
        """
        Predict shot type from a single set of keypoints.

        Args:
            keypoints: COCO-format keypoints [x1, y1, v1, ...].

        Returns:
            Dict with 'label', 'confidence', and 'color' keys.
        """
        features = self.extract_features(keypoints).reshape(1, -1)
        pred_idx = self.model.predict(features)[0]
        label = self.label_encoder.inverse_transform([pred_idx])[0]

        probs = self.model.predict_proba(features)[0]
        confidence = float(probs.max())

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "color": self.SHOT_COLORS.get(label, (255, 255, 255)),
        }

    def predict_batch(self, keypoints_list: list) -> list:
        """
        Predict shot types for multiple keypoint sets at once.

        Args:
            keypoints_list: List of COCO-format keypoints lists.

        Returns:
            List of prediction dicts.
        """
        return [self.predict(kp) for kp in keypoints_list]
