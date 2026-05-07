# 🎾 Padel 2D-Tracking & Shot Classification

> Automatic two-dimensional tracking of Padel games with real-time shot type classification using computer vision and machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)

---

## 📌 Overview

This project implements a complete pipeline for analyzing Padel match videos from a single camera angle:

1. **Player Detection & Tracking** — YOLOv8 detects and tracks all 4 players on the court
2. **Ball Detection & Tracking** — TrackNet locates the fast-moving ball frame by frame
3. **Court Detection** — Contour detection + homography maps real-world positions to a 2D court view
4. **Shot Classification** — A trained Random Forest classifier identifies shot types from player pose keypoints

### Shot Types Detected

| Shot | Description |
|------|-------------|
| 🟢 Forehand | Standard forehand drive |
| 🟠 Backhand | Backhand stroke |
| 🔴 Smash | Overhead smash/bandeja |
| 🟡 Serve | Service motion |
| 🟣 Dropshot | Soft drop shot |
| ⚪ Other | Unclassified contact |

---

## 🏗️ Project Structure

```
DS_Padel/
│
├── main.py                      # Original tracking pipeline (players, ball, court)
├── analyze.py                   # Shot analysis entry point (video → annotated video + CSV/JSON)
├── requirements.txt             # Python dependencies
├── court_reference.png          # 2D court template for homography
│
├── models/
│   ├── model.py                 # TrackNet neural network architecture
│   └── shot_classifier.py       # Shot classifier wrapper (loads trained .pkl model)
│
├── trackers/
│   ├── __init__.py
│   ├── player_tracker.py        # YOLOv8 player detection & tracking
│   ├── ball_tracknet.py         # TrackNet ball detection & tracking
│   ├── court_tracker.py         # Court line detection & homography
│   └── shot_tracker.py          # Shot event detection & logging
│
├── utils/
│   ├── __init__.py
│   ├── bbox_utils.py            # Bounding box utilities
│   ├── video_utils.py           # Video I/O helpers
│   ├── image_processor.py       # Frame annotation & drawing
│   └── conversions.py           # Coordinate transformations
│
├── minicourt/
│   ├── __init__.py
│   └── mini_court.py            # 2D court visualization & position mapping
│
├── notebooks/
│   └── training.ipynb           # Kaggle notebook: data loading, training, evaluation
│
├── output/                      # Generated videos, CSVs, and JSONs
└── tracker_stubs/               # Pre-computed detection caches (.pkl)
```

---

## 🧠 Shot Classification Model

### Training Data

Trained on the [**PadelTracker100**](https://doi.org/10.5281/zenodo.14653706) dataset — the first large-scale annotated padel dataset containing ~100,000 frames from two 2022 World Padel Tour Finals matches with frame-level pose keypoints and shot labels.

### Methodology

| Step | Details |
|------|---------|
| **Input Features** | 34 values per sample (17 COCO body joints × 2 coordinates) |
| **Model** | Random Forest Classifier (200 estimators) |
| **Train/Test Split** | 80/20 stratified split |
| **Training Samples** | 42,404 |
| **Test Samples** | 10,601 |

### Results

```
Overall Accuracy: 93.3%

              precision    recall  f1-score   support
    Backhand       0.93      0.93      0.93      2703
    Dropshot       0.98      0.82      0.90        73
    Forehand       0.91      0.94      0.92      3092
       Other       0.96      0.82      0.89       767
       Serve       0.98      0.99      0.98      1387
       Smash       0.93      0.94      0.94      2579
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/Layman_Intern.git
cd DS_Padel
pip install -r requirements.txt
```

### 2. Download Model Weights

Place the trained model file in the `models/` directory:
- `shot_classifier_rf.pkl` — Shot classification model (from Kaggle notebook)
- `model_best.pt` — TrackNet ball detection weights

### 3. Run Shot Analysis

```bash
python analyze.py \
    --video data/2022_BCN_FinalF_1_sample.mp4 \
    --poses data/labels/2022_BCN_FinalF_1_pose.json \
    --model models/shot_classifier_rf.pkl
```

### 4. Run Full Tracking Pipeline

```bash
python main.py
```

---

## 📊 Output Format

The analyzer produces structured output in both CSV and JSON:

### CSV (`shot_analysis.csv`)

| frame_id | timestamp_sec | timestamp | player | shot_type | confidence |
|----------|--------------|-----------|--------|-----------|------------|
| 42 | 1.400 | 00:01.40 | player_1 | Forehand | 0.923 |
| 58 | 1.933 | 00:01.93 | player_2 | Backhand | 0.871 |

### JSON (`shot_analysis.json`)

```json
{
  "video": "data/sample.mp4",
  "total_frames": 407,
  "fps": 30.0,
  "duration_sec": 13.57,
  "shot_summary": {
    "Forehand": 12,
    "Backhand": 8,
    "Smash": 3
  },
  "events": [...]
}
```

---

## 📓 Training Notebook

The full training pipeline is available in [`notebooks/training.ipynb`](notebooks/training.ipynb):

1. Downloads the PadelTracker100 dataset from Zenodo
2. Loads and merges pose annotations with shot labels
3. Extracts skeleton features from COCO keypoints
4. Trains and evaluates Random Forest, XGBoost, and LSTM models
5. Exports the best model as a `.pkl` file

You can run it directly on [Kaggle](https://www.kaggle.com/) with free GPU access.

---

## Acknowledgments

- **PadelTracker100 Dataset** — Bada-Nerín et al. ([Zenodo](https://doi.org/10.5281/zenodo.14653706))
- **DS_Padel Framework** — Novillo et al. ([GitHub](https://github.com/AlvaroNovillo/DS_Padel))
- **YOLOv8** — [Ultralytics](https://github.com/ultralytics/ultralytics)
- **TrackNet** — Ball tracking neural network architecture

---

## 📄 License

This project uses data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
