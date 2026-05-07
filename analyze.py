"""
Padel Shot Analyzer — Entry Point

Analyzes a padel video using pre-computed pose annotations to classify
shot types (Forehand, Backhand, Smash, Serve, Dropshot) and exports
results as an annotated video + structured CSV/JSON.

Usage:
    python analyze.py --video data/sample.mp4 --poses data/labels/poses.json --model models/shot_classifier_rf.pkl
"""

import argparse
import json
import cv2
import pandas as pd
from trackers.shot_tracker import ShotTracker


def load_pose_annotations(pose_path: str) -> tuple:
    """
    Load COCO-format pose annotations and group them by frame.

    Args:
        pose_path: Path to the COCO keypoints JSON file.

    Returns:
        Tuple of (pose_dataframe_grouped_by_frame, keypoints_column_name).
    """
    with open(pose_path) as f:
        data = json.load(f)

    img_map = {img["id"]: img["file_name"] for img in data["images"]}
    df = pd.DataFrame(data["annotations"])
    df["file_name"] = df["image_id"].map(img_map)
    df["frame_id"] = df["file_name"].apply(
        lambda x: int(x.split("_")[1].split(".")[0])
    )

    kp_col = "keypoints" if "keypoints" in df.columns else "keypoints_x"
    return df.groupby("frame_id"), kp_col


def run_analysis(video_path: str, pose_path: str, model_path: str,
                 output_video: str, output_csv: str, output_json: str):
    """
    Run the full shot analysis pipeline.

    Args:
        video_path: Path to the input video.
        pose_path: Path to the COCO pose annotations JSON.
        model_path: Path to the trained .pkl classifier.
        output_video: Path for the annotated output video.
        output_csv: Path for the CSV shot log.
        output_json: Path for the JSON shot report.
    """
    # Initialize
    tracker = ShotTracker(model_path)
    pose_groups, kp_col = load_pose_annotations(pose_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    print(f"Processing: {video_path}")
    print(f"  Resolution: {w}x{h} @ {fps:.0f} FPS | {total} frames")
    print(f"  Model: {model_path}")

    # Frame-by-frame processing
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id in pose_groups.groups:
            players = pose_groups.get_group(frame_id)
            keypoints_list = []
            bboxes = []

            for _, row in players.iterrows():
                kp = row[kp_col]
                if isinstance(kp, str):
                    kp = json.loads(kp)
                keypoints_list.append(kp)
                bboxes.append(row.get("bbox", None))

            preds = tracker.process_frame(frame_id, fps, keypoints_list)
            frame = tracker.draw_annotations(frame, bboxes, preds)

        writer.write(frame)
        frame_id += 1

        if frame_id % 500 == 0:
            print(f"  Progress: {frame_id}/{total} frames ({frame_id/total*100:.0f}%)")

    cap.release()
    writer.release()

    # Export results
    tracker.export_csv(output_csv)
    report = tracker.export_json(output_json, video_path, frame_id, fps)

    print(f"\nDone! Processed {frame_id} frames.")
    print(f"  Shot events logged: {report['total_shot_events']}")
    print(f"  Shot summary: {report['shot_summary']}")
    print(f"\nOutputs:")
    print(f"  Video -> {output_video}")
    print(f"  CSV   -> {output_csv}")
    print(f"  JSON  -> {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Padel Shot Analyzer")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--poses", required=True, help="COCO pose JSON path")
    parser.add_argument("--model", default="models/shot_classifier_rf.pkl",
                        help="Trained classifier .pkl path")
    parser.add_argument("--out-video", default="output/result_shots.mp4")
    parser.add_argument("--out-csv", default="output/shot_analysis.csv")
    parser.add_argument("--out-json", default="output/shot_analysis.json")
    args = parser.parse_args()

    run_analysis(args.video, args.poses, args.model,
                 args.out_video, args.out_csv, args.out_json)
