import cv2
import pandas as pd
import torch
from ultralytics import YOLO
from tqdm import tqdm
import argparse

def load_annotations(csv_path):
    """Loads ground truth annotations from CSV."""
    df = pd.read_csv(csv_path)
    return df

def run_yolo_tracking(video_path):
    """Runs YOLO object detection and tracking on the video."""
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        yolo_results = model.track(frame, show=False, save=False, stream=True)
        # yolo_results = model.track(frame, persist=True)
        
        print(f"Frame {frame_id}: YOLO Results:")
        if yolo_results[0].boxes is not None:
            for box in yolo_results[0].boxes:
                x_center, y_center, width, height = box.xywh[0].tolist()
                track_id = int(box.id[0].item()) if box.id is not None else -1
                print(f"\tTrack ID: {track_id}, x: {x_center}, y: {y_center}, w: {width}, h: {height}")
                results.append([frame_id, track_id, x_center, y_center, width, height])
        
        frame_id += 1
    
    cap.release()
    return results

def compare_results(yolo_results, ground_truth):
    """Compares YOLO detections with ground truth annotations."""
    # Convert to DataFrame for easier processing
    yolo_df = pd.DataFrame(yolo_results, columns=["frame_id", "track_id", "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"])
    
    # Merge on frame_id for comparison
    merged = pd.merge(yolo_df, ground_truth, on="frame_id", suffixes=("_yolo", "_gt"))
    
    # Compute IoU (as an example metric)
    def compute_iou(row):
        x1 = max(row['bbox_x_center_yolo'] - row['bbox_width_yolo'] / 2, row['bbox_x_center_gt'] - row['bbox_width_gt'] / 2)
        y1 = max(row['bbox_y_center_yolo'] - row['bbox_height_yolo'] / 2, row['bbox_y_center_gt'] - row['bbox_height_gt'] / 2)
        x2 = min(row['bbox_x_center_yolo'] + row['bbox_width_yolo'] / 2, row['bbox_x_center_gt'] + row['bbox_width_gt'] / 2)
        y2 = min(row['bbox_y_center_yolo'] + row['bbox_height_yolo'] / 2, row['bbox_y_center_gt'] + row['bbox_height_gt'] / 2)
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        gt_area = row['bbox_width_gt'] * row['bbox_height_gt']
        yolo_area = row['bbox_width_yolo'] * row['bbox_height_yolo']
        union_area = gt_area + yolo_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    merged['IoU'] = merged.apply(compute_iou, axis=1)
    
    # Compute mean IoU
    mean_iou = merged['IoU'].mean()
    print(f"Mean IoU: {mean_iou:.4f}")
    
    return merged

import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=r'data/input.mkv')
    parser.add_argument("--csv_path", type=str, default=r'data/annotations.csv')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = YOLO("yolo11n.pt").to(device)
    results = model.track(args.video_path, show=False, save=False, stream=True)

    # Prepare a list to store results
    tracking_data = []

    # Iterate over frames
    for frame_idx, result in enumerate(results):
        if result.boxes is not None:
            for box in result.boxes:
                track_id = box.id if box.id is not None else -1  # Some versions don't assign IDs
                x_center, y_center, width, height = box.xywh[0].tolist()  # Get bounding box info
                
                # Append results
                tracking_data.append([int(track_id.item()), frame_idx, x_center, y_center, width, height])

    # Convert to DataFrame
    df_tracking = pd.DataFrame(tracking_data, columns=["track_id", "frame_id", "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"])

    # Save to CSV
    df_tracking.to_csv("tracking_results.csv", index=False)
    print("Tracking results saved to tracking_results.csv")
    
