import cv2
import os
import time
import pandas as pd

from trackers import create_tracker
from degradation import add_occlusion, add_gaussian_noise
from metrics import iou

# ------------------------------------------------------------
# Main experiment script.
# This script evaluates multiple visual object trackers under
# baseline, occlusion, and noise conditions across several
# OTB benchmark sequences.
# ------------------------------------------------------------

# List of trackers to be evaluated
TRACKERS = ["CSRT", "KCF", "MEDIANFLOW", "MIL"]

# Paths to selected OTB sequences and corresponding ground-truth files
ARRAY_SEQ_PATH = [
    "data/OTB/Basketball/img",
    "data/OTB/BlurBody/img",
    "data/OTB/CarDark/img",
    "data/OTB/FaceOcc2/img",
    "data/OTB/Girl2/img",
    "data/OTB/Jogging/img",
    "data/OTB/Subway/img",
    "data/OTB/Walking2/img",
    "data/OTB/Woman/img"
]

ARRAY_GT_PATH = [
    "data/OTB/Basketball/groundtruth_rect.txt",
    "data/OTB/BlurBody/groundtruth_rect.txt",
    "data/OTB/CarDark/groundtruth_rect.txt",
    "data/OTB/FaceOcc2/groundtruth_rect.txt",
    "data/OTB/Girl2/groundtruth_rect.txt",
    "data/OTB/Jogging/groundtruth_rect.txt",
    "data/OTB/Subway/groundtruth_rect.txt",
    "data/OTB/Walking2/groundtruth_rect.txt",
    "data/OTB/Woman/groundtruth_rect.txt"
]

# Containers for storing evaluation results across all sequences
results = []             # Baseline IoU results
resultsOcclusion = []    # Occlusion IoU results
resultsNoise = []        # Noise IoU results
fps_results = []         # FPS measurements

# ------------------------------------------------------------
# Loop over all selected sequences
# ------------------------------------------------------------
for seq_index in range(len(ARRAY_SEQ_PATH)):

    # Extract sequence name for logging and result storage
    NAME_SEQ = ARRAY_SEQ_PATH[seq_index].split("/")[-2]
    print("Processing sequence:", NAME_SEQ)

    SEQ_PATH = ARRAY_SEQ_PATH[seq_index]
    GT_PATH = ARRAY_GT_PATH[seq_index]

    # Load all frame paths and ground-truth annotations
    frames = sorted([os.path.join(SEQ_PATH, f) for f in os.listdir(SEQ_PATH)])
    gt = pd.read_csv(GT_PATH, header=None).values

    # ===================== BASELINE EXPERIMENT =====================
    print("Baseline Experiment")

    for tracker_name in TRACKERS:

        # Initialize tracker instance
        tracker = create_tracker(tracker_name)

        # Initialize tracker using ground-truth bounding box
        first_frame = cv2.imread(frames[0])
        init_bbox = tuple(gt[0])
        tracker.init(first_frame, init_bbox)

        # Start FPS timing
        fps_start = time.time()

        # Process remaining frames
        for i in range(1, len(frames)):
            frame = cv2.imread(frames[i])

            # Update tracker
            success, pred_bbox = tracker.update(frame)

            # Penalize tracking failure by assigning IoU = 0
            if success == False:
                score = 0
            else:
                score = iou(pred_bbox, gt[i])

            # Store baseline IoU result
            results.append([NAME_SEQ, tracker_name, i, score])

        # Compute and store FPS
        fps = len(frames) / (time.time() - fps_start)
        print(tracker_name, "FPS:", round(fps, 2))
        fps_results.append([NAME_SEQ, tracker_name, "Baseline", fps])

    # ===================== OCCLUSION EXPERIMENT =====================
    print("Occlusion Experiment")

    for tracker_name in TRACKERS:

        tracker = create_tracker(tracker_name)
        first_frame = cv2.imread(frames[0])
        init_bbox = tuple(gt[0])
        tracker.init(first_frame, init_bbox)

        fps_start = time.time()

        for i in range(1, len(frames)):
            frame = cv2.imread(frames[i])

            # Apply partial occlusion during a predefined interval
            if 30 <= i <= 60:
                frame = add_occlusion(frame, gt[i], ratio=0.4)

            success, pred_bbox = tracker.update(frame)

            # Penalize tracking failure
            if success == False:
                score = 0
            else:
                score = iou(pred_bbox, gt[i])

            # Store occlusion IoU result
            resultsOcclusion.append([NAME_SEQ, tracker_name, i, score])

        fps = len(frames) / (time.time() - fps_start)
        print(tracker_name, "FPS:", round(fps, 2))
        fps_results.append([NAME_SEQ, tracker_name, "Occlusion", fps])

    # ===================== NOISE EXPERIMENT =====================
    print("Noise Experiment")

    for tracker_name in TRACKERS:

        tracker = create_tracker(tracker_name)
        first_frame = cv2.imread(frames[0])
        init_bbox = tuple(gt[0])
        tracker.init(first_frame, init_bbox)

        fps_start = time.time()

        for i in range(1, len(frames)):
            frame = cv2.imread(frames[i])

            # Apply Gaussian noise during a predefined interval
            if 30 <= i <= 60:
                frame = add_gaussian_noise(frame, sigma=25)

            success, pred_bbox = tracker.update(frame)

            # Penalize tracking failure
            if success == False:
                score = 0
            else:
                score = iou(pred_bbox, gt[i])

            # Store noise IoU result
            resultsNoise.append([NAME_SEQ, tracker_name, i, score])

        fps = len(frames) / (time.time() - fps_start)
        print(tracker_name, "FPS:", round(fps, 2))
        fps_results.append([NAME_SEQ, tracker_name, "Noise", fps])

    print("\n")

# ------------------------------------------------------------
# Save all results to CSV files for analysis and reporting
# ------------------------------------------------------------
df_baseline = pd.DataFrame(results, columns=["Sequence", "Tracker", "Frame", "IoU"])
df_occlusion = pd.DataFrame(resultsOcclusion, columns=["Sequence", "Tracker", "Frame", "IoU"])
df_noise = pd.DataFrame(resultsNoise, columns=["Sequence", "Tracker", "Frame", "IoU"])

df_baseline.to_csv("results/baseline_results.csv", index=False)
df_occlusion.to_csv("results/occlusion_results.csv", index=False)
df_noise.to_csv("results/noise_results.csv", index=False)

# Save FPS measurements separately
df_fps = pd.DataFrame(fps_results, columns=["Sequence", "Tracker", "Experiment", "FPS"])
df_fps.to_csv("results/fps_results.csv", index=False)
