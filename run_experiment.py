import cv2
import os
import time
import pandas as pd

from trackers import create_tracker
from degradation import add_occlusion, add_gaussian_noise
from metrics import iou

TRACKERS = ["CSRT", "KCF", "MEDIANFLOW"]

#Array storing all sequences path and ground truth path
ARRAY_SEQ_PATH = ["data/OTB/Basketball/img","data/OTB/BlurBody/img","data/OTB/CarDark/img","data/OTB/FaceOcc2/img",
                  "data/OTB/Girl2/img","data/OTB/Jogging/img","data/OTB/Subway/img","data/OTB/Walking2/img","data/OTB/Woman/img"]
ARRAY_GT_PATH = ["data/OTB/Basketball/groundtruth_rect.txt","data/OTB/BlurBody/groundtruth_rect.txt",
                 "data/OTB/CarDark/groundtruth_rect.txt","data/OTB/FaceOcc2/groundtruth_rect.txt",
                 "data/OTB/Girl2/groundtruth_rect.txt","data/OTB/Jogging/groundtruth_rect.txt",
                 "data/OTB/Subway/groundtruth_rect.txt","data/OTB/Walking2/groundtruth_rect.txt","data/OTB/Woman/groundtruth_rect.txt"]
results = []
resultsOcclusion = []
resultsNoise = []
fps_results = []

for seq_index in range(len(ARRAY_SEQ_PATH)):
    NAME_SEQ = ARRAY_SEQ_PATH[seq_index].split("/")[-2]
    print("Processing sequence:", NAME_SEQ)
    SEQ_PATH = ARRAY_SEQ_PATH[seq_index]
    GT_PATH = ARRAY_GT_PATH[seq_index]

    frames = sorted([os.path.join(SEQ_PATH, f) for f in os.listdir(SEQ_PATH)])
    gt = pd.read_csv(GT_PATH, header=None).values

    # ===================== BASELINE =====================
    print("Baseline Experiment")
    for tracker_name in TRACKERS:
        tracker = create_tracker(tracker_name)

        first_frame = cv2.imread(frames[0])
        init_bbox = tuple(gt[0])
        tracker.init(first_frame, init_bbox)

        fps_start = time.time()

        for i in range(1, len(frames)):
            frame = cv2.imread(frames[i])

            success, pred_bbox = tracker.update(frame)
            if success == False:
                score = 0
            else:
                score = iou(pred_bbox, gt[i])

            results.append([NAME_SEQ, tracker_name, i, score])

        fps = len(frames) / (time.time() - fps_start)
        print(tracker_name, "FPS:", round(fps, 2))
        fps_results.append([NAME_SEQ, tracker_name, "Baseline", fps])

    # ===================== OCCLUSION =====================
    print("Occlusion Experiment")
    for tracker_name in TRACKERS:
        tracker = create_tracker(tracker_name)

        first_frame = cv2.imread(frames[0])
        init_bbox = tuple(gt[0])
        tracker.init(first_frame, init_bbox)

        fps_start = time.time()

        for i in range(1, len(frames)):
            frame = cv2.imread(frames[i])

            # Apply occlusion between frames 30–60
            if 30 <= i <= 60:
                frame = add_occlusion(frame, gt[i], ratio=0.4)

            success, pred_bbox = tracker.update(frame)
            if success == False:
                score = 0
            else:
                score = iou(pred_bbox, gt[i])

            resultsOcclusion.append([NAME_SEQ, tracker_name, i, score])

        fps = len(frames) / (time.time() - fps_start)
        print(tracker_name, "FPS:", round(fps, 2))
        fps_results.append([NAME_SEQ, tracker_name, "Occlusion", fps])

    # ===================== NOISE =====================
    print("Noise Experiment")
    for tracker_name in TRACKERS:
        tracker = create_tracker(tracker_name)

        first_frame = cv2.imread(frames[0])
        init_bbox = tuple(gt[0])
        tracker.init(first_frame, init_bbox)

        fps_start = time.time()

        for i in range(1, len(frames)):
            frame = cv2.imread(frames[i])

            # Apply Gaussian noise between frames 30–60
            if 30 <= i <= 60:
                frame = add_gaussian_noise(frame, sigma=25)

            success, pred_bbox = tracker.update(frame)
            if success == False:
                score = 0
            else:
                score = iou(pred_bbox, gt[i])

            resultsNoise.append([NAME_SEQ, tracker_name, i, score])

        fps = len(frames) / (time.time() - fps_start)
        print(tracker_name, "FPS:", round(fps, 2))
        fps_results.append([NAME_SEQ, tracker_name, "Noise", fps])

    print("\n")

# ===================== SAVE RESULTS =====================
df_baseline = pd.DataFrame(results, columns=["Sequence", "Tracker", "Frame", "IoU"])
df_occlusion = pd.DataFrame(resultsOcclusion, columns=["Sequence", "Tracker", "Frame", "IoU"])
df_noise = pd.DataFrame(resultsNoise, columns=["Sequence", "Tracker", "Frame", "IoU"])

df_baseline.to_csv("results/baseline_results.csv", index=False)
df_occlusion.to_csv("results/occlusion_results.csv", index=False)
df_noise.to_csv("results/noise_results.csv", index=False)

df_fps = pd.DataFrame(fps_results, columns=["Sequence", "Tracker", "Experiment", "FPS"])
df_fps.to_csv("results/fps_results.csv", index=False)
