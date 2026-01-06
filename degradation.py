import cv2
import numpy as np

# ------------------------------------------------------------
# This file contains functions used to simulate visual
# degradation conditions (occlusion and noise) in order to
# evaluate the robustness of visual object trackers.
# ------------------------------------------------------------

def add_occlusion(frame, bbox, ratio=0.5):
    """
    Simulates partial occlusion by masking a portion of the
    target bounding box.

    Parameters:
    - frame: current video frame (numpy array)
    - bbox: ground-truth bounding box [x, y, width, height]
    - ratio: proportion of the bounding box to be occluded

    The occlusion is applied as a black rectangle over the
    upper-left region of the target.
    """
    # Extract bounding box coordinates
    x, y, w, h = map(int, bbox)

    # Compute occlusion size based on the given ratio
    occ_w = int(w * ratio)
    occ_h = int(h * ratio)

    # Apply occlusion by setting pixel values to zero (black)
    frame[y:y + occ_h, x:x + occ_w] = 0

    return frame


def add_gaussian_noise(frame, sigma=15):
    """
    Adds Gaussian noise to the input frame to simulate
    sensor noise or poor lighting conditions.

    Parameters:
    - frame: current video frame (numpy array)
    - sigma: standard deviation of the Gaussian noise

    Returns a noisy frame with the same resolution.
    """
    # Generate Gaussian noise with zero mean
    noise = np.random.normal(0, sigma, frame.shape).astype(np.uint8)

    # Add noise to the frame using OpenCV for proper saturation
    return cv2.add(frame, noise)

