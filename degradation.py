import cv2
import numpy as np

def add_occlusion(frame, bbox, ratio=0.5):
    x, y, w, h = map(int, bbox)
    occ_w = int(w * ratio)
    occ_h = int(h * ratio)
    frame[y:y+occ_h, x:x+occ_w] = 0
    return frame

def add_gaussian_noise(frame, sigma=15):
    noise = np.random.normal(0, sigma, frame.shape).astype(np.uint8)
    return cv2.add(frame, noise)

def add_motion_blur(frame, k=9):
    kernel = np.zeros((k, k))
    kernel[k//2, :] = 1
    kernel /= k
    return cv2.filter2D(frame, -1, kernel)
