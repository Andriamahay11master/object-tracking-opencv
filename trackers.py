import cv2

def create_tracker(name):
    if name == "CSRT":
        return cv2.TrackerCSRT_create()
    elif name == "KCF":
        return cv2.TrackerKCF_create()
    elif name == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()
    else:
        raise ValueError("Unknown tracker")
