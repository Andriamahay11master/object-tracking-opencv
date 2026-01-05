import cv2

# ------------------------------------------------------------
# This file provides a unified interface for creating
# different OpenCV-based visual object trackers used in
# the experimental evaluation.
# ------------------------------------------------------------

def create_tracker(name):
    """
    Factory function that initializes and returns a tracker
    instance based on the specified tracker name.

    Parameters:
    - name: string identifier of the tracker type
            ("CSRT", "KCF", or "MEDIANFLOW")

    Returns:
    - An initialized OpenCV tracker object
    """

    # Create a CSRT tracker (robust but computationally expensive)
    if name == "CSRT":
        return cv2.TrackerCSRT_create()

    # Create a KCF tracker (good trade-off between speed and accuracy)
    elif name == "KCF":
        return cv2.TrackerKCF_create()

    # Create a MedianFlow tracker (fast but sensitive to occlusion)
    elif name == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()

    # Create a MIL tracker (robust but slower)
    elif name == "MIL":
         return cv2.TrackerMIL_create()
    
    # Handle invalid tracker names
    else:
        raise ValueError("Unknown tracker")
