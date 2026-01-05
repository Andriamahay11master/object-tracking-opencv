# ------------------------------------------------------------
# This file contains evaluation metrics used to assess
# tracking performance. In this project, Intersection-over-
# Union (IoU) is used as the primary accuracy metric.
# ------------------------------------------------------------

def iou(boxA, boxB):
    """
    Computes the Intersection-over-Union (IoU) between two
    bounding boxes.

    Parameters:
    - boxA: predicted bounding box [x, y, width, height]
    - boxB: ground-truth bounding box [x, y, width, height]

    Returns:
    - IoU value in the range [0, 1], where higher values
      indicate better overlap.
    """

    # Compute the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection
    inter = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of union
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter

    # Return IoU; if union is zero, return 0 to avoid division by zero
    return inter / union if union > 0 else 0
