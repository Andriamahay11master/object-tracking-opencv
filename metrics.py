def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / union if union > 0 else 0
