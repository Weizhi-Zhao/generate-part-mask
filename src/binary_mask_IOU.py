import numpy as np

def iomax(a_mask, b_mask):
    intersection = a_mask & b_mask
    a_acore = intersection.sum() / (a_mask.sum() + 1e-7)
    b_score = intersection.sum() / (b_mask.sum() + 1e-7)
    return max(a_acore, b_score)


def iou(a_mask, b_mask, thres=0):
    intersection = a_mask & b_mask
    union = a_mask | b_mask
    if intersection.sum() / (union.sum() + 1e-7) < thres:
        return 0
    else:
        return intersection.sum() / (union.sum() + 1e-7)