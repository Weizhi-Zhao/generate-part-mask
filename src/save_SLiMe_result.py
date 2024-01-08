import numpy as np
import torch
import cv2


def save_mask(output_dir, mask: torch.tensor):
    mask = mask.squeeze(0).cpu().numpy()
    np.save(output_dir, mask)
