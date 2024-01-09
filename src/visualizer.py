import matplotlib.axes as axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask
from typing import Optional


def show_int32_masks(img, mask: np.ndarray, alpha=0.5):
    ax = plt.gca()
    ax.imshow(img)
    ax.imshow(mask, cmap='tab10', alpha=alpha)


def show_bbox(bboxs, names: Optional[list] = None):
    ax = plt.gca()
    for bbox in bboxs:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    if names == None:
        return 
    for bbox, name in zip(bboxs, names):
        plt.text(bbox[0], bbox[1], name, color='r')

def show_rle_masks(masks, step=False, alpha=0.35):
    if len(masks) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    binary_masks = [mask.decode(m).astype(np.bool_) for m in masks]
    img = np.ones((binary_masks[0].shape[0],
                  binary_masks[0].shape[1], 4))
    img[:, :, 3] = 0
    for m in binary_masks:
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
        if step:
            ax.imshow(img)
            plt.draw()
            plt.waitforbuttonpress()
    ax.imshow(img)


'''
python src/visualizer.py --img_path datasets/SLiMe/bus

python src/visualizer.py --img_path datasets/SLiMe/dog

python src/visualizer.py --img_path datasets/coco/bus_square
'''
if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    args = parser.parse_args()
    _, ax = plt.subplots()
    for file in os.listdir(args.img_path):
        if not file.endswith('.npy'):
            continue
        img = plt.imread(os.path.join(args.img_path, file.replace('.npy', '.png')))
        masks = np.load(os.path.join(args.img_path, file))
        show_int32_masks(img, masks)
        plt.waitforbuttonpress()
        # os.remove(os.path.join(args.img_path, file))
