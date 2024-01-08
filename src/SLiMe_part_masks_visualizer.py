import matplotlib.axes as axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def mask_visualizer(ax: axes,
                    img: np.ndarray,
                    mask: np.ndarray) -> axes:
    ax.cla()
    ax.imshow(img)
    ax.imshow(mask, cmap='tab10', alpha=0.7)


def bbox_visualizer(ax, bbox):
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 -
                             y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


'''
python src/SLiMe_part_masks_visualizer.py --img_path datasets/SLiMe/bus

python src/SLiMe_part_masks_visualizer.py --img_path datasets/SLiMe/dog
'''
if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    args = parser.parse_args()
    _, ax = plt.subplots()
    for file in os.listdir(args.img_path):
        if not file.endswith('.png'):
            continue
        img = plt.imread(os.path.join(args.img_path, file))
        masks = np.load(os.path.join(args.img_path, file.split('.')[0] + '.npy'))
        mask_visualizer(ax, img, masks)
        plt.pause(0.3)
