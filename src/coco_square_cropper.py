from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import json


class COCO_Square_Cropper:
    def __init__(self, cat_name, bbox_thres, coco_ann_dir, img_dir, output_dir):
        self.coco = COCO(coco_ann_dir)
        self.cat_name = cat_name
        self.cat_id = self.coco.getCatIds(cat_name)[0]
        self.img_dir = img_dir
        self.bbox_thres = bbox_thres
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process(self):
        img_ids = self._get_img_ids()
        annotations = []
        print(f"COCO Square Cropper: processing images of {self.cat_name}")
        for img_id in tqdm(img_ids):
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.cat_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                crop_res = self._crop(img_id, ann)
                if crop_res is None:
                    continue
                square_img, position = crop_res
                square_img.save(os.path.join(self.output_dir, f'{img_id}_{ann["id"]}.png'))
                ann.update({'position': position})
                annotations.append(ann)
        with open(os.path.join(self.output_dir, f'annotations_{self.cat_name}.json'), 'w') as f:
            json.dump(annotations, f)

    def _crop(self, img_id, ann):
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_dir, img_info['file_name']))
        # make sure the image is not grayscale
        if len(img.split()) == 1:
            return None
        bbox = ann['bbox']
        x_bbox, y_bbox, w_bbox, h_bbox = bbox
        if (h_bbox * w_bbox) / (img_info['width'] * img_info['height']) < self.bbox_thres:
            return None
        x1, y1, x2, y2 = x_bbox, y_bbox, x_bbox + w_bbox, y_bbox + h_bbox
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # the desired length of new croped image
        desired_len = round(1.2 * max(w_bbox, h_bbox))
        square_x1 = round(center_x - desired_len / 2.0)
        square_y1 = round(center_y - desired_len / 2.0)
        square_x2 = round(center_x + desired_len / 2.0)
        square_y2 = round(center_y + desired_len / 2.0)
        square_img = img.crop((max(0, square_x1), max(0, square_y1), min(
            img_info['width'], square_x2), min(img_info['height'], square_y2)))
        if square_x1 < 0 or square_y1 < 0 or square_x2 > img_info['width'] or square_y2 > img_info['height']:
            avg_color = np.array(img).mean(axis=(0, 1))
            background = Image.new(
                'RGB', (desired_len, desired_len), color=tuple(avg_color.astype(int)))
            background.paste(square_img, (max(0, -square_x1), max(0, -square_y1)))
            square_img = background
        square_img = square_img.resize((512, 512))
        return square_img, [square_x1, square_y1, square_x2, square_y2]

    def _get_img_ids(self):
        img_id_list = []
        for file in os.listdir(self.img_dir):
            if not file.endswith('.jpg'):
                continue
            img_id_list.append(int(file.split('.')[0]))
        return img_id_list

'''
python src/coco_square_cropper.py --cat_name bus --bbox_thres 0.2 --coco_ann_dir datasets/coco/instances_train2017.json --img_dir datasets/coco/bus --output_dir datasets/coco/bus_square
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat_name', type=str, required=True)
    parser.add_argument('--bbox_thres', type=float, required=True)
    parser.add_argument('--coco_ann_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    csc = COCO_Square_Cropper(args.cat_name, args.bbox_thres, args.coco_ann_dir, args.img_dir, args.output_dir)
    csc.process()
