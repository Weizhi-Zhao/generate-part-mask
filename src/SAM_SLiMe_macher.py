import json
import yaml
import os
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils
import cv2
import pickle
from binary_mask_IOU import iomax, iou
from visualizer import show_rle_masks, show_bbox, show_int32_masks
import matplotlib.pyplot as plt
from typing import Optional


class SAMSLiMeMacher():
    def __init__(self, coco_ann_dir, dataset_dir, config_dir, visulize_step: Optional[int]=None, save_vis_result: Optional[str]=None, SAM_GT_THRES=0.3):
        with open(coco_ann_dir, 'r') as f:
            self.coco_anns = json.load(f)
        self.dataset_dir = dataset_dir
        with open(config_dir, 'r') as f:
            self.config = yaml.safe_load(f)
        self.visulize_step = visulize_step
        self.save_vis_result = save_vis_result
        if self.save_vis_result is not None:
            if not os.path.exists(self.save_vis_result):
                os.makedirs(self.save_vis_result)
        self.SAM_GT_THRES = SAM_GT_THRES

    def process(self):
        annotations = []
        if self.visulize_step is not None:
            fig = plt.figure(figsize=(18, 9))
        print(f"SAM SLiMe Macher: processing images of {self.config['category_name']}")
        for step, ann in enumerate(tqdm(self.coco_anns)):
            if self.visulize_step is not None and step % self.visulize_step != 0:
                continue
            file_name = str(ann['image_id']) + '_' + str(ann['id'])
            if not os.path.exists(os.path.join(self.dataset_dir, file_name + '.png')):
                print(f"SAM SLiMe Macher error: file {file_name + '.png'} not found")
                continue
            SLiMe_masks, mask_ids = self.read_SLiMe_masks(file_name)
            gt_mask = self.get_cropped_gt_binary_mask(ann)
            restricted_SLiMe_masks = self.restrict_SLiMe_masks(SLiMe_masks, gt_mask)
            # ori_SAM_masks is in RLE format
            SAM_masks, ori_SAM_masks = self.read_SAM_masks(file_name)
            part_anns = []
            for SLiMe_mask, mask_id in zip(restricted_SLiMe_masks, mask_ids):
                best_mask_ids = self.find_best_masks(SAM_masks, SLiMe_mask, gt_mask)
                part_name = self.part_id_to_name(mask_id)
                for best_mask_id in best_mask_ids:
                    part_ann = {
                        'part_mask_id': best_mask_id,
                        'part_id': mask_id,
                        'part_name': part_name,
                        'bbox': ori_SAM_masks[best_mask_id]['bbox'],
                        'segmentation': ori_SAM_masks[best_mask_id]['segmentation'],
                    }
                    part_anns.append(part_ann)
            annotations.append({
                'image_id': ann['image_id'],
                'id': ann['id'],
                'position': ann['position'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'part_annotations': part_anns
            })

            if self.visulize_step is not None:
                img = cv2.imread(os.path.join(self.dataset_dir, str(ann['image_id']) + '_' + str(ann['id']) + '.png'))
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.title('SAM')
                plt.imshow(img)
                part_rle_masks = [part_ann['segmentation'] for part_ann in part_anns]
                show_rle_masks(part_rle_masks)
                bboxs = [part_ann['bbox'] for part_ann in part_anns]
                names = [part_ann['part_name'] for part_ann in part_anns]
                show_bbox(bboxs, names)
                plt.subplot(1, 2, 2)
                plt.title('SLiMe')
                np_masks = np.zeros(restricted_SLiMe_masks[0].shape, dtype=np.int32)
                for r_mask, mask_id in zip(restricted_SLiMe_masks, mask_ids):
                    np_masks[r_mask] = mask_id
                show_int32_masks(img, np_masks)
                plt.draw()
                if self.save_vis_result is not None:
                    plt.savefig(os.path.join(self.save_vis_result, f"{file_name}.png"))
                plt.waitforbuttonpress(1)

        if self.visulize_step is None:
            with open(os.path.join(self.dataset_dir, f"part_annotations_{self.config['category_name']}.json"), 'w') as f:
                json.dump(annotations, f)

    def find_best_masks(self, SAM_MASKS, SLIME_MASK, gt_mask):
        ALPHA = 0.5
        BETA = 0.5
        IOU_THRES = 0

        # dp[i] represents the best score of selecting several masks from [0~i-1] (must contain the i-1 th mask)
        dp = []
        dp.append({
            'mask_ids': [],
            'score': 0,
            'neg_score': 0,
            'union_mask': np.zeros(SLIME_MASK.shape, dtype=np.bool_)
        })

        for mask_id in range(len(SAM_MASKS)):
            # ignore unqualified masks
            if iomax(SAM_MASKS[mask_id], gt_mask) < self.SAM_GT_THRES:
                continue
            new_dp = {
                'mask_ids': [],
                'score': -1e9,
                'neg_score': 0,
                'union_mask': np.zeros(SLIME_MASK[0].shape, dtype=np.bool_)
            }
            for dp_from in dp:
                neg_score = 0
                pos_score = 0

                used_mask_ids = dp_from['mask_ids']
                neg_score -= iomax(SAM_MASKS[mask_id], dp_from['union_mask']) * BETA
                neg_score += dp_from['neg_score']
                union_mask = SAM_MASKS[mask_id] | dp_from['union_mask']
                pos_score = iou(union_mask, SLIME_MASK, thres=IOU_THRES) * ALPHA
                
                if pos_score + neg_score > new_dp['score']:
                    new_dp['score'] = pos_score + neg_score
                    new_dp['mask_ids'] = used_mask_ids + [mask_id]
                    new_dp['neg_score'] = neg_score
                    new_dp['union_mask'] = union_mask

            if new_dp['score'] > 0:
                # speed up 100 times
                dp.append(new_dp)

        best_dp = {
            'mask_ids': [],
            'score': -1e9,
            'neg_score': 0,
            'union_mask': np.zeros(SLIME_MASK[0].shape, dtype=np.bool_)
        }
        for dp_from in dp:
            if dp_from['score'] > best_dp['score']:
                best_dp = dp_from
        return best_dp['mask_ids']

    def read_SAM_masks(self, file_name):
        with open(os.path.join(self.dataset_dir, file_name + '.pkl'), 'rb') as f:
            ori_SAM_masks = pickle.load(f)
        SAM_masks = [maskUtils.decode(m['segmentation']).astype(np.bool_) for m in ori_SAM_masks]
        return SAM_masks, ori_SAM_masks
    
    def get_cropped_gt_binary_mask(self, ann):
        x1, y1, x2, y2 = ann['position']
        polygon_masks = ann['segmentation']
        final_binary_mask = np.zeros((512, 512), dtype=np.bool_)
        for polygon_mask in polygon_masks:
            polygon_mask = np.array(polygon_mask).reshape(-1, 2)
            polygon_mask = polygon_mask - np.array([x1, y1])
            polygon_mask = polygon_mask.reshape(1, -1).tolist()

            rle = maskUtils.frPyObjects(polygon_mask, y2-y1+1, x2-x1+1)
            binary_mask = maskUtils.decode(rle)
            binary_mask = cv2.resize(binary_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            binary_mask = binary_mask.astype(np.bool_)
            final_binary_mask = np.logical_or(final_binary_mask, binary_mask)
        return final_binary_mask

    def restrict_SLiMe_masks(self, SLiMe_masks, gt_mask):
        for i, mask in enumerate(SLiMe_masks):
            SLiMe_masks[i] = np.logical_and(mask, gt_mask)
        return SLiMe_masks

    def read_SLiMe_masks(self, file_name):
        np_masks = np.load(os.path.join(self.dataset_dir, file_name + '.npy'))
        SLiMe_masks = []
        mask_ids = np.unique(np_masks).tolist()
        # ignore background
        mask_ids.remove(0)
        for id in mask_ids:
            SLiMe_masks.append(np_masks == id)
        return SLiMe_masks, mask_ids,
    
    def part_id_to_name(self, part_id):
        for part_name in self.config['part_id'].keys():
            if self.config['part_id'][part_name] == part_id:
                return part_name
        print(f"part id to name error: part name {part_name} not found")
        return 'wrong'
    
'''
python src/SAM_SLiMe_macher.py --coco_ann_dir datasets/coco/bus_square/annotations_bus.json --dataset_dir datasets/coco/bus_square --config_dir configs/pascalpart_bus.yaml --visulize_step 50 --save_vis_result outputs/coco_bus_result

python src/SAM_SLiMe_macher.py --coco_ann_dir datasets/coco/bus_square/annotations_bus.json --dataset_dir datasets/coco/bus_square --config_dir configs/pascalpart_bus.yaml
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_ann_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--config_dir', type=str, required=True)
    # parser.add_argument('--visulize_step', type=int, default=1)
    # parser.add_argument('--save_vis_result', type=str, required=True)
    args = parser.parse_args()
    # matcher = SAMSLiMeMacher(args.coco_ann_dir, args.dataset_dir, args.config_dir, args.visulize_step, args.save_vis_result)
    matcher = SAMSLiMeMacher(args.coco_ann_dir, args.dataset_dir, args.config_dir)
    matcher.process()