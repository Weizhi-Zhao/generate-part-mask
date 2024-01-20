import itertools
from SAM_SLiMe_matcher import SAMSLiMeMacher
import multiprocessing
from typing import Optional
import numpy as np
import os
import json
from tqdm import tqdm

class MultiprocessingSSM(SAMSLiMeMacher):
    def __init__(self, coco_ann_dir, dataset_dir, config_dir, num_workers=None, visulize_step: Optional[int]=None, save_vis_result: Optional[str]=None, SAM_GT_THRES=0.3):
        super().__init__(coco_ann_dir, dataset_dir, config_dir, visulize_step, save_vis_result, SAM_GT_THRES)
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count() - 2
        else:
            self.num_workers = num_workers

    def match(self):
        print(f"SAM SLiMe Macher: processing images of {self.config['category_name']}")
        self.coco_anns = self.coco_anns[:100]
        coco_anns_split = np.array_split(self.coco_anns, len(self.coco_anns) / 3)
        annotations = []
        with multiprocessing.Pool(self.num_workers) as pool:
            for result in tqdm(pool.imap(self.worker, coco_anns_split), total=len(coco_anns_split)):
                annotations.extend(result)
        if self.visulize_step is None:
            with open(os.path.join(self.dataset_dir, f"part_annotations_{self.config['category_name']}.json"), 'w') as f:
                json.dump(annotations, f)
        
'''
python src/multiprocessing_SAM_SLiMe_Matcher.py --coco_ann_dir datasets/coco/bus_square/square_annotations_bus.json --dataset_dir datasets/coco/bus_square --config_dir configs/pascalpart_bus.yaml
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_ann_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--config_dir', type=str, required=True)
    # parser.add_argument('--visulize_step', type=int, default=1)
    # parser.add_argument('--save_vis_result', type=str, required=True)
    # parser.add_argument('--num_workers', type=int, required=True)
    args = parser.parse_args()
    matcher = MultiprocessingSSM(args.coco_ann_dir, args.dataset_dir, args.config_dir)

    matcher.match()
