# zwz
from tqdm import tqdm
import pickle
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from binary_mask_IOU import iomax
from pycocotools import mask
import sys
sys.path.append("./src/Semantic-SAM")
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator

class SemanticSAMMaskGenerator():
    def __init__(self, file_path, cover_thres=0.9):
        self.mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='checkpoints/Semantic-SAM/swinl_only_sam_many2many.pth')) # model_type: 'L' / 'T', depends on your checkpint
        self.file_path = file_path
        # how much area of a mask should be covered by another mask to be considered as a cover relation
        self.cover_thres = cover_thres

    def run(self):
        print(f"gnerating masks for {self.file_path}")
        for file in tqdm(os.listdir(self.file_path)):
            if not file.endswith('.png'):
                continue
            # img = cv2.imread(os.path.join(self.file_path, file))
            original_image, input_image = prepare_image(image_pth=os.path.join(self.file_path, file))
            masks = self.mask_generator.generate(input_image)
            masks = self.topological_sort_masks(masks)
            with open(os.path.join(self.file_path, file.replace('.png', '.pkl')), 'wb') as f:
                pickle.dump(masks, f)
    
    '''
    sort the masks by cover relation, 
    smaller masks are in a higher priority, 
    masks who have a cover relation shouls be put more near
    '''
    def topological_sort_masks(self, ori_masks):
        self.graph = defaultdict(list)
        binary_masks = [mask.decode(m['segmentation']).astype(np.bool_) for m in ori_masks]
        # in degree of each node
        node_in = [0] * len(ori_masks)
        for i, u in enumerate(ori_masks):
            for j, v in enumerate(ori_masks):
                if u['area'] > v['area'] and iomax(binary_masks[i], binary_masks[j]) > self.cover_thres:
                    self.graph[i].append(j)
                    node_in[j] += 1
        
        visited = [False] * len(ori_masks)
        order = []
        for u in range(len(ori_masks)):
            if node_in[u] == 0 and not visited[u]:
                self.topological_sort(u, visited, order)
        # import pdb; pdb.set_trace()
        new_masks = [ori_masks[i] for i in order]
        return new_masks

    def topological_sort(self, u, visited, order):
        visited[u] = True
        for v in self.graph[u]:
            if not visited[v]:
                self.topological_sort(v, visited, order)
        order.append(u)

'''
python src/semantic_sam_full_image_mask_generator.py --checkpoint checkpoints\sam\sam_vit_h_4b8939.pth --file_path datasets/coco/dog_square
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--file_path', type=str, required=True)
    args = parser.parse_args()

    generator = SemanticSAMMaskGenerator(args.file_path)
    generator.run()
