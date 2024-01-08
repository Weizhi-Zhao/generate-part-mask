# zwz
from tqdm import tqdm
import pickle
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

class SAM_Mask_Generator():
    def __init__(self, model_name, checkpoint, device, file_path):
        sam = sam_model_registry[model_name](checkpoint=checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.file_path = file_path

    def run(self):
        for file in tqdm(os.listdir(self.file_path)):
            if not file.endswith('.png'):
                continue
            img = cv2.imread(os.path.join(self.file_path, file))
            masks = self.mask_generator.generate(img)
            with open(os.path.join(self.file_path, file.replace('.png', '.pkl')), 'wb') as f:
                pickle.dump(masks, f)

'''
python src/sam_full_image_mask_generator.py --checkpoint checkpoints\sam\sam_vit_h_4b8939.pth --file_path datasets/coco/dog_square
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vit_h')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--file_path', type=str, required=True)
    args = parser.parse_args()
    # for file in os.listdir(args.file_path):
    #     if not file.endswith('.pkl'):
    #         continue
    #     print(file)
    #     img = cv2.imread(os.path.join(
    #         args.file_path, file.replace('.pkl', '.png')))
    #     with open(os.path.join(args.file_path, file), 'rb') as f:
    #         masks = pickle.load(f)
    #     plt.figure()
    #     plt.imshow(img)
    #     show_anns(masks)
    #     plt.axis('off')
    #     plt.show()
    generator = SAM_Mask_Generator(args.model_name, args.checkpoint, args.device, args.file_path)
    generator.run()

# plt.figure(figsize=(20,20))
# plt.imshow(img)
# show_anns(masks)
# plt.axis('off')
# plt.show()
