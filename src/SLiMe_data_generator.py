import yaml
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import cv2


class SLiMeDataGenerator:
    def __init__(self, config, obj_size_thres, dataset_path, target_path):
        '''
        obj_size_thres: area of object box / area of image, if smaller than this thres, ignore the image
        '''
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_path = dataset_path
        self.obj_size_thres = obj_size_thres
        self.target_path = os.path.join(
            target_path, self.config['category_name'])
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)

    def save(self):
        masks = self.generate_masks()
        print(
            f"SLiMe Data Genetator: saving image and part masks of {self.config['category_name']} to {self.target_path}")
        for mask in tqdm(masks):
            img_name, part_masks = mask['name'], mask['part_masks']
            img = cv2.imread(os.path.join(
                self.dataset_path, img_name + '.jpg'))
            cv2.imwrite(os.path.join(self.target_path, img_name + '.png'), img)
            np.save(os.path.join(self.target_path,
                    img_name + '.npy'), part_masks)

    def generate_masks(self):
        '''
        returns a list, each element is a dict:
        {
            image name: str (without extension (.jpg))
            mask of part: ndarray(int32)
        }
        '''
        path = self.dataset_path
        masks = []
        print(
            f"SLiMe Data Genetator: generating image and part masks of {self.config['category_name']} from {self.dataset_path}")
        for file in tqdm(os.listdir(path)):
            if not file.endswith('.mat'):
                continue
            mat = sio.loadmat(os.path.join(path, file))
            objs = mat['anno']['objects'][0][0]
            img_shape = objs[0][0][2].shape
            part_masks = np.zeros(objs[0][0][2].shape, np.int32)
            obj_masks = np.zeros(objs[0][0][2].shape, np.bool_)
            for obj in objs[0][:]:
                # select objects that have the same name
                if obj['class'][0] != self.config['category_name']:
                    continue
                # make sure part number is not 0
                if obj[3][:].shape == (0, 0):
                    continue
                # conbine all object masks
                obj_masks = np.logical_or(obj_masks, obj[2] == 1)\
                    # conbine all part masks
                for part in obj[3][:][0]:
                    part_name = part['part_name'][0].split('_')[0]
                    part_id = self.part_name_to_id(part_name)
                    prior = np.ones(part_masks.shape, np.int32) * part_id
                    part_masks[np.logical_and(
                        part['mask'] == 1, part_masks <= prior)] = part_id
            # make sure there is at least one object
            if np.all(~obj_masks):
                continue
            # bbox: [x1, y1, x2, y2]
            _, bbox_area = self.get_bbox(obj_masks)
            # make sure the object is not too small
            if bbox_area / (img_shape[0] * img_shape[1]) < self.obj_size_thres:
                continue
            one_img_mask = {
                'name': file.split('.')[0],
                'part_masks': part_masks
            }
            masks.append(one_img_mask)
        return masks

    def part_name_to_id(self, part_name):
        for aggregate_part_name in self.config['part_aggregation'].keys():
            if part_name in self.config['part_aggregation'][aggregate_part_name]:
                return self.config['part_id'][aggregate_part_name]
        print("error: part name not found")
        return 0

    def get_bbox(self, mask):
        # mask: ndarray(np.bool_)
        rows, cols = np.where(mask)
        x1 = np.min(cols)
        y1 = np.min(rows)
        x2 = np.max(cols)
        y2 = np.max(rows)
        area = (x2 - x1) * (y2 - y1)
        return [x1, y1, x2, y2], area


'''
python src/SLiMe_data_generator.py --config configs/pascalpart_bus.yaml --obj_size_thres 0.2 --dataset_path datasets/pascalpart/bus --target_path datasets/SLiMe

python src/SLiMe_data_generator.py --config configs/pascalpart_dog.yaml --obj_size_thres 0.2 --dataset_path datasets/pascalpart/dog --target_path datasets/SLiMe
'''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--obj_size_thres', type=float, default=0.25)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    args = parser.parse_args()
    generator = SLiMeDataGenerator(args.config, args.obj_size_thres,
                                     args.dataset_path, args.target_path)
    generator.save()
