# copy images of specific categories from pascalpart

import scipy.io as sio
import os
import shutil
from tqdm import tqdm

class PascalPartSelecter():
    def __init__(self, annotations_part_path, images_path):
        self.annotations_part_path = annotations_part_path
        self.images_path = images_path

    def copy(self, category_name, output_dir):
        img_name_list = self.img_name_list_of_cat(category_name)
        print(f'pascalpart selecter: copying {category_name} images to {output_dir}')
        for img_name in tqdm(img_name_list):
            img_path = os.path.join(self.images_path, img_name + '.jpg')
            anno_path = os.path.join(self.annotations_part_path, img_name + '.mat')
            shutil.copy(img_path, output_dir)
            shutil.copy(anno_path, output_dir)

    def img_name_list_of_cat(self, cat):
        img_name_list = []
        print(f"pascalpart selecter: getting image name list of category {cat}")
        for file in tqdm(os.listdir(self.annotations_part_path)):
            mat = sio.loadmat(os.path.join(self.annotations_part_path, file))
            objs = mat['anno']['objects'][0][0]
            for obj in objs['class'][0][:]:
                if obj[0] == cat:
                    img_name_list.append(file.split('.')[0])
                    break
        return img_name_list

'''
python src/pascalpart_selecter.py --annotations_part_path D:/dataset/pascal_part/Annotations_Part --images_path D:/dataset/pascalVOC2010/VOCdevkit/VOC2010/JPEGImages --cat_names bus dog --output_dir ./datasets/pascalpart
'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_part_path', type=str, required=True)
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--cat_names', nargs='+', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    selecter = PascalPartSelecter(args.annotations_part_path, args.images_path)
    for cat_name in args.cat_names:
        output_dir = os.path.join(args.output_dir, cat_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        selecter.copy(cat_name, output_dir)
