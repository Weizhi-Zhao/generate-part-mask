# copy images of specific categories from coco

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm

# deleted iscrowd images

class COCOSelecter:
    def __init__(self, ann_file, coco_path):
        self.coco = COCO(ann_file)
        self.coco_path = coco_path

    def copy(self, category_name, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cat_ids = self.coco.getCatIds(catNms=[category_name])
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        img_infos = self.coco.loadImgs(img_ids)
        print(f"coco selecter: copying {category_name} images to {output_dir}")
        for img_info in tqdm(img_infos):
            ann_ids = self.coco.getAnnIds(imgIds=img_info["id"], catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            # if any of the annotations is crowd, skip the image
            if any(ann['iscrowd'] for ann in anns):
                continue
            img_path = os.path.join(self.coco_path, img_info["file_name"])
            shutil.copy(img_path, output_dir)


'''
python src/coco_selecter.py --ann_file D:/dataset/coco/annotations_trainval2017/annotations/instances_train2017.json --coco_path D:/dataset/coco/train2017 --cat_names bus dog --output_dir ./datasets/coco
'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--coco_path", type=str, required=True)
    parser.add_argument("--cat_names", nargs='+', type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    selecter = COCOSelecter(args.ann_file, args.coco_path)
    for cat_name in args.cat_names:
        output_dir = os.path.join(args.output_dir, cat_name)
        selecter.copy(cat_name, output_dir)
