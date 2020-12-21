#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import pycococreatortools

ROOT_DIR = "./rendering_1/"
IMAGE_DIR = os.path.join(ROOT_DIR, "mix")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "masks_standard")

json_filename = 'mix_instances_bearing_train.json'

INFO = {
    "description": "Bearing Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "Sun",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'Bearing',
        'supercategory': 'Bearing',
    },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg','*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, time_id):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    # basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    # file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(time_id, os.path.splitext(os.path.basename(f))[0].split('_')[-3])]
    print(files)

    return files

def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 0
    segmentation_id = 0
    
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                time_id = os.path.splitext(image_filename)[0].split('_')[-1]
                # annotation_files = filter_for_annotations(root, files, image_filename.replace('rgbImg', 'Mask4Seg'))
                annotation_files = filter_for_annotations(root, files, time_id)
                print(annotation_files)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename.split('/')[-1]][0]
                    print(class_id)

                    #COCO 支持两种类型的标注，其格式取决于标注的是单个物体(single object) 还是密集物体("crowd" objects).
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    
                    # tolerance 参数表示了用于记录单个物体的轮廓精度. 该参数数值越大，则标注的质量越低，但文件大小也越小. 一般采用 tolerance=2
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/{}'.format(ROOT_DIR, json_filename), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
