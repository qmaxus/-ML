import cv2
import numpy as np
import os
import json
import glob

if not os.path.exists("annotations_prepped_train"):
    os.makedirs("annotations_prepped_train")

if not os.path.exists("images_prepped_train"):
    os.makedirs("images_prepped_train")

color = {'Mixed': (1, 1, 1), 'Dense': (2, 2, 2), 'Diffuse': (3, 3, 3)}
annotation = {}

for v in glob.glob('MLtest_dataset/MLtest_json (*).json'):
    with open(v, 'r') as f:
        json_annotation = json.load(f)
        for key, cap in json_annotation.items():
            if key in annotation:
                annotation[key]['regions'] += cap['regions']
            else:
                annotation[key] = cap

for key, value in annotation.items():
    name_photo = f"MLtest_dataset/{value['filename']}"
    try:
        img = cv2.imread(name_photo)
        msk_img = np.zeros((768, 768, 3), dtype='uint8')
        for b in value['regions']:
            crd = b["shape_attributes"]
            if 'Type' in b["region_attributes"]:
                x_min = int((crd["x"] * msk_img.shape[1]) / img.shape[1])
                y_min = int((crd["y"] * msk_img.shape[0]) / img.shape[0])
                x_max = int(((crd["width"] + crd["x"]) * msk_img.shape[1]) / img.shape[1])
                y_max = int(((crd["height"] + crd["y"]) * msk_img.shape[0]) / img.shape[0])
                box = np.array([[[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]], dtype=np.int32)
                cv2.fillPoly(msk_img, box, color[b["region_attributes"]["Type"]])
        img = cv2.resize(img, (768, 768))
        cv2.imwrite('images_prepped_train/' + value['filename'].replace('jpg', 'png'), img)
        cv2.imwrite('annotations_prepped_train/' + value['filename'].replace('jpg', 'png'), msk_img)
    except Exception as e:
        print(f'err {e} {name_photo}')
