import json, os
import cv2

SRC_PATH = r"../jRealGAN/datasets/Car_pair/multiscale_sub_ref"
TGT_PATH = r"../jrealGAN/datasets/Car_pair/multiscale_sub_ref_new"

for root, dirs, files in os.walk(SRC_PATH):
    for file in files:
        img = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
        rows, cols = img.shape[:2]
        res = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(TGT_PATH, file), res)
