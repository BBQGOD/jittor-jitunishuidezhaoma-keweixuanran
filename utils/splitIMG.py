import json, os
import cv2

SRC_PATH = r"../JNeRF/logs/Car/valid"
TGT_PATH = r"."

for root, dirs, files in os.walk(SRC_PATH):
    for file in files:
        if file[-4:] == ".png":
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
            if "gt" in file:
                cv2.imwrite(os.path.join(TGT_PATH, "ref/" + file[:file.find("gt")] + "r" + file[file.find("gt")+2:]), img)
            elif "_r_" in file:
                cv2.imwrite(os.path.join(TGT_PATH, "src/" + file), img)