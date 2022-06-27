import json, os
import cv2

JSONPATH = [r"../data/nerf_synthetic/CarTestAll/transforms_train.json",
            r"../data/nerf_synthetic/CarTestAll/transforms_val.json"]
TARGET_PATH = r"../data/nerf_synthetic/CarTestAll/transforms_valid_new.json"
ROOT_PATH = r"../data/nerf_synthetic/CarTestAll"

imgs = []
camera_angle_x = 0.0
for file in JSONPATH:
    with open(file, "r") as f:
        data = json.load(f)
        camera_angle_x = data["camera_angle_x"]
        for frame in data["frames"]:
            img = cv2.imread(os.path.join(ROOT_PATH, frame["file_path"]) + ".png", cv2.IMREAD_UNCHANGED)
            imgs.append((frame["transform_matrix"], img))

new_json = {
    "camera_angle_x": camera_angle_x,
    "frames": []
}
for id, (tr, img) in enumerate(imgs):
    cv2.imwrite(os.path.join(ROOT_PATH, "new/r_"+str(id)+".png"), img)
    new_json["frames"].append({
        "file_path": "./new/r_"+str(id),
        "transform_matrix": tr
    })
with open(TARGET_PATH, "w") as f:
    json.dump(new_json, f)
