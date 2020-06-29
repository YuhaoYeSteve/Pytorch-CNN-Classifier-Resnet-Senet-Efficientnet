import CameraLineAutoDetectImpl as clad
import os
import cv2
import datetime

with open("trt_path.txt", "r") as f:
    trt_model_path = f.readlines()[0] 


test_img_root = "./test_pics"
img_path_list = []

with open("test_imgs.txt", "r") as f:
    img_path_list = [os.path.join(test_img_root, _.strip("\n")) for _ in f.readlines()]

img_path_list.append(trt_model_path)
results = clad.big_blue_six_direction_classifier(img_path_list)
print(results)