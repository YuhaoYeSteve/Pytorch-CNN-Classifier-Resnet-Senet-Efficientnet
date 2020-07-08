import cv2
import numpy as np

# 读取图片
def read_image(path):
    # return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    return cv2.imread(path)