import cv2

# 读取图片
def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)