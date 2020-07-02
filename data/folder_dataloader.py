from torch.utils import data
from utils.img_utils import read_image
import os
import cv2


class DataSet(data.Dataset):
    def __init__(self, data_root, if_training=True):
        super(DataSet, self).__init__()
        self.is_training = if_training
        self.class_num = self.find_class_number(data_root)
        self.class_dict_before_balance = {}
        self.init_class_dict_before_balance()

        # create list for __getitm__ usage
        self.img_path_list, self.label_list = self.get_data_list(data_root)

    # Generate image and label list
    def get_data_list(self, data_root):
        for class_name in os.listdir(data_root):
            class_root = os.path.join(data_root, class_name)
            class_num = 0
            for img_name in os.listdir(class_root):
                if os.path.exists(img_path):
                    class_num += 1
                    class_dict_before_balance[class_name].append(img_name)
        return balanced_img_list, balanced_label_list

    # Make each class has the same number of images(max number of all classes) by randomly copying for t
    def balance_img_list():
        pass

    def find_class_number(self, path):
        class_number = len([_ for _ in os.listdir(path)])
        return class_number
    
    def init_class_dict_before_balance(self):
        for class_name in range(self.class_num):
                self.class_dict_before_balance[str(class_name)] = []

    def __getitem__(self, item):
        return img, label

    def __len__(self):
        return len(self.img_path_list)
