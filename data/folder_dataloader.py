from torch.utils import data
from utils.img_utils import read_image
import torchvision.transforms.functional as T
import os
import cv2
import torch
import numpy as np


class DataSet(data.Dataset):
    def __init__(self, data_root, transform, if_training=True):
        super(DataSet, self).__init__()
        self.data_root = data_root
        self.is_training = if_training
        self.transform = transform
        self.class_dict_before_balance = {}
        self.class_num_before_balance = {}
        self.class_num_after_balance = {}
        self.img_path_and_label_list = []
        self.class_max_num = 0
        self.class_num = self.find_class_number(data_root)
        self.init_class_dict_before_balance()
        self.get_data_list(data_root)
        self.balance_img_list()
        # If you want to debug, remember to set DataLoader's num_workers as 0
        self.if_debug = False

    # Build class_dict_before_balance={}
    # {
        # "0"        : ["0_0.jpg", "0_1.jpg"]
        # "........" : []
        # "9"        : ["9_0.jpg", "9_1.jpg"]
    # }

    def get_data_list(self, data_root):
        for class_name in os.listdir(data_root):
            class_root = os.path.join(data_root, class_name)
            class_num = 0
            for img_name in os.listdir(class_root):
                img_path = os.path.join(class_root, img_name)
                if os.path.exists(img_path):
                    class_num += 1
                    self.class_dict_before_balance[class_name].append(img_name)
            self.class_num_before_balance[class_name] = len(
                self.class_dict_before_balance[class_name])
            if class_num > self.class_max_num:
                self.class_max_num = class_num

    # Make each class has the same number of images(max number of all classes) by randomly copying
    def balance_img_list(self):
        for class_name in self.class_dict_before_balance.keys():
            class_sample_length = len(
                self.class_dict_before_balance[class_name])
            copy_times = self.class_max_num - class_sample_length

            for i in range(copy_times):
                index = i % class_sample_length
                self.class_dict_before_balance[class_name].append(
                    self.class_dict_before_balance[class_name][index])

            self.class_num_after_balance[class_name] = len(
                self.class_dict_before_balance[class_name])
            # Build img_path_and_label_list= []
            # [["0_0.jpg", "0"], ["0_1.jpg", "0"], ["0_2.jpg", "0"],....., ["1_0.jpg", "1"],........]
            img_info = []
            for img_name in self.class_dict_before_balance[str(class_name)]:
                img_info.append(img_name)
                img_info.append(class_name)
                self.img_path_and_label_list.append(img_info)
                img_info = []

    def find_class_number(self, path):
        class_number = len([_ for _ in os.listdir(path)])
        return class_number

    def init_class_dict_before_balance(self):
        for class_name in range(self.class_num):
            self.class_dict_before_balance[str(class_name)] = []

    def __getitem__(self, item):
        img_name, label = self.img_path_and_label_list[item]
        img_path = os.path.join(self.data_root, label, img_name)

        if not os.path.exists(img_path):
            raise ValueError("{} not exist".format(img_path))
        origin_img = read_image(img_path)
        if self.is_training:
            aug_img = self.transform(image=origin_img)['image']
        else:
            aug_img = origin_img

        if self.if_debug:
            cv2.imshow("origin_img", origin_img)
            cv2.imshow("aug", aug_img)
            if cv2.waitKey() & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        return T.to_tensor(aug_img), torch.from_numpy(np.array(int(label))), aug_img, origin_img

    def __len__(self):
        return len(self.img_path_and_label_list)
