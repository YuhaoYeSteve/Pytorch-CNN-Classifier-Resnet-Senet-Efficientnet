import os
import cv2
import torch
import numpy as np
from torch.utils import data
from utils.general_utils import log_dict
import torchvision.transforms.functional as T
from utils.img_utils import read_image, visdom_show_opencv



class DataSet(data.Dataset):
    def __init__(self, data_root, transform, config, if_training=True):
        super(DataSet, self).__init__()
        self.config = config
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
        self.log_class_num()
        if self.if_debug:
            self.num_workers = 0
            self.batch_size = 1
    # Build class_dict_before_balance={}
    # {
        # "0"        : ["0_0.jpg", "0_1.jpg"]
        # "........" : []
        # "9"        : ["9_0.jpg", "9_1.jpg"]
    # }

    def get_data_list(self, data_root):
        for index, class_name in enumerate(os.listdir(data_root)):
            class_root = os.path.join(data_root, class_name)
            class_num = 0
            for img_name in os.listdir(class_root):
                img_path = os.path.join(class_root, img_name)
                if os.path.exists(img_path):
                    class_num += 1
                    self.class_dict_before_balance[str(index)].append(img_name)
            self.class_num_before_balance[str(index)] = len(
                self.class_dict_before_balance[str(index)])
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
        img_path = os.path.join(self.data_root, self.config.class_name_list[int(label)], img_name)

        if not os.path.exists(img_path):
            raise ValueError("{} not exist".format(img_path))
        img = read_image(img_path)
        origin_img = cv2.resize(img.copy(), (self.config.input_size, self.config.input_size))
        aug_img = self.transform(image=img)['image']
        aug_img_show = aug_img.copy()

        if self.if_debug:
            # server debug
            title = '**********train_origin_img_getitem__ {} * {}**********'
            win = '**********train_origin_img_getitem__**********'
            visdom_show_opencv(self.config.vis, origin_img.copy(), title, win)
            title = '**********train_auged_img_getitem__ {} * {}**********'
            win = '**********train_auged_img_getitem__**********'
            visdom_show_opencv(self.config.vis, aug_img.copy(), title, win)
            # local debug
            # cv2.imshow("origin_img", origin_img)
            # cv2.imshow("aug", aug_img)
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()

        aug_img = (aug_img.astype(np.float32) / 255.)
        aug_img = (aug_img - self.config.mean) / self.config.std
        # aug_img = aug_img.transpose(2, 0, 1)
        aug_img = aug_img.astype(np.float32)

        ret = {"imgs": T.to_tensor(aug_img), "labels": torch.from_numpy(np.array(int(label))), "aug_img": aug_img_show, "origin_img": origin_img}
        return ret

    def __len__(self):
        return len(self.img_path_and_label_list)

    def log_class_num(self):
        # log before balance
        if self.is_training:
            self.config.logger.info("Training Set")
        else:
            self.config.logger.info("Val Set")
        format_ = "Num of \"{}\" class: {}"
        log_dict(self.class_num_before_balance, format_, self.config.logger)
        # log after balance
        # format_ = "Num of whole Training Set After Balancing: {}".format(len(path_and_label_list))