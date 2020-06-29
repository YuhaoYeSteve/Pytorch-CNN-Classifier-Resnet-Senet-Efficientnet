from torch.utils import data
from aug_ import *
import random
from utils import *
from config_cifar10 import Config
# from config import Config
# from visual import vis
import time




class DataSet(data.Dataset):
    def __init__(self, config, data_list=[], if_training=True, if_cal_mean_and_std=False):
        super(DataSet, self).__init__()
        self.config = config
        self.is_training = if_training
        self.path_and_label_list = data_list
        self.class_num = 0

        # 是否在训练前根据当前样本得到mean和std
        if if_cal_mean_and_std:
            self.training_dataset_mean_list = []
            self.training_dataset_std_list = []
        else:
            self.training_dataset_mean_list = config.mean
            self.training_dataset_std_list = config.std

    def __getitem__(self, item):

        img_name, label = self.path_and_label_list[item]
        img_path = os.path.join(self.config.data_and_anno_root, str(label), img_name)
        test_flag = False
        img = read_image(img_path)

        if self.is_training:
            img = cv2_to_pil_without_cvt(img)
            if test_flag:
                img = cv2_to_pil(img)
                img_origin = transforms.ToTensor()(img)
                # vis.image(img_origin,
                #           win="origin", opts={'title': 'origin'}
                #           )
            if_aug = random.randint(0, 1)  # 50% chance for augmentation
            if test_flag:
                if_aug = 1
            if if_aug == 1:

                # num_of_aug = random.randint(1, 3)
                num_of_aug = 1

                if test_flag:
                    aug_funtion = [RandomCrop]
                else:
                    # aug_funtion = [adjust_brightness, adjust_contrast, adjust_saturation, rotate,
                    #                random_erasing, gaussian_noise, motion_blur, zuoyou_flip, shangxia_flip]

                    aug_funtion = [adjust_brightness, adjust_contrast, adjust_saturation, rotate,
                                   random_erasing, gaussian_noise, motion_blur]

                aug_funtion_choose = random.sample(aug_funtion, num_of_aug)

                if test_flag:
                    # vis.text(win="aug_func", text=aug_funtion_choose[0].__name__)
                    pass

                for single_func in aug_funtion_choose:
                    if single_func.__name__ == "adjust_saturation":
                        factor = random.uniform(0.5, 1.5)
                        img = single_func(factor, img)

                    elif single_func.__name__ == "adjust_contrast":
                        factor = random.uniform(0.5, 1.5)
                        img = single_func(factor, img)

                    elif single_func.__name__ == "adjust_brightness":
                        factor = random.uniform(0.5, 1.5)
                        img = single_func(factor, img)

                    elif single_func.__name__ == "rotate":
                        factor = random.randint(-20, 20)
                        img = single_func(factor, img)

                    elif single_func.__name__ == "random_erasing":
                        img = single_func(img)

                    elif single_func.__name__ == "gaussian_noise":
                        img = single_func(img)

                    elif single_func.__name__ == "motion_blur":
                        img = single_func(img)

                    elif single_func.__name__ == "perspective_transfor":
                        img = single_func(img)

                    elif single_func.__name__ == "zuoyou_flip":
                        img = single_func(img)

                    elif single_func.__name__ == "shangxia_flip":
                        img = single_func(img)

                    elif single_func.__name__ == "RandomCrop":
                        img = single_func(img)
            else:
                if test_flag:
                    pass
                    # vis.text(win="aug_func", text="origin")

            # 后期替换成albumentations库，纯opencv操作
            img = pil_to_cv2_without_cvt(img)
            img = cv2.resize(
                img, (self.config.input_size, self.config.input_size))

            if test_flag:
                img = cv2_to_pil(img)
            else:
                img = cv2_to_pil_without_cvt(img)

            img = transforms.ToTensor()(img)

            if test_flag:
                # vis.image(img,
                #           win="after_aug", opts={'title': 'after_aug'}
                #           )
                time.sleep(2)

            img = normalize(img, tuple(self.training_dataset_mean_list), tuple(self.training_dataset_std_list),
                            inplace=True)
            label = torch.from_numpy(np.array(int(label)))

            return img, label
        else:
            img = cv2.resize(
                img, (self.config.input_size, self.config.input_size))
            img = cv2_to_pil_without_cvt(img)
            img = transforms.ToTensor()(img)
            img = normalize(img, tuple(self.training_dataset_mean_list), tuple(self.training_dataset_std_list),
                            inplace=True)
            label = torch.from_numpy(np.array(int(label)))
            return img, label

    def __len__(self):

        return len(self.path_and_label_list)

    def get_image_mean_std(self, image):
        """input: image object PIL or numpy ndarray"""
        if type(image) != 'numpy.ndarray':
            image = np.array(image)
        image = image / 255.
        image_mean = []
        image_std = []
        for i in range(3):
            image_mean.append(np.mean(image[:, :, i]))
            image_std.append(np.std(image[:, :, i]))
        return image_mean, image_std

    def get_training_mean_std(self, train_image_dataset):
        """input, train_image_dataset: [[classes:image]]"""
        mean_ = []
        std_ = []
        for path in train_image_dataset:
            image = Image.open(path)
            mean, std = self.get_image_mean_std(image)
            mean_.append(mean)
            std_.append(std)
        return mean_, std_

    def calculate_mean_std(self, mean_list, std_list):
        sum_count = len(mean_list)
        sum_mean = np.sum(np.array(mean_list), axis=0)
        sum_std = np.sum(np.array(std_list), axis=0)
        mean = sum_mean / sum_count
        std = sum_std / sum_count
        return mean, std
