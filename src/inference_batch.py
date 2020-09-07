import os
import time
import torch
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from torch.utils import data
from torchvision import models
from config.config import TaskConfig
from utils.img_utils import read_image
from data.folder_dataloader import DataSet
from utils.img_utils import visual_process
from network.efficientnet_pytorch import EfficientNet


class DataSet(data.Dataset):
    def __init__(self, data_root, transform):
        super(DataSet, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.img_path_and_label_list = []
        self.get_img_and_label()

    def __getitem__(self, item):
        img_name, label = self.img_path_and_label_list[item]
        img_path = os.path.join(
            self.data_root, self.config.class_name_list[int(label)], img_name)

        if not os.path.exists(img_path):
            raise ValueError("{} not exist".format(img_path))
        img = read_image(img_path)
        origin_img = cv2.resize(
            img.copy(), (self.config.input_size, self.config.input_size))
        aug_img = self.transform(image=img)['image']
        aug_img_show = aug_img.copy()

        aug_img = (aug_img.astype(np.float32) / 255.)
        aug_img = (aug_img - self.config.mean) / self.config.std
        # aug_img = aug_img.transpose(2, 0, 1)
        aug_img = aug_img.astype(np.float32)

        ret = {"imgs": T.to_tensor(aug_img), "labels": torch.from_numpy(
            np.array(int(label))), "aug_img": aug_img_show, "origin_img": origin_img}
        return ret

    def __len__(self):
        return len(self.img_path_and_label_list)

    def get_img_and_label(self):
        for class_name in os.listdir(self.data_root):
            class_img_root = os.path.join(self.data_root, class_name)
            for img_name in os.listdir(class_img_root):
                self.img_path_and_label_list.append([img_name, class_name])


def find_path(log_and_model_root):
    for file_name in os.listdir(log_and_model_root):
        if file_name.endswith("pth"):
            model_path = os.path.join(log_and_model_root, file_name)
            label_map_path = os.path.join(log_and_model_root, "mapfile.json")
    return model_path, label_map_path


class Evaluator(object):
    def __init__(self, config):
        self.config = config
        # ---------------------------------  Set Val Set    --------------------------------#
        self.val_data = DataSet(config.val_data_root,
                                config.val_transform, config, if_training=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=config.batch_size,
                                                      shuffle=False, num_workers=config.dataLoader_num_worker)

    def eval_val(self, net, epoch):
        net = net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, ret in enumerate(self.val_loader):
                ret = self.add_data_to_cuda(ret)
                # visual_process(self.config, ret, model="val")
                output = net(ret["imgs"])
                _, predicted = torch.max(output, 1)
                total += ret["labels"].size(0)
                correct += (predicted == ret["labels"]).sum()
            val_acc = correct.item() / total

            train_info = "One Epoch training acc: {}".format(val_acc)
            print(train_info)

    def add_data_to_cuda(self, ret, config):
        if config.use_multi_gpu:
            ret["imgs"] = ret["imgs"].cuda(gpu_num[0])
            ret["labels"] = ret["labels"].cuda(gpu_num[0])
        else:
            ret["imgs"] = ret["imgs"].cuda()
            ret["labels"] = ret["labels"].cuda()
        return ret


if __name__ == "__main__":
    log_and_model_root = "./train_out/cat_dog/efficientnet-b0_"
    pics_root = "/data/yyh/classify/data/cat_dog/val"
    gpu_num = "2, 3, 4, 5"
    size = 224
    model_path, label_map_path = find_path(log_and_model_root)
    transform = A.Resize(height=size, width=size, p=1)
    DataSet(pics_root, transform)
