import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from torch.utils import data
from torchvision import models
import torch.nn.functional as F
from config.config import TaskConfig
from utils.img_utils import read_image
from utils.general_utils import load_yaml
from data.folder_dataloader import DataSet
from utils.img_utils import visual_process
import torchvision.transforms.functional as T
from network.efficientnet_pytorch import EfficientNet


class DataSet(data.Dataset):
    def __init__(self, data_root, transform):
        super(DataSet, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.class_name_list = self.get_class_name()
        self.img_path_and_label_list = []
        self.get_img_and_label()

    def __getitem__(self, item):
        img_name, label = self.img_path_and_label_list[item]
        img_path = os.path.join(
            self.data_root, self.class_name_list[label], img_name)

        if not os.path.exists(img_path):
            raise ValueError("{} not exist".format(img_path))
        img = read_image(img_path)
        aug_img = self.transform(image=img)['image']

        aug_img = (aug_img.astype(np.float32) / 255.)
        aug_img = (aug_img - mean) / std
        # aug_img = aug_img.transpose(2, 0, 1)
        aug_img = aug_img.astype(np.float32)

        ret = {"imgs": T.to_tensor(aug_img), "labels": torch.from_numpy(
            np.array(label)), "img_index": torch.from_numpy(np.array(int(item)))}
        return ret

    def __len__(self):
        return len(self.img_path_and_label_list)

    def get_class_name(self):
        return [class_name for class_name in os.listdir(self.data_root)]

    def get_img_and_label(self):
        for class_name in os.listdir(self.data_root):
            class_img_root = os.path.join(self.data_root, class_name)
            for img_name in os.listdir(class_img_root):
                self.img_path_and_label_list.append(
                    [img_name, self.class_name_list.index(class_name)])


def find_path(log_and_model_root):
    for file_name in os.listdir(log_and_model_root):
        if file_name.endswith("pth"):
            model_name = file_name.split(".")[0]
            model_path = os.path.join(log_and_model_root, file_name)
            label_map_path = os.path.join(log_and_model_root, "mapfile.json")
    return model_path, label_map_path, model_name


class Evaluator(object):
    def __init__(self, pics_root, transform):
        # ---------------------------------  Set Val Set    --------------------------------#
        self.pics_root = pics_root
        self.val_data = DataSet(pics_root, transform)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=len(gpu_num) * 64,
                                                      shuffle=False, num_workers=min(8 * len(gpu_num), 32))
        self.real_label_list = []
        self.wrong_index_list = []
        self.wrong_label_list = []
        self.wrong_score_list = []
        # self.val_loader = torch.utils.data.DataLoader(
        #     self.val_data, batch_size=4, shuffle=False, num_workers=0)

    def eval_val(self, net):
        net = net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, ret in enumerate(self.val_loader):
                ret = self.add_data_to_cuda(ret)
                # visual_process(self.config, ret, model="val")
                output = net(ret["imgs"])
                _, predicted = torch.max(output, 1)
                score = F.softmax(output, 1)
                total += ret["labels"].size(0)
                right = predicted == ret["labels"]
                wrong = list(torch.eq(right, 0).cpu().numpy())
                for index, answer in enumerate(wrong):
                    if answer == 1:
                        self.wrong_index_list.append(
                            int(ret["img_index"][index].cpu()))
                        self.wrong_label_list.append(
                            int(predicted[index].cpu()))
                        self.wrong_score_list.append(
                            float(score[index, predicted[index]].cpu()))
                        self.real_label_list.append(ret["labels"][index])
                correct += (right).sum()
            val_acc = correct.item() / total

            train_info = "One Epoch training acc: {}".format(val_acc)
            print(train_info)

    def save_wrong_case(self):
        for index, wrong_dataloader_index in enumerate(self.wrong_index_list):
            wrong_pic_name = self.val_data.img_path_and_label_list[wrong_dataloader_index][0]
            real_label_index = int(self.real_label_list[index].cpu())
            real_label = self.val_data.class_name_list[real_label_index]
            wrong_pic_save_root = os.path.join(
                log_and_model_root, "wrong_case", real_label)
            if not os.path.exists(wrong_pic_save_root):
                os.makedirs(wrong_pic_save_root)
            wrong_pic = cv2.imread(os.path.join(
                pics_root, self.val_data.class_name_list[real_label_index], wrong_pic_name))
            wrong_label_index = self.wrong_label_list[index]
            wrong_label = self.val_data.class_name_list[wrong_label_index]
            score = self.wrong_score_list[index]
            wrong_pic_name = "{}_{}.jpg".format(wrong_label, round(score, 4))
            wrong_pic_save_path = os.path.join(
                wrong_pic_save_root, wrong_pic_name)
            cv2.imwrite(wrong_pic_save_path, wrong_pic)
            # print(wrong_pic_path_list)

    def add_data_to_cuda(self, ret):
        if len(gpu_num) > 1:
            ret["imgs"] = ret["imgs"].cuda(gpu_num[0])
            ret["labels"] = ret["labels"].cuda(gpu_num[0])
        else:
            ret["imgs"] = ret["imgs"].cuda()
            ret["labels"] = ret["labels"].cuda()
        return ret


if __name__ == "__main__":
    # -----------------------   Path   -------------------------#
    # log_and_model_root = "./train_out/cat_dog/efficientnet-b0_"
    # pics_root = "/data/yyh/classify/data/cat_dog/val"
    pics_root = "./dataset/cifar10/val"
    log_and_model_root = "./train_out/cifar10/efficientnet-b0"
    model_path, label_map_path, model_name = find_path(log_and_model_root)
    # -----------------------  Config   ------------------------#
    if "efficientnet" in model_name:
        mean = [0.408, 0.447, 0.47]
        std = [0.289, 0.274, 0.278]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    # gpu_num = [1, 2, 3]
    gpu_num = [3, 4, 5, 6]
    size = 224
    transform = A.Resize(height=size, width=size, p=1)
    # ----------------------- Load Model   ---------------------#
    net = torch.load(model_path, map_location='cpu')
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    net = net.cuda(gpu_num[0])
    if len(gpu_num) > 0:
        net = nn.DataParallel(net, device_ids=gpu_num)
    # ----------------------- Inference   ----------------------#
    evaluator = Evaluator(pics_root, transform)
    evaluator.eval_val(net)
    evaluator.save_wrong_case()
