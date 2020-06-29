import cv2
import os
import shutil
import datetime
import random
import numpy as np
import torch


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def write_ctext_on_img(img, contant, position, color="red"):
    if color == "red":
        bgr = (0, 0, 255)
    elif color == "blue":
        bgr = (255, 0, 0)
    elif color == "green":
        bgr = (0, 255, 0)
    else:
        raise Exception("Choose from \"red\", \"blue\", \"green\", Please!!!! ")

    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    img_show = cv2.putText(img.copy(), str(contant), (x, y), font, 1, (0, 0, 255), 1)
    return img_show


def save_txt(prediction, img_name_head, output_txt_name):
    contant = img_name_head + "," + str(prediction) + "\n"
    with open(output_txt_name, "a") as f:
        f.write(contant)


def check_file(path):
    if os.path.exists(path):
        os.remove(path)
        print("Delete： ", path)
    else:
        print("Do not exist：", path)


def check_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Delete： ", path)
        os.mkdir(path)
    else:
        os.mkdir(path)
        print("Create： ", path)


def check_path_without_delete(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        print("Create： ", path)


def seed_torch(seed=3):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# torch.backends.cudnn.deterministic = True


# 跟新打印loss的间隔
def update_print_loss_interval(config, length_of_dataset):
    if (length_of_dataset / config.batch_size) < 2 * 10:
        config.print_loss_interval = 1
        config.print_loss_remainder = 0
    elif 2 * 10 <= (length_of_dataset / config.batch_size) < 2 * 100:
        config.print_loss_interval = 10
        config.print_loss_remainder = 9
    elif 2 * 100 <= (length_of_dataset / config.batch_size) < 2 * 1000:
        config.print_loss_interval = 100
        config.print_loss_remainder = 99
    elif 2 * 1000 <= (length_of_dataset / config.batch_size) < 2 * 10000:
        config.print_loss_interval = 1000
        config.print_loss_remainder = 999
    elif (length_of_dataset / config.batch_size) >= 2 * 10000:
        config.print_loss_interval = 10000
        config.print_loss_remainder = 9999


# 读取图片
def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)


# 建立label和goodsID, name, barcode的对应字典
def build_sku_info_dict(label_list=[], goodsID_list=[], name_list=[]):
    sku_info_dict = {
        "goods": []
    }

    for index, label in enumerate(label_list):
        single_sku_info = {}
        single_sku_info["index"] = index + 1
        single_sku_info["name"] = name_list[index]
        single_sku_info["goodsID"] = goodsID_list[index]
        sku_info_dict["goods"].append(single_sku_info)

    return sku_info_dict


def count_list(list_which_need_to_count, top_n=1, if_debug=False):
    num_dic = {}
    maximun_num_list = []
    for i in list_which_need_to_count:  # 循环列表中的元素
        s = set()  # 定义集合
        if i not in s:  # 如果i不在集合中
            num_dic[i] = list_which_need_to_count.count(i)

    if if_debug:
        for i in num_dic:
            print("{} : {}".format(i, num_dic[i]))

    if top_n < len(num_dic.keys()):
        for i in range(top_n):
            max_num = 0
            max_key = 0
            for key in num_dic:
                if num_dic[key] > max_num:
                    max_num = num_dic[key]
                    max_key = key
            num_dic.pop(max_key)
            maximun_num_list.append(max_key)
    else:
        for i in range(len(num_dic.keys())):
            max_num = 0
            max_key = 0
            for key in num_dic:
                if num_dic[key] > max_num:
                    max_num = num_dic[key]
                    max_key = key
            num_dic.pop(max_key)
            maximun_num_list.append(max_key)

    return maximun_num_list


# def cal_class_weight(class_num_lsit):
#     import numpy as np
#     class_weight_lsit = []
#     sum = np.sum(np.asarray(class_num_lsit), axis=0)
#     for num in class_num_lsit:
#         class_weight_lsit.append(num / sum)
#     return class_weight_lsit


def cal_class_weight(nums, weight_low=0.5, weight_high=2):
    minnum = min(nums)
    maxnum = max(nums)
    numgap = maxnum - minnum
    weightgap = weight_high - weight_low

    weights = []
    for num in nums:
        weight = (1 - (num - minnum) / numgap) * weightgap + weight_low
        weights.append(weight)

    weights = np.array(weights)
    return weights


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 2, 3, 4, 1, 1, 1, 1. - 87, -9999, -123]
    # top_n_list = count_list(a, 31)
    # print(top_n_list)
    print(cal_class_weight(a, 0.5, 2))
