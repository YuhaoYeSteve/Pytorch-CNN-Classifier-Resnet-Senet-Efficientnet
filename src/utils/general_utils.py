import datetime
import os
import json
import torch
import shutil
import numpy as np
import yaml


# ---------------------------   Time  ------------------------------#
def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# ---------------------------   Log   ------------------------------#
# Write Log into txt
def save_log(txt_path, string_contant):
    with open(txt_path, "a") as f:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S:')
        string_contant = nowTime + string_contant + "\n"
        print(string_contant)
        f.writelines(string_contant)


# ---------------------------   File   -----------------------------#
def check_file_without_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
    else:
        print("Do not exist：", path)
        os.system("touch {}".format(path))
        print("Create： ", path)


def check_file_with_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
        os.system("rm {} -r".format(path))
        print("Delete： ", path)
        os.system("touch {}".format(path))
        print("Create： ", path)
    else:
        print("Do not exist：", path)
        os.system("touch {}".format(path))
        print("Create： ", path)


def check_path_with_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
        shutil.rmtree(path)
        print("Delete： ", path)
        os.makedirs(path)
        print("Create： ", path)
    else:
        print("Do not exist：", path)
        os.makedirs(path)
        print("Create： ", path)


def check_path_without_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
    else:
        os.makedirs(path)
        print("Create： ", path)


# --------------------------   Random   ----------------------------#
def seed_torch(seed=3):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed

# --------------------------   Print    -----------------------------#
# 跟新打印loss的间隔


def update_print_loss_interval(config, length_of_dataset, k=1.5):
    if (length_of_dataset / config.batch_size) < k * 10:
        config.print_loss_interval = 1
        config.print_loss_remainder = 0
    elif k * 10 <= (length_of_dataset / config.batch_size) < k * 100:
        config.print_loss_interval = 10
        config.print_loss_remainder = 9
    elif k * 100 <= (length_of_dataset / config.batch_size) < k * 1000:
        config.print_loss_interval = 100
        config.print_loss_remainder = 99
    elif k * 1000 <= (length_of_dataset / config.batch_size) < k * 10000:
        config.print_loss_interval = 1000
        config.print_loss_remainder = 999
    elif (length_of_dataset / config.batch_size) >= k * 10000:
        config.print_loss_interval = 10000
        config.print_loss_remainder = 9999


# --------------------------   Yaml    -----------------------------#
def load_yaml(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data)
    return data


# ---------------------   Save class info    ------------------------#
def save_class_info(class_info_save_path, class_info_dict):
    class_info_save_path = os.path.join(class_info_save_path, "mapfile.json")
    check_file_with_delete(class_info_save_path)

    # 保存此次训练的mapfile
    with open(class_info_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(class_info_dict, json_file, ensure_ascii=False, indent=2)


# ------- Build Relationship between class num and class Name -------#
def build_class_info_dict(name_list=[]):
    sku_info_dict = {
        "info": []
    }

    for index, label in enumerate(range(len(name_list))):
        single_sku_info = {}
        single_sku_info["index"] = index + 1
        single_sku_info["name"] = name_list[index]
        sku_info_dict["info"].append(single_sku_info)

    return sku_info_dict
    # {
    #   "info": [
    #     {
    #       "index": 1,
    #       "name": "cat"
    #     },
    #     {
    #       "index": 2,
    #       "name": "dog"
    #     }
    #   ]
    # }


def log_dict(dictionaries, format_, logger):
    for key in dictionaries.keys():
        contant = format_.format(key, dictionaries[key])
        logger.info(contant)


def add_data_to_cuda(ret, config):
    if config.use_multi_gpu:
        ret["img"] = ret["img"].cuda(config.gpu_num[0])
        ret["label"] = ret["label"].cuda(config.gpu_num[0])
        index = torch.randperm(ret["img"].size(0)).cuda(config.gpu_num[0])
    else:
        ret["img"] = ret["img"].cuda()
        ret["label"] = ret["label"].cuda()
        index = torch.randperm(ret["img"].size(0)).cuda()
    return index, ret


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    log_path = "./1.txt"
    train_info = "loss: {}".format(round(0.2222123123124, 6))
    save_log(log_path, train_info)