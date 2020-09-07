# -*- coding: UTF-8 -*-
from torchvision import transforms
from network.efficientnet_pytorch import EfficientNet
from torchvision import datasets, models, transforms
from utils import *
import torch.nn.functional as F
import json
import os
import torch
import cv2
from utils.img_utils import read_image
from PIL import Image


model_name = "efficientnet-b0"

sku_name_list = ["cat", "dog"]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_out_root = "/data/yyh/classify/train_out/"

model_path = "./train_out/cat_dog/efficientnet-b0/50_0.992125.pth"
# size = (112, 112)
size = (224, 224)
net = torch.load(model_path, map_location="cuda:0")
print("Load model: {}".format(model_path))
net = net.cuda()
net.eval()
anno_path = "./train_out/cat_dog/efficientnet-b0/mapfile.json"
with open(anno_path, "r") as mapfile:
    label_name_relation = json.load(mapfile)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor

def pre_processing(img_path):
    img = read_image(img_path)
    img = cv2.resize(img, size)
    img = Image.fromarray(img)

    img = transforms.ToTensor()(img)
    # img = normalize(img, tuple([0.5, 0.5, 0.5]), tuple([0.5, 0.5, 0.5]),
    #                 inplace=True)
    img = normalize(img, tuple([0.408, 0.447, 0.47]), tuple([0.289, 0.274, 0.278]),
                    inplace=True)
    return img


def inference_and_show_test_pics():
    if_show = False
    if_break_single_good = False
    show_wrong_stop = False
    show_right_stop = False
    show_top_n_wrong = 3
    # img_root = "/data/yyh/classify/data/shiguankou_8_17/test"
    img_root = "/data/yyh/classify/data/cat_dog/val"
    good_id_num_dict = {}
    good_id_acc_dict = {}
    good_id_wrong_dict = {}
    # 创建商品名称和数量对应的dict
    for direction in os.listdir(img_root):
        real_name = direction
        good_id_num_dict[real_name] = 0

    # 创建商品名称和ACC的对应的dict
    for direction in os.listdir(img_root):
        real_name = direction
        good_id_acc_dict[real_name] = -999

    # 创建商品名称和错误预测列表的对应的dict
    for direction in os.listdir(img_root):
        real_name = direction
        good_id_wrong_dict[real_name] = []

    all_test_img_num = 0
    all_right_num = 0
    num_of_goods_for_test = 0
    for index, direction in enumerate(os.listdir(img_root)):
        num_of_goods_for_test = index
        break_flag = False
        img_path_root = os.path.join(img_root, direction)
        single_good_test_img_num = 0
        single_good_right_num = 0
        wrong_predict_list = []
        for img_name in os.listdir(img_path_root):
            single_good_test_img_num += 1
            all_test_img_num += 1
            img_path = os.path.join(img_path_root, img_name)
            img = pre_processing(img_path)
            img = img.unsqueeze(0)
            net_out = net(img.cuda())
            predict_label = torch.max(net_out, 1)[1]

            real_label = sku_name_list.index(direction)
            score = F.softmax(net_out, 1)
            score = score[:, predict_label.item()].item()
            predict_name = label_name_relation["goods"][predict_label]["name"]
            real_name = label_name_relation["goods"][real_label]["name"]
            if int(predict_label) == real_label:
                single_good_right_num += 1
                all_right_num += 1

                print("Right")
                wrong_predict_list.append(int(predict_label.cpu()))
                print("img_path: {}".format(img_path))
                print(
                    ("img_name: {} | predict_label: {} | real_label: {}").format(img_name, int(predict_label),
                                                                                 real_label))
                print(
                    ("predict_name: {} | real_name: {} | confidence: {} ").format(predict_name, real_name,
                                                                                  round(score, 4)))

                print("#" * 60)
                # image = ((img[0, :, :, :].numpy().transpose(1, 2, 0) * [0.289, 0.274, 0.278] + [0.408, 0.447,
                #                                                                                 0.47]) * 255).astype(
                #     np.uint8)
                image = ((img[0, :, :, :].numpy().transpose(1, 2, 0) * [0.5, 0.5, 0.5] + [0.5, 0.5,
                                                                                          0.5]) * 255).astype(
                    np.uint8)
                if if_show:
                    if show_right_stop:
                        cv2.imshow("origin", image)
                        if cv2.waitKey() & 0xFF == ord('q'):
                            break_flag = True
                            cv2.destroyAllWindows()
                            break
                    else:
                        cv2.imshow("origin", image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break_flag = True
                            cv2.destroyAllWindows()
                            break

            else:
                print("Wrong")
                wrong_predict_list.append(int(predict_label.cpu()))
                print("img_path: {}".format(img_path))
                print(
                    ("img_name: {} | predict_label: {} | real_label: {}").format(img_name, int(predict_label),
                                                                                 real_label))
                print(
                    ("predict_name: {} | real_name: {} | confidence: {} ").format(predict_name, real_name,
                                                                                  round(score, 4)))

                print("#" * 60)
                image = ((img[0, :, :, :].numpy().transpose(1, 2, 0) * [0.289, 0.274, 0.278] + [0.408, 0.447,
                                                                                                0.47]) * 255).astype(
                    np.uint8)
                if if_show:
                    if show_wrong_stop:
                        cv2.imshow("origin", image)
                        if cv2.waitKey() & 0xFF == ord('q'):
                            break_flag = True
                            cv2.destroyAllWindows()
                            break
                    else:
                        cv2.imshow("origin", image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break_flag = True
                            cv2.destroyAllWindows()
                            break

            # image = cv2.resize(image, (image.shape[0] * 2, image.shape[1] * 2))
            # im_cv2_show = write_ctext_on_img(image, predict_label, (1, 30), "red")
            # im_cv2_show = write_ctext_on_img(im_cv2_show, round(scroe, 2), (10, 80), "red")
        single_good_acc = round(
            single_good_right_num / single_good_test_img_num, 4)
        if single_good_acc != 1.0:
            wrong_name_list = []
            wrong_id = count_list(wrong_predict_list, show_top_n_wrong)
            for id in wrong_id:
                wrong_name = label_name_relation["goods"][int(id)]["name"]
                wrong_name_list.append(wrong_name)

            good_id_wrong_dict[real_name] = (wrong_name_list)

        good_id_acc_dict[real_name] = single_good_acc
        good_id_num_dict[real_name] = single_good_test_img_num

        if not if_break_single_good:
            if break_flag:
                break

    acc_list = []
    test_pics_num_list = []

    for good_name in good_id_num_dict:
        num = good_id_num_dict[good_name]
        print("num of {} : {} ".format(good_name, num))
        test_pics_num_list.append(num)

    print("#" * 60)
    for good_name in good_id_acc_dict:
        single_good_acc = good_id_acc_dict[good_name]
        acc_list.append(single_good_acc)
        print("Acc of {} : {} ".format(good_name, single_good_acc))
    print("#" * 60)
    print("Average_acc: {}".format(all_right_num / all_test_img_num))
    print("{} goods for test".format(num_of_goods_for_test + 1))
    print("#" * 60)
    print("每个类别对应的错误分类Top3： ")
    for key in good_id_wrong_dict:
        print("{} \t: {} ".format(key, good_id_wrong_dict[key]))


if __name__ == "__main__":
    inference_and_show_test_pics()
