# -*- coding: UTF-8 -*-
from aug_ import *
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from utils import *
import torch.nn.functional as F
import json
from net_yyh import *
import os
model_id = 0
model_id = "%06d" % model_id
sku_name_list = ["top", "side", "front"]

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
root = "/data-rbd/newbb/bigblue/innGoodsClean/"
model_out_root = "/data/yyh/classify/train_out/"

model_path = os.path.join(model_out_root, "{}/{}.pth".format(model_id, model_id)) 

net = torch.load(model_path)
net = net.cuda()
net.eval()
anno_path = os.path.join(model_out_root,"{}/mapfile.json".format(model_id))
with open(anno_path, "r") as mapfile:
    label_name_relation = json.load(mapfile)

# sku_list = []
sku_list = ["test.video.images"]


# for sku_info in label_name_relation["goods"]:
#     sku_list.append(sku_info["goodsID"])


def pre_processing(img_path):
    img = read_image(img_path)
    img = cv2.resize(img, (64, 64))
    img = cv2_to_pil_without_cvt(img)

    img = transforms.ToTensor()(img)
    img = normalize(img, tuple([0.408, 0.447, 0.47]), tuple([0.289, 0.274, 0.278]),
                    inplace=True)
    return img


def find_real_label(sku_id):
    global label_name_relation
    real_label = 0
    for single_sku_info in label_name_relation["goods"]:
        if single_sku_info["goodsID"] == str(sku_id):
            return real_label
        real_label += 1


def inference_and_show_sku():
    for sku_id in sku_list:

        sku_root = os.path.join(root, sku_id)
        folder_list = os.listdir(sku_root)
        for folder in folder_list:
            if folder.endswith(".json"):
                anno_file_name = folder
                json_path = os.path.join(sku_root, anno_file_name)
                with open(json_path) as anno_file:
                    anno_list = json.load(anno_file)
                    for each_img_anno_dict in anno_list["imgs"]:
                        img_name = each_img_anno_dict["filename"]
                        img_path = os.path.join(sku_root, "imgs", img_name + ".jpg")
                        img = pre_processing(img_path)
                        img = img.unsqueeze(0)
                        net_out = net(img.cuda())
                        predict_label = torch.max(net_out, 1)[1]
                        score = F.softmax(net_out, 1)
                        score = score[:, predict_label.item()].item()
                        predict_name = label_name_relation["goods"][predict_label]["name"]
                        print("predict_label: {} | predict_name: {} | confidence: {} ".format(int(predict_label) + 1),
                              predict_name, score)
                        image = ((img[0, :, :, :].numpy().transpose(1, 2, 0) * [0.289, 0.274, 0.278] + [0.408, 0.447,
                                                                                                        0.47]) * 255).astype(
                            np.uint8)
                        # image = cv2.resize(image, (image.shape[0] * 2, image.shape[1] * 2))
                        # im_cv2_show = write_ctext_on_img(image, predict_label, (1, 30), "red")
                        # im_cv2_show = write_ctext_on_img(im_cv2_show, round(scroe, 2), (10, 80), "red")
                        cv2.imshow("origin", image)
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break


# def inference_and_show_test_pics():
#     for sku_id in sku_list:
#         img_root = os.path.join(root, sku_id)
#         for img_name in os.listdir(img_root):
#             img_path = os.path.join(img_root, img_name)
#             img = pre_processing(img_path)
#             img = img.unsqueeze(0)
#             net_out = net(img.cuda())
#             predict_label = torch.max(net_out, 1)[1]
#             score = F.softmax(net_out, 1)
#             score = score[:, predict_label.item()].item()
#             predict_name = label_name_relation["goods"][predict_label]["name"]

#             print(("img_name: {} | predict_label: {} | predict_name: {} | confidence: {} ").format(img_name, int(
#                 predict_label) + 1,
#                                                                                                    predict_name, score))
#             image = ((img[0, :, :, :].numpy().transpose(1, 2, 0) * [0.289, 0.274, 0.278] + [0.408, 0.447,
#                                                                                             0.47]) * 255).astype(
#                 np.uint8)
#             # image = cv2.resize(image, (image.shape[0] * 2, image.shape[1] * 2))
#             # im_cv2_show = write_ctext_on_img(image, predict_label, (1, 30), "red")
#             # im_cv2_show = write_ctext_on_img(im_cv2_show, round(scroe, 2), (10, 80), "red")
#             cv2.imshow("origin", image)
#             if cv2.waitKey(100) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break


def inference_and_show_test_pics_model8_release():
    if_show = True
    if_show_stop = True
    if_break_single_good = True
    show_top_n_wrong = 3
    img_root = "/datav/yyh/big_blue_release_no8_test_samples/cleaned"
    good_id_num_dict = {}
    good_id_acc_dict = {}
    good_id_wrong_dict = {}
    # 创建商品名称和数量对应的dict
    for good_id in os.listdir(img_root):
        real_name = label_name_relation["goods"][int(good_id)]["name"]
        good_id_num_dict[real_name] = 0

    # 创建商品名称和ACC的对应的dict
    for good_id in os.listdir(img_root):
        real_name = label_name_relation["goods"][int(good_id)]["name"]
        good_id_acc_dict[real_name] = -999

    # 创建商品名称和错误预测列表的对应的dict
    for good_id in os.listdir(img_root):
        real_name = label_name_relation["goods"][int(good_id)]["name"]
        good_id_wrong_dict[real_name] = []

    all_test_img_num = 0
    all_right_num = 0
    num_of_goods_for_test = 0
    for index, good_id in enumerate(os.listdir(img_root)):
        num_of_goods_for_test = index
        break_flag = False
        img_path_root = os.path.join(img_root, good_id)
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
            real_label = int(good_id)
            score = F.softmax(net_out, 1)
            score = score[:, predict_label.item()].item()
            predict_name = label_name_relation["goods"][predict_label]["name"]
            real_name = label_name_relation["goods"][real_label]["name"]

            if int(predict_label) == real_label:
                single_good_right_num += 1
                all_right_num += 1
                # print("Right")
            else:
                print("Wrong")
                wrong_predict_list.append(int(predict_label.cpu()))
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
                    if if_show_stop:
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
        single_good_acc = round(single_good_right_num / single_good_test_img_num, 4)
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
    for key in good_id_wrong_dict:
        print("{} : {} ".format(key, good_id_wrong_dict[key]))


def inference_and_show_test_pics():
    if_show = True
    if_break_single_good = False 
    show_wrong_stop = True 
    show_right_stop = True 
    show_top_n_wrong = 3
    img_root = "/data/yyh/classify/test_trt_pics"
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
            print(net_out)
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
                image = ((img[0, :, :, :].numpy().transpose(1, 2, 0) * [0.289, 0.274, 0.278] + [0.408, 0.447,
                                                                                                0.47]) * 255).astype(
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
        single_good_acc = round(single_good_right_num / single_good_test_img_num, 4)
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


def test_single_pics():
    pass


def inference_main():
    inference_and_show_sku()
    # inference_and_show_test_pics()


if __name__ == "__main__":
    inference_and_show_test_pics()
