# -*- coding: UTF-8 -*-
import os

from dataloader_ import DataSet
from config import Config
import torch.optim as optim
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from utils import *
import torch
import time
from torch.autograd import Variable
# import onnx_to_trt as transform_trt
from logctrl import getLogger
from loss_func import CrossEntropyLabelSmoothClassWeight
from net_yyh import *
import json

# -----------------------------------   Init Log File  -------------------------------------#
logger = getLogger(__name__, "logs/all.log")
if_train_success = False
Training_message_list = []


def efficientnet_to_onnx(model, save_path="/data-rbd/newbb/bigblue/goodsGroupModels/000000000011", model_name="000000000011.onnx"):
    global if_train_success
    if "efficientnet" in config.model_name:
        model.set_swish(memory_efficient=False)
    model.eval()
    # if config.use_apex_amp_mix_precision:
    #     print("*"*60)
    #     apex_torch_model_path = save_path + "apex_temp_torch_model.pth"
    #     torch.save(apex_torch_model_path, model)
    #
    #     model = torch.load(apex_torch_model_path)
    #     model.eval()
    #     print("*" * 60)

    dummy_input = Variable(torch.randn(10, 3, config.input_size, config.input_size)).cuda()

    onnx_temp_save_path = os.path.join(save_path, model_name)
    torch.onnx.export(model, dummy_input, onnx_temp_save_path, verbose=True)
    info = "save onnx model to {}".format(onnx_temp_save_path)
    logger.info(info)
    # # -----------------------------------   Onnx to TRT   -------------------------------------#
    # transform_trt_info = []
    # transform_trt_info.append(onnx_temp_save_path)
    # transform_trt_info.append(os.path.join(save_path, "model.fp16.b32.trtmodel"))
    # results = transform_trt.onnx_to_trt(transform_trt_info)
    # os.remove(onnx_temp_save_path)
    # logger.debug(results)
    if_train_success = True


def save_mapfile(mapfile_save_path, sku_info_dict__):
    mapfile_save_path = os.path.join(mapfile_save_path, "mapfile.json")
    check_file(mapfile_save_path)

    # 保存此次训练的mapfile
    with open(mapfile_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(sku_info_dict__, json_file, ensure_ascii=False, indent=2)


def find_train_and_val_data(sku_list_, data_and_anno_root, model_info_path, sku_info_dict_):
    class_dict = {}
    path_and_label_list = []
    class_num = 0
    class_max_num = -999
    bad_data_flag = False
    class_num_lsit = []
    # 初始化类别信息(图片地址和Label)存储地址
    for _ in range(len(sku_list_)):
        class_dict[str(_)] = []

    for sku_id_name in sku_list_:
        sku_root = os.path.join(data_and_anno_root, sku_id_name)
        for img_name in os.listdir(sku_root):
            img_path = os.path.join(sku_root, img_name)

            if not os.path.exists(img_path):
                continue
            else:
                class_dict[str(class_num)].append(img_name)

        current_class_num = len(class_dict[str(class_num)])
        class_num_lsit.append(current_class_num)
        # 记录每个类别有多少样本
        info = "Num of \"{}\" class: {}".format(str(class_num), current_class_num)

        logger.info(info)

        if current_class_num > class_max_num:
            class_max_num = current_class_num
        class_num += 1
    config.class_weight = cal_class_weight(class_num_lsit)
    # 如果某一类没有一张训练图片则结束训练，返回错误
    for class_num in class_dict.keys():
        if len(class_dict[str(class_num)]) == 0:
            bad_data_flag = True
            Training_message += "goodsID: {} name: {} barcode：{} has no positive data".format(
                class_num, sku_info_dict_["goods"][class_num]["goodsID"], sku_info_dict_["goods"][class_num]["name"],
                sku_info_dict_["goods"][class_num]["barcode"])

            Training_message_list.append(Training_message)

    if bad_data_flag:
        return [path_and_label_list, Training_message_list]
    else:

        class_dict_clean = class_dict
        for class_num in class_dict_clean.keys():

            # 均衡类别样本数
            class_sample_length = len(class_dict_clean[class_num])
            copy_times = class_max_num - class_sample_length

            for i in range(copy_times):
                index = i % class_sample_length
                class_dict_clean[str(class_num)].append(class_dict_clean[str(class_num)][index])

            # 制造给__getitem__读取的list
            img_info = []
            for img_name in class_dict_clean[str(class_num)]:
                img_info.append(img_name)
                img_info.append(class_num)
                path_and_label_list.append(img_info)
                img_info = []

        save_mapfile(model_info_path, sku_info_dict_)
        logger.debug("*" * 60)
        info = "Num of Training Set After Balancing: {}".format(len(path_and_label_list))
        logger.debug(info)
        logger.debug("*" * 60)
        return [path_and_label_list, Training_message_list]


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_main(sku_id_list, sku_name_list, model_id):
    global config
    global Training_message

    # ---------------------------------   Init Config   --------------------------------#
    start_training_time = time.time()
    config = Config(sku_id_list, model_id)
    mapfile_and_trt_info_path = os.path.join(config.mapfile_and_trt_model_out_root, model_id)
    check_path_without_delete(mapfile_and_trt_info_path)

    # -----------------------------  Check and Set GPU  ---------------------------------#

    if not config.use_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_num

    # -------------------------------   Set Random Sed   -------------------------------#
    if config.set_seed:
        seed_torch()
        logger.debug("Set Random Sed")

    # -----------------------------   Init Hyper-Paramet   -----------------------------#
    logger.debug("Start Init Hyper-Paramet")

    best_acc = -1
    best_epoch = -999

    # ----------------------------- Use Cudnn-acceleration -----------------------------#
    if config.use_cudnn_accelerate:
        logger.debug("Use Cudnn-acceleration")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # ---------------------------   Build SKU Info Dict    -----------------------------#

    sku_info_dict = build_sku_info_dict(label_list=[num for num in range(config.class_num)],
                                        goodsID_list=sku_id_list,
                                        name_list=sku_name_list)

    # ----------------------------      Init DataSet    --------------------------------#
    logger.info("*" * 60)
    logger.info("Init DataSet")
    split_dataset_start_time = time.time()
    path_and_label_train_list, Training_message_list = find_train_and_val_data(sku_id_list, config.data_and_anno_root,
                                                                               mapfile_and_trt_info_path, sku_info_dict)

    if len(Training_message_list) != 0:
        return [if_train_success, Training_message_list, best_epoch, best_acc]

    else:

        if (time.time() - split_dataset_start_time) / 60 > 1:
            info = "Init DataSet Spend {} Minutes!!!!".format((time.time() - split_dataset_start_time) / 60)
            logger.debug(info)
        else:
            info = "Init DataSet Spend {} Seconds!!!!".format((time.time() - split_dataset_start_time))
            logger.debug(info)

        # -----------------------------   Init DataLoader    -------------------------------#
        train_dataset = DataSet(config, data_list=path_and_label_train_list, if_training=True, )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=config.dataLoader_num_worker, pin_memory=True)
        update_print_loss_interval(config, len(train_dataset))

        # ------------------------------   Show Config    ----------------------------------#
        logger.info("*" * 60)
        logger.info("Config Info: ")
        for item in config.__dict__.items():
            train_info = "{}: {}".format(item[0], item[1])
            logger.info(train_info)
        logger.info("*" * 60)

        # 打印GPU相关信息
        info = "Pytorch VERSION: {}".format(torch.__version__)
        logger.debug(info)
        info = "CUDA VERSION: {}".format(torch.version.cuda)
        logger.debug(info)
        info = "CUDNN VERSION: {}".format(torch.backends.cudnn.version())
        logger.debug(info)
        for i in range(torch.cuda.device_count()):
            info = "Device Name: {}".format(torch.cuda.get_device_name(i))
            logger.info(info)
        # -------------------------------   Init Model    ----------------------------------#

        if "efficientnet" in config.model_name:
            if config.use_multi_gpu:
                net = EfficientNet.from_pretrained(config.model_name, num_classes=config.class_num).cuda(
                    config.gpu_num[0])
                net.train()
            else:
                net = EfficientNet.from_pretrained(config.model_name, num_classes=config.class_num).cuda()
                # net = EfficientNet.from_pretrained_big_blue(config.model_name, num_classes=config.class_num, big_blue_pre_train_model_path=config.SKU2032_pretrain_model_path).cuda()
                net.train()
        elif "resnet" in config.model_name:
            if config.use_multi_gpu:
                net = ResNet50_minist(class_num=config.class_num).cuda(config.gpu_num[0])
                net.train()
            else:
                net = ResNet50_minist(class_num=config.class_num).cuda()
                net.train()

        # -------------------------------Use FP16 Training----------------------------------#
        if config.use_apex_amp_mix_precision:
            try:
                from apex import amp
                # from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

        # -------------------------------Set Loss Function----------------------------------#
        if config.use_label_smoothing:
            # 损失函数使用带Label Smooting的交叉熵
            criterion = CrossEntropyLabelSmoothClassWeight(config.class_num, 0.1, class_weight_list=config.class_weight)
            # criterion = CrossEntropyLabelSmooth(config.class_num, 0.05)
            if config.use_multi_gpu:
                criterion = criterion.cuda(config.gpu_num[0])
            else:
                criterion = criterion.cuda()
        else:
            # 损失函数使用交叉熵
            if config.use_multi_gpu:
                criterion = nn.CrossEntropyLoss().cuda(config.gpu_num[0])
            else:
                criterion = nn.CrossEntropyLoss().cuda()

        # -------------------------------   Set Optimizer ----------------------------------#
        # 优化函数使用 SGD 自适应优化算法
        if config.which_optimizer == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=config.base_lr, momentum=0.9, weight_decay=1e-5)
        # 优化函数使用 Adam 自适应优化算法
        elif config.which_optimizer == "adam":
            optimizer = optim.Adam(net.parameters(), lr=config.base_lr)

        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=config.train_epoch,
        #                                           steps_per_epoch=int(config.train_epoch / config.batch_size) + 1)

        # ----------------------------   Apex Mixing precision  -----------------------------#
        if config.use_apex_amp_mix_precision:
            net, optimizer = amp.initialize(net, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”
            info = "Use Apex, opt_level: {}".format("O1")
            logger.debug(info)

        if config.use_multi_gpu:
            net = nn.DataParallel(net, device_ids=config.gpu_num)

        # ------------------------------  Start Training     ---------------------------------#
        print("*" * 60)
        info = "Start Training"
        logger.info(info)
        for epoch in range(config.train_epoch):
            epoch = epoch + 1
            epoch_start_time = time.time()

            # ---------------------------  Tuning Learning Rate ------------------------------#
            if epoch in config.lr_schedule:
                use_lr = config.lr_schedule[epoch]
                set_lr(optimizer, use_lr)

            sum_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                iteration_start_time = time.time()

                if config.use_multi_gpu:
                    inputs, labels = inputs.cuda(config.gpu_num[0]), labels.cuda(config.gpu_num[0])
                    index = torch.randperm(inputs.size(0)).cuda(config.gpu_num[0])
                else:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    index = torch.randperm(inputs.size(0)).cuda()

                if config.use_mix_up:
                    lam = np.random.beta(config.mix_up_alpha, config.mix_up_alpha)
                    inputs = lam * inputs + (1 - lam) * inputs[index, :]
                    targets_a, targets_b = labels, labels[index]

                optimizer.zero_grad()  # 将梯度归零
                outputs = net(inputs)  # 将数据传入网络进行前向运算

                if config.use_mix_up:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, labels)  # 得到损失函数

                if config.use_apex_amp_mix_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()  # 反向传播
                optimizer.step()  # 通过梯度做一步参数更新
                sum_loss += loss.item()

                if i % config.print_loss_interval == config.print_loss_remainder:
                    one_iteration_time = time.time() - iteration_start_time
                    train_info = '[Epoch: %d,Iteration: %d] loss:%.03f  lr= %f  time=%.02f s' % (
                        epoch, i + 1, sum_loss / config.print_loss_interval, optimizer.param_groups[0]['lr'],
                        one_iteration_time)
                    logger.info(train_info)
                    sum_loss = 0.0

            if epoch >= config.start_eval_epoch:
                # -------------------------------   Training Acc Calculate ----------------------------------#
                net.eval()  # 将模型变换为测试模式
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in train_loader:  # 再测试整个训练集
                        if config.use_multi_gpu:
                            images, labels = images.cuda(config.gpu_num[0]), labels.cuda(config.gpu_num[0])
                        else:
                            images, labels = images.cuda(), labels.cuda()
                        output_test = net(images)
                        _, predicted = torch.max(output_test, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    one_epoch_training_acc = correct.item() / total

                train_info = "One Epoch training acc: {}".format(one_epoch_training_acc)
                logger.debug(train_info)

                if one_epoch_training_acc >= config.best_acc:
                    config.best_acc = one_epoch_training_acc
                    best_epoch = epoch
                    best_model = net

                one_epoch_time = "One epoch spend {} minutes!!!!".format((time.time() - epoch_start_time) / 60)
                logger.debug(one_epoch_time)

                if config.best_acc >= config.model_ready_train_acc and epoch >= config.model_ready_epoch:
                    logger.info(
                        "Training Acc >= {}, Early Stop !!!! Best Epoch is {}".format(config.model_ready_train_acc,
                                                                                      epoch))
                    break

                # if config.best_acc == 1.0:
                #     logger.info("Training Acc == 1.0, Early Stop !!!! Best Epoch is {}".format(epoch))
                #     break

        best_info = ("best_train_acc: {}  best_epoch: {}".format(config.best_acc, best_epoch))
        logger.debug(best_info)

        onnx_model_name = ("{}" + ".onnx").format(model_id)
        torch_model_name = os.path.join(mapfile_and_trt_info_path, model_id + ".pth")
        torch_state_dict_model_name = os.path.join(mapfile_and_trt_info_path, model_id + ".state_dict")

        if config.save_strategy == "last":
            net.eval()
            transform_trt_start_time = time.time()
            efficientnet_to_onnx(net, mapfile_and_trt_info_path, onnx_model_name)
            logger.debug(
                "Transform from pth to trt FP16 spend {} seconds".format((time.time() - transform_trt_start_time)))
        elif config.save_strategy == "best":
            transform_trt_start_time = time.time()
            best_model.eval()
            torch.save(best_model, torch_model_name)
            torch.save(best_model.state_dict(), torch_state_dict_model_name)
            efficientnet_to_onnx(best_model, mapfile_and_trt_info_path, onnx_model_name)
            logger.debug(
                "Transform from pth to trt FP16 spend {} seconds".format((time.time() - transform_trt_start_time)))
        info = "Whole Training Process Spends {} Minutes!!!!".format((time.time() - start_training_time) / 60)
        logger.debug(info)
        return [if_train_success, Training_message_list, best_epoch, config.best_acc]



if __name__ == "__main__":
     # 试验台方向
    # this_sku_id_list = ["0", "1", "2"]
    # this_sku_name_list = ["top", "side", "front"]
    # this_model_id = "%06d" % 0
    # train_result = train_main(sku_id_list=this_sku_id_list, sku_name_list=this_sku_name_list, model_id = this_model_id)
    # print("train_result: ", train_result)

    # 熊猫公交年龄
    this_sku_id_list = ["0", "1", "2", "3", "4"]
    this_sku_name_list = ["mid-life", "elders", "youth", "juvenile", "kids"]
    this_model_id = "pandabus_%06d" % 0
    train_result = train_main(sku_id_list=this_sku_id_list, sku_name_list=this_sku_name_list, model_id = this_model_id)
    print("train_result: ", train_result)