# -*- coding: UTF-8 -*-
import os

# from dataloader_ import DataSet
from config_cifar10 import Config
import torch.optim as optim
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from utils import *
import torch
import time
from torch.autograd import Variable
from logctrl import getLogger
from loss_func import CrossEntropyLabelSmoothClassWeight
from net_yyh import *
import json
from torchvision import transforms


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_main(sku_id_list, sku_name_list, model_id):
    global config

    # ---------------------------------   Init Config   --------------------------------#
    start_training_time = time.time()
    config = Config(sku_id_list, model_id)

    if config.if_warmup:
        if config.use_apex_amp_mix_precision:
            model_id = "cifar10_base_224_warmup_apex"
        else:
            model_id = "cifar10_base_224_warmup"
    else:
        if config.use_apex_amp_mix_precision:
            model_id = "cifar10_base_224_apex_{}".format(config.warmup_lr)
        else:
            model_id = "cifar10_base_224_apex_{}".format(config.warmup_lr)
            
    print("model_id: ", model_id)
    mapfile_and_trt_info_path = os.path.join(config.mapfile_and_trt_model_out_root, model_id)
    # -------------------------------   Init Log File  ---------------------------------#
    logger = getLogger(__name__, os.path.join(mapfile_and_trt_info_path, "train.log"))
    check_path_without_delete(mapfile_and_trt_info_path)

    # ----------------------------- Check and Set GPU  ---------------------------------#
    gpu_str = ""
    for index, gpu_num in enumerate(config.gpu_num):
        if index < len(config.gpu_num) - 1:
            gpu_str +=  str(gpu_num) + ", "
        else:
            gpu_str += str(gpu_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    config.gpu_num = [_ for _ in range(len(config.gpu_num))]
    # -------------------------------   Set Random Sed   -------------------------------#
    if config.set_seed:
        seed_torch()
        logger.info("Set Random Sed")

    # -----------------------------   Init Hyper-Paramet   -----------------------------#
    logger.info("Start Init Hyper-Paramet")

    best_acc = -1
    best_epoch = -999

    # ----------------------------- Use Cudnn-acceleration -----------------------------#
    if config.use_cudnn_accelerate:
        logger.info("Use Cudnn-acceleration")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # ----------------------------      Init DataSet    --------------------------------#
    logger.info("*" * 60)
    logger.info("Init DataSet")
    split_dataset_start_time = time.time()


    cifar10_aug = [
                    transforms.Resize((config.input_size, config.input_size)),
                    transforms.CenterCrop(config.input_size),
                    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(config.mean, config.std)
                ]
    cifar10_aug_log = [fun_name.__class__.__name__ for fun_name in cifar10_aug]
    transform_train = transforms.Compose(cifar10_aug)
    logger.info(cifar10_aug_log)
    # transform_train = transforms.Compose([
    #                                 transforms.Resize((config.input_size, config.input_size)),
    #                                 # transforms.RandomRotation(degrees=30),
    #                                 transforms.CenterCrop(config.input_size),
    #                                 transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.2),
    #                                 transforms.RandomHorizontalFlip(p=0.5),
    #                                 # transforms.RandomVerticalFlip(p=0.5),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(config.mean, config.std)]
    #                                 )

    transform_test = transforms.Compose([
                                transforms.Resize((config.input_size, config.input_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(config.mean, config.std)]
                                )

    trainset = torchvision.datasets.CIFAR10(
        root='./', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.dataLoader_num_worker)

    testset = torchvision.datasets.CIFAR10(
        root='./', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=config.dataLoader_num_worker)

    update_print_loss_interval(config, len(trainset))

    # ------------------------------   Show Config    ----------------------------------#
    logger.info("*" * 60)
    logger.info("Config Info: ")
    for item in config.__dict__.items():
        train_info = "{}: {}".format(item[0], item[1])
        logger.info(train_info)
    logger.info("*" * 60)

    # 打印GPU相关信息
    info = "Pytorch VERSION: {}".format(torch.__version__)
    logger.info(info)
    info = "CUDA VERSION: {}".format(torch.version.cuda)
    logger.info(info)
    info = "CUDNN VERSION: {}".format(torch.backends.cudnn.version())
    logger.info(info)
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
        optimizer = optim.Adam(net.parameters(), lr=config.warmup_lr)
        print("config.warmup_lr: ", config.warmup_lr)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=config.train_epoch,
    #                                           steps_per_epoch=int(config.train_epoch / config.batch_size) + 1)

    # ----------------------------   Apex Mixing precision  -----------------------------#
    if config.use_apex_amp_mix_precision:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”
        info = "Use Apex, opt_level: {}".format("O1")
        logger.info(info)

    if config.use_multi_gpu:
        net = nn.DataParallel(net, device_ids=config.gpu_num)

    # ------------------------------  Start Training     ---------------------------------#
    print("*" * 60)
    info = "Start Training"
    logger.info(info)
    for epoch in range(config.train_epoch):
        net.train()
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
                train_info = '[Epoch: %d,Iteration: %d] loss:%.06f  lr= %f  time=%.02f s' % (
                    epoch, i + 1, sum_loss / config.print_loss_interval, optimizer.param_groups[0]['lr'],
                    one_iteration_time)
                logger.info(train_info)
                sum_loss = 0.0

        if epoch >= 0:
            # -------------------------------   Test Acc Calculate ----------------------------------#
            net.eval()  # 将模型变换为测试模式
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:  # 测试测试集
                    if config.use_multi_gpu:
                        images, labels = images.cuda(config.gpu_num[0]), labels.cuda(config.gpu_num[0])
                    else:
                        images, labels = images.cuda(), labels.cuda()
                    output_test = net(images)
                    _, predicted = torch.max(output_test, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                test_acc = correct.item() / total

            test_info = "Test Acc: {}".format(test_acc)
            logger.info(test_info)

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in train_loader:  # 测试整个训练集
                    if config.use_multi_gpu:
                        images, labels = images.cuda(config.gpu_num[0]), labels.cuda(config.gpu_num[0])
                    else:
                        images, labels = images.cuda(), labels.cuda()
                    output_test = net(images)
                    _, predicted = torch.max(output_test, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                one_epoch_training_acc = correct.item() / total

            train_info = "Train Acc: {}".format(one_epoch_training_acc)
            logger.info(train_info)
            torch_model_name = os.path.join(mapfile_and_trt_info_path, str(epoch) + "_" + str(round(test_acc, 4)) + ".pth")
            torch_state_dict_model_name = os.path.join(mapfile_and_trt_info_path, str(epoch) + "_" + str(round(test_acc, 4)) + ".state_dict")
            print(torch_model_name)
            if config.use_multi_gpu:
                torch.save(net.module, torch_model_name)
                torch.save(net.module.state_dict(), torch_state_dict_model_name)
            else:
                torch.save(net, torch_model_name)
                torch.save(net.state_dict(), torch_state_dict_model_name)
            logger.info("Save model to {}".format(torch_model_name))
            if test_acc >= config.best_acc:
                config.best_acc = test_acc
                best_epoch = epoch
                best_model = net

            one_epoch_time = "One epoch spend {} minutes!!!!".format((time.time() - epoch_start_time) / 60)
            logger.info(one_epoch_time)


    best_info = ("best_train_acc: {}  best_epoch: {}".format(config.best_acc, best_epoch))
    logger.info(best_info)

    

    if config.save_strategy == "best":
        torch_best_model_name = os.path.join(mapfile_and_trt_info_path, "best_{}".format(round(config.best_acc, 4)) + ".pth")
        torch_best_state_dict_name = os.path.join(mapfile_and_trt_info_path, "best_{}".format(round(config.best_acc, 4)) + ".state_dict")
        if config.use_multi_gpu:
                torch.save(best_model.module, torch_best_model_name)
                torch.save(best_model.module.state_dict(), torch_best_state_dict_name)
        else:
            torch.save(best_model, torch_best_model_name)
            torch.save(best_model.state_dict(), torch_best_state_dict_name)
    info = "Whole Training Process Spends {} Minutes!!!!".format((time.time() - start_training_time) / 60)
    logger.info(info)
    return [best_epoch, config.best_acc]



if __name__ == "__main__":

    # Cifar-10
    this_sku_id_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    this_sku_name_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    this_model_id = "cifar10_base_224_warmup"
    # this_model_id = "cifar10_base_224_0.01"
    train_result = train_main(sku_id_list=this_sku_id_list, sku_name_list=this_sku_name_list, model_id = this_model_id)
    print("train_result: ", train_result)