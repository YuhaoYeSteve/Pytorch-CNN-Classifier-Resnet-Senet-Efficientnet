import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from network.resnet_pytorch.resnet import resnet50
from config.config import TaskConfig
from data.folder_dataloader import DataSet
from utils.img_utils import visual_process
from network.efficientnet_pytorch import EfficientNet
from utils.general_utils import seed_torch, build_class_info_dict, save_class_info, update_print_loss_interval, add_data_to_cuda, set_lr, save_model, load_model


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
                ret = add_data_to_cuda(ret, config)
                # visual_process(self.config, ret, model="val")
                output = net(ret["imgs"])
                _, predicted = torch.max(output, 1)
                total += ret["labels"].size(0)
                correct += (predicted == ret["labels"]).sum()
            val_acc = correct.item() / total

            train_info = "One Epoch val acc: {}".format(val_acc)
            self.config.logger.info(train_info)

            if val_acc >= self.config.best_acc:
                self.config.best_acc = val_acc
                self.config.best_epoch = epoch
                self.config.best_model = net

        net = net.train()
        return net


class Trainer(object):
    def __init__(self, config):
        self.config = config
        # -------------------------------   Set Random Sed   -------------------------------#
        if config.set_seed:
            seed = seed_torch()
            config.logger.info("Set Random Sed: {}".format(seed))

        # ---------------------------   Build SKU Info Dict    -----------------------------#
        class_info_dict = build_class_info_dict(config.class_name_list)
        save_class_info(config.log_and_model_path, class_info_dict)

        # -------------------------------  Set Training Set   ------------------------------#
        # train
        self.train_data = DataSet(
            config.train_data_root, config.train_transform, config, if_training=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=config.batch_size,
                                                        shuffle=True, num_workers=config.dataLoader_num_worker)
        update_print_loss_interval(config, len(self.train_data))
        # --------------------------------  Set Evaluator   --------------------------------#
        self.evaluator = Evaluator(config)

        # -------------------------------   Init Network  ----------------------------------#
        if "efficientnet" in config.model_name:
            if config.use_multi_gpu:
                if config.task_config["task_pretrain"]:
                    self.net = torch.load(config.efficientnet_task_pre_train_path, map_location='cpu')
                    if isinstance(self.net, torch.nn.DataParallel):
                        self.net = self.net.module
                    self.net = self.net.cuda(config.gpu_num[0])
                else:
                    self.net = EfficientNet.from_pretrained(config.model_name, num_classes=config.class_num, weights_path=self.config.efficientnet_pre_train_path).cuda(
                        config.gpu_num[0])
                self.net.train()
            else:
                self.net = EfficientNet.from_pretrained(
                    config.model_name, num_classes=config.class_num).cuda()
                self.net.train()
        elif "resnet50" in config.model_name:
            if config.use_multi_gpu:
                if config.task_config["task_pretrain"]:
                    self.net = torch.load(config.resnet_pre_train_path, map_location='cpu')
                    if isinstance(self.net, torch.nn.DataParallel):
                        self.net = self.net.module
                    self.net = self.net.cuda(config.gpu_num[0])
                else:
                    self.net = resnet50(pretrained=True, cached_file=config.resnet_pre_train_path)
                    num_ftrs = self.net.fc.in_features
                    self.net.fc = nn.Linear(num_ftrs, config.class_num) 
                    self.net = self.net.cuda(config.gpu_num[0])
                    self.net.train()
            else:
                self.net = resnet50(pretrained=True, cached_file=config.resnet_pre_train_path)
                num_ftrs = self.net.fc.in_features
                self.net.fc = nn.Linear(num_ftrs, config.class_num)
                self.net = self.net.cuda()
                self.net.train()
        if self.config.load_dataset_specified_pre_train:
            load_model(self.config.pre_train_path, self.net)
        if config.use_multi_gpu:
            self.net = nn.DataParallel(self.net, device_ids=config.gpu_num)
        # -------------------------------   Set Optimizer ----------------------------------#
        # Use SGD
        if config.which_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), lr=config.base_lr, momentum=0.9, weight_decay=1e-5)
        # Use ADAM
        elif config.which_optimizer == "adam":
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=config.base_lr)
        # -------------------------------Set Loss Function----------------------------------#
        if config.use_multi_gpu:
            self.loss_func = nn.CrossEntropyLoss().cuda(config.gpu_num[0])
        else:
            self.loss_func = nn.CrossEntropyLoss().cuda()

    def train(self):
        self.net.train()
        self.train_epoch()

    def train_epoch(self):
        for epoch in range(config.train_epochs):
            self.net = self.net.train()
            epoch = epoch + 1
            epoch_start_time = time.time()
            # ---------------------------  Tuning Learning Rate ------------------------------#
            if epoch in config.lr_schedule:
                use_lr = config.lr_schedule[epoch]
                set_lr(self.optimizer, use_lr)

            # # ----------------------------  One Epoch Training    ----------------------------#
            # for iteration, ret in enumerate(self.train_loader):
            #     iteration_start_time = time.time()
            #     index, ret = add_data_to_cuda(ret, config, True)

            #     # ------------------------  Show Training Images(Visdom)  --------------------------#
            #     if ((iteration % self.config.print_loss_interval) == int(self.config.print_loss_remainder/10) or iteration == 0) and self.config.use_visdom:
            #         visual_process(self.config, ret, model="train")

            #     # ----------------------------  Forward and Backward    ----------------------------#
            #     self.optimizer.zero_grad()
            #     outputs = self.net(ret["imgs"])                # Forward
            #     loss = self.loss_func(outputs, ret["labels"])  # Loss
            #     loss.backward()                                # Backpropagation
            #     self.optimizer.step()                          # Update Net Parameter

            #     if (iteration % self.config.print_loss_interval) == self.config.print_loss_remainder or iteration == 0:
            #         one_iteration_time = time.time() - iteration_start_time
            #         train_info = '[Epoch: %d,Iteration: %d] loss:%.06f  lr= %f  time=%.02f s' % (
            #             epoch, iteration + 1, loss.item(), self.optimizer.param_groups[0]['lr'], one_iteration_time)
            #         self.config.logger.info(train_info)

            # one_epoch_time = "One epoch spend {} minutes!!!!".format(
            #     (time.time() - epoch_start_time) / 60)
            # self.config.logger.info(one_epoch_time)
            # -------------------------------  Evaluate  ----------------------------------#
            self.net = self.evaluator.eval_val(self.net, epoch)
            if self.config.best_acc >= config.model_ready_acc and epoch >= config.model_ready_epoch:
                self.config.logger.info(
                    "Val Acc{} >= {}, Early Stop !!!! Best Epoch is {}".format(self.config.best_acc, config.model_ready_acc,
                                                                               epoch))
                break
        # ------------------------------  Save Model  ---------------------------------#
        save_model(config)


if __name__ == "__main__":
    # # efficientnet-b0
    # config = TaskConfig("efficientnet-b0")
    # tainer = Trainer(config)
    # tainer.train()
    # resnet50
    config = TaskConfig("resnet50")
    tainer = Trainer(config)
    tainer.train()
