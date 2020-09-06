import torch
from tqdm import tqdm
from torchvision import models
import torch.optim as optim
from config.config import TaskConfig
from network.efficientnet_pytorch import EfficientNet
from data.folder_dataloader import DataSet
from utils.general_utils import seed_torch, build_class_info_dict, save_class_info, update_print_loss_interval, add_data_to_cuda, set_lr
from utils.img_utils import visual_training_process
import torch.nn as nn
import time


class Evaltor(object):
    def __init__(self, config):
        # ---------------------------------  Set Val Set    --------------------------------#
        self.val_data = DataSet(config.val_data_root,
                                config.transform, config, if_training=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=config.batch_size,
                                                      shuffle=True, pin_memory=True, num_workers=config.dataLoader_num_worker)


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
            config.train_data_root, config.transform, config, if_training=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=config.batch_size,
                                                        shuffle=True,  num_workers=config.dataLoader_num_worker)
        update_print_loss_interval(config, len(self.train_data))
        # --------------------------------  Set Evaltor   ----------------------------------#
        self.evaltor = Evaltor(config)

        # -------------------------------   Init Network  ----------------------------------#
        if "efficientnet" in config.model_name:
            if config.use_multi_gpu:
                self.net = EfficientNet.from_pretrained(config.model_name, num_classes=config.class_num).cuda(
                    config.gpu_num[0])
                self.net.train()
            else:
                self.net = EfficientNet.from_pretrained(
                    config.model_name, num_classes=config.class_num).cuda()
                self.net.train()
        elif "resnet50" in config.model_name:
            if config.use_multi_gpu:
                self.net = models.resnet50(pretrained=True)
                num_ftrs = self.net.fc.in_features
                self.net.fc = nn.Linear(num_ftrs, config.class_num)
                self.net = self.net.cuda(config.gpu_num[0])
                self.net.train()
            else:
                self.net = models.resnet50(pretrained=True)
                num_ftrs = self.net.fc.in_features
                self.net.fc = nn.Linear(num_ftrs, config.class_num)
                self.net = self.net.cuda()
                self.net.train()

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
            epoch = epoch + 1
            epoch_start_time = time.time()
            # ---------------------------  Tuning Learning Rate ------------------------------#
            if epoch in config.lr_schedule:
                use_lr = config.lr_schedule[epoch]
                set_lr(self.optimizer, use_lr)
            # bar = tqdm(iter(self.train_loader), ascii=True)

            # ----------------------------  One Epoch Training    ----------------------------#
            # for iteration, ret in enumerate(bar):
            for iteration, ret in enumerate(self.train_loader):
                iteration_start_time = time.time()
                index, ret = add_data_to_cuda(ret, config)
                if (iteration % self.config.print_loss_interval) == int(self.config.print_loss_remainder/10) or iteration == 0:
                    visual_training_process(self.config, ret)
                # imgs = ret["img"].cuda()
                # labels = ret["label"].cuda()
                # print(labels)

                # bar.set_description("label{}".format(ret["label"]))
                # bar.update()
                self.optimizer.zero_grad()
                outputs = self.net(ret["img"])                # Forward
                loss = self.loss_func(outputs, ret["label"])  # Loss
                loss.backward()                               # Backpropagation
                self.optimizer.step()                         # Update Net Parameter

                if (iteration % self.config.print_loss_interval) == self.config.print_loss_remainder or iteration == 0:
                    one_iteration_time = time.time() - iteration_start_time
                    train_info = '[Epoch: %d,Iteration: %d] loss:%.06f  lr= %f  time=%.02f s' % (
                        epoch, iteration + 1, loss.item() * 10, self.optimizer.param_groups[0]['lr'], one_iteration_time)
                    self.config.logger.info(train_info)
            one_epoch_time = "One epoch spend {} minutes!!!!".format(
                (time.time() - epoch_start_time) / 60)
            self.config.logger.debug(one_epoch_time)


if __name__ == "__main__":
    config = TaskConfig()
    tainer = Trainer(config)
    tainer.train()
