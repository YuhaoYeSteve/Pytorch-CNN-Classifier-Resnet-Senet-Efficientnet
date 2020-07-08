import torch
from tqdm import tqdm
import torch.optim as optim
from config.config import TaskConfig
from init_visdom import init_visdom_
from network.efficientnet_pytorch import EfficientNet
from data.folder_dataloader import DataSet
vis = init_visdom_(window_name="train_centernet_test")


class Evaltor(object):
    def __init__():
        # ---------------------------------  Set Val Set  --------------------------------#
        self.val_data = DataSet(config.val_data_root, if_training=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=config.batch_size,
                                                      shuffle=True, pin_memory=True, num_workers=config.num_workers)


class Trainer(object):
    def __init__(self):
        # -------------------------------   Set Random Sed   -------------------------------#
        if config.set_seed:
            seed_torch()
            logger.debug("Set Random Sed")
        # -------------------------------  Set Training Set  ------------------------------#
        # train
        self.train_data = DataSet(config.train_data_root, if_training=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=config.batch_size,
                                                        shuffle=True, pin_memory=True, num_workers=config.num_workers)
        # --------------------------------  Set Evaltor   ----------------------------------#
        self.evaltor = Evaltor()

        # -------------------------------   Init Network  ----------------------------------#
        if "efficientnet" in config.model_name:
            self.model = EfficientNet.from_pretrained(config.model_name, num_classes=config.class_num).cuda(
                config.gpu_num[0])

        # -------------------------------   Set Optimizer ----------------------------------#
        # Use SGD
        if config.which_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=config.base_lr, momentum=0.9, weight_decay=1e-5)
        # Use ADAM
        elif config.which_optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=config.base_lr)

    def train(self):
        self.model.train()
        for epoch in range(config.train_epochs):
            bar = tqdm(iter(self.train_loader), ascii=True)
            for imgs, labels in bar:
                imgs = imgs.cuda()
                labels = labels.cuda()
                print(labels)


if __name__ == "__main__":
    config = TaskConfig()
    tainer = Trainer()
    tainer.train()
