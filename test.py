import torch
from init_visdom import init_visdom_
from data.folder_dataloader import DataSet
from config.config import TaskConfig
from utils.general_utils import update_print_loss_interval
vis = init_visdom_(window_name="classifier_test")


if __name__ == "__main__":
    config = TaskConfig()
    dataset = DataSet(
        data_root="/data/yyh/2020/Pytorch-CNN-Classifier-Resnet-Senet-Efficientnet/dataset/test/train", transform=config.transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=True)
    update_print_loss_interval(config, len(dataset))
    for i, (aug_img, labels, origin_img) in enumerate(train_loader):
        if i % config.print_loss_interval == config.print_loss_remainder:
            
            print(i)
            print(labels)
