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
    for i, (aug_img_tensor, labels, aug_img, origin_img) in enumerate(train_loader):
        if i % config.print_loss_interval == config.print_loss_remainder:
            if config.use_visdom:
                origin_img_show = origin_img[0].numpy().copy()
                aug_img_show = aug_img[0].numpy().copy()
                vis.image(origin_img_show.transpose(2, 0, 1)[::-1, ...], win="**********train_origin_img**********", opts={
                          'title': '**********train_origin_img {} * {}**********'.format(origin_img_show.shape[1], origin_img_show.shape[0])})

                vis.image(aug_img_show.transpose(2, 0, 1)[::-1, ...], win="**********train_aug_img**********", opts={
                          'title': '**********train_aug_img {} * {}**********'.format(aug_img_show.shape[1], aug_img_show.shape[0])})
