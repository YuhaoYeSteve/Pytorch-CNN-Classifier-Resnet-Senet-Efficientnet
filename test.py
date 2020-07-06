from data.folder_dataloader import DataSet
import torch
if __name__ == "__main__":
    dataset = DataSet(
        "/data/yyh/2020/Pytorch-CNN-Classifier-Resnet-Senet-Efficientnet/dataset/cifar10/train")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                               num_workers=0, pin_memory=True)
    for i, (inputs, labels) in enumerate(train_loader):
        print(inputs)
        print(labels)
