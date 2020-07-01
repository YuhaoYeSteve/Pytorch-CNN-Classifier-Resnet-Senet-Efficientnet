from config.config import TaskConfig
from init_visdom import init_visdom_
from data.folder_dataloader import
from network.efficientnet_pytorch import EfficientNet
vis = init_visdom_(window_name="train_centernet_test")

class Evaltor(object):
    def __init__():
        pass

class Trainer(object):
    def __init__(self):



        # -------------------------------    Set Dataset  ----------------------------------#
        self.train_data = DataSet(config, if_training=True)
        self.train_loader = DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)

        # -------------------------------   Init Network  ----------------------------------#
        if "efficientnet" in config.model_name:
            self.model = EfficientNet.from_pretrained(config.model_name, num_classes=config.class_num).cuda(
                        config.gpu_num[0])
        
        # -------------------------------   Set Optimizer ----------------------------------#


        
    
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

    
    