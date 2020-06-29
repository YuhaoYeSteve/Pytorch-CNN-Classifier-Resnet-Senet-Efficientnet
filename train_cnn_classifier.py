from torch import optim
from torch.utils.data import DataLoader
from config.config_big_blue import config_bb as config
from feature_extractor import GeneralFeatureExtractor
from utils_coleection.net_utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from data.bigblue_dataset import DataSet
from tqdm import tqdm

img = cv2.imread()

class Evaltor(object):
    def __init__():
        pass

class Trainer(object):
    def __init__(self):

        # -------------------------------   Init Network  ----------------------------------#
        self.model = GeneralFeatureExtractor()
        
        # -------------------------------   Set Optimizer ----------------------------------#
        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model.backbone)

        
        if config.model_type == "mobilfacenet":
            self.optimizer = optim.SGD([
                                {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                {'params': [paras_wo_bn[-1]] + [self.model.head.kernel], 'weight_decay': 4e-4},
                                {'params': paras_only_bn}
                            ], lr = config.lr, momentum = config.momentum)
        else:
            self.optimizer = optim.SGD([
                                {'params': paras_wo_bn + [self.model.head.kernel], 'weight_decay': 5e-4},
                                {'params': paras_only_bn}
                            ], lr = config.lr, momentum = config.momentum)

        # -------------------------------   Set Dataset  ----------------------------------#
        self.train_data = DataSet(config, if_training=True)
        self.train_loader = DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    
    def train(self):
        self.model.backbone.train()
        self.model.head.train()
        for epoch in range(config.train_epochs):
            bar = tqdm(iter(self.train_loader), ascii=True)
            for imgs, labels in bar:
                imgs = imgs.cuda()
                labels = labels.cuda()
                print(labels)
                




if __name__ == "__main__":
    tainer = Trainer()
    tainer.train()

    
    