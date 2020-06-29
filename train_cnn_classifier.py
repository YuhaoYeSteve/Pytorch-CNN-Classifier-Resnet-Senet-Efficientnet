from torch import optim
from torch.utils.data import DataLoader
from config.config_big_blue import config_bb as config
from feature_extractor import GeneralFeatureExtractor
from utils_coleection.net_utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from data.bigblue_dataset import DataSet
from tqdm import tqdm

from init_visdom import init_visdom_
vis = init_visdom_(window_name="train_centernet_test")

class Evaltor(object):
    def __init__():
        pass

class Trainer(object):
    def __init__(self):

        # -------------------------------   Init Network  ----------------------------------#
        self.model = 
        
        # -------------------------------   Set Optimizer ----------------------------------#


        # -------------------------------   Set Dataset  ----------------------------------#
        self.train_data = DataSet(config, if_training=True)
        self.train_loader = DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    
    def train(self):
        self.modeltrain()
        for epoch in range(config.train_epochs):
            bar = tqdm(iter(self.train_loader), ascii=True)
            for imgs, labels in bar:
                imgs = imgs.cuda()
                labels = labels.cuda()
                print(labels)
                




if __name__ == "__main__":
    tainer = Trainer()
    tainer.train()

    
    