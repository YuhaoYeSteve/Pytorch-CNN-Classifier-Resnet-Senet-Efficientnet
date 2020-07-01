from torch.utils import data
import os
from utils.img_utils import read_image


class DataSet(data.Dataset):
    def __init__(self, data_root, if_training=True):
        super(DataSet, self).__init__()
        self.is_training = if_training
        self.img_path_list, self.label_list = self.get_data_list(data_root)


    # Generate image and label list
    def get_data_list(self, data_root):
        for class_name in os.listdir(data_root):
            print(class_name)
        return balanced_img_list, balanced_label_list



    def __getitem__(self, item):
            return img, label
    def __len__(self):
        return len(self.img_path_list)
