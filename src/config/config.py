import albumentations as A
import cv2
import os
from utils.general_utils import load_yaml, get_time, check_path_with_delete
from utils.logctrl import getLogger
from utils.init_visdom import init_visdom_


class Config:
    def __init__(self):

        # -------------------------------   Switch  ----------------------------------#
        # if use visdom
        self.use_visdom = True

        # if use mix up
        self.use_mix_up = False

        # if use label_smoothing on loss
        self.use_label_smoothing = False

        # if use dataset specified pre-trained model
        self.load_dataset_specified_pre_train = False

        # if use Apex FP16 training
        self.use_apex_amp_mix_precision = False

        # if use cudnn Accleration
        self.use_cudnn_accelerate = False

        # if use random seed(better for doing experiment)
        self.set_seed = True

        # if use multiple GPU
        self.use_multi_gpu = True

        # Show training accuracy
        self.if_show_training_acc = True

        # self.if_debug = True
        self.if_debug = False
        # -------------------------------   Choice  ----------------------------------#
        # Choose optimizer
        self.which_optimizer = "adam"  # "adam" or "sgd"

        # Choose GPU number
        if self.use_multi_gpu:
            # self.gpu_num = [3, 4, 5, 6]  # Multiple GPU
            # self.gpu_num = [4, 5, 6, 7]  # Multiple GPU
            self.gpu_num = [2, 3, 4, 5]  # Multiple GPU
        else:
            self.gpu_num = "1"  # Single GPU

        self.base_lr = 0.0001
        self.lr_schedule = {          # learning rate strategy
            10: self.base_lr * 0.5,
            20: self.base_lr * 0.3,
            25: self.base_lr * 0.1,
            45: self.base_lr * 0.01,
        }

        # ---------------------------   Hyper-Parameter  ------------------------------#
        self.model_name = "efficientnet-b0"  # "resnet50"/ SeNet / "efficientnet-b0"
        self.train_epochs = 50
        self.best_acc = 0.0
        self.batch_size = len(self.gpu_num) * 32
        self.input_size = 224
        if "efficientnet" in self.model_name:
            self.mean = [0.408, 0.447, 0.47]
            self.std = [0.289, 0.274, 0.278]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        self.model_and_training_info_save_root = "./train_out/"
        self.mix_up_alpha = 0.2
        self.print_loss_interval = 100
        self.print_loss_remainder = 99
        self.dataLoader_num_worker = min(8 * len(self.gpu_num), 32)
        if self.if_debug:
            self.dataLoader_num_worker = 0
            self.batch_size = 1
        self.class_weight = []
        self.train_time = get_time()
        # --------------------------------   Path   ----------------------------------#
        self.pretrain_model_path = ""
        self.log_and_model_root = "./train_out/"


class TaskConfig(Config):
    def __init__(self):
        super(TaskConfig, self).__init__()
        self.task_config_path = "./task_config.yaml"
        self.task_config = self.get_task_config()
        self.input_size = self.task_config["train_size"]
        self.train_data_root = os.path.join(
            self.task_config["dataset_root_path"], "train")
        self.val_data_root = os.path.join(
            self.task_config["dataset_root_path"], "val")
        self.task_name = self.get_task_name()
        self.class_num = len(self.task_name)
        self.log_and_model_path = os.path.join(
            self.log_and_model_root, self.task_name, self.train_time)
        check_path_with_delete(self.log_and_model_path)
        self.logger = getLogger(__name__, os.path.join(
            self.log_and_model_path, "logs/all.log"))
        self.vis = init_visdom_(window_name=self.task_name)
        self.class_name_list = self.get_class_name()
        self.class_id_list = [str(_) for _ in range(len(self.class_name_list))]
        self.transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Resize(height=self.input_size, width=self.input_size, p=1)
        ])

    def get_task_config(self):
        return load_yaml(self.task_config_path)

    def get_task_name(self):
        return self.task_config["dataset_root_path"].split("/")[-1]

    def get_class_name(self):
        return [class_name for class_name in os.listdir(self.train_data_root)]


if __name__ == "__main__":
    config = TaskConfig()
    print(config.training_name)
