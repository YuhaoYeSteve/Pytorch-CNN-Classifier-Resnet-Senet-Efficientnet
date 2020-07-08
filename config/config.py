import albumentations as A
import cv2


class Config:
    def __init__(self):

        # ---------------------------   Hyper-Parameter  ------------------------------#

        self.model_name = "efficientnet-b0"  # "resnet50"/ SeNet / "efficientnet-b0"
        self.train_epoch = 100

        self.best_acc = 0.0
        self.task_name = "cifar10"
        self.batch_size = 64
        self.input_size = 224
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        self.model_and_training_info_save_root = "./train_out/"  #
        self.mix_up_alpha = 0.2
        self.print_loss_interval = 100
        self.print_loss_remainder = 99
        self.dataLoader_num_worker = 32
        self.class_weight = []
        self.pretrain_model_path = ""

        # -------------------------------   Switch  ----------------------------------#
        # if use visdom
        self.use_visdom = True

        # if use mix up
        self.use_mix_up = False

        # if use label_smoothing on loss
        self.use_label_smoothing = False

        # if use class specified pre-trained model
        self.load_dataset_specified_pre_train = False

        # if use Apex FP16 training
        self.use_apex_amp_mix_precision = True

        # if use cudnn Accleration
        self.use_cudnn_accelerate = True

        # if use random seed(better for doing experiment)
        self.set_seed = True

        # if use multiple GPU
        self.use_multi_gpu = True

        # if use warm-up leanring rate
        self.if_warmup = False

        # -------------------------------   Choice  ----------------------------------#

        # Choose optimizer
        self.which_optimizer = "adam"  # "adam" or "sgd"

        # Choose GPU number
        if self.use_multi_gpu:
            # self.gpu_num = [3, 4, 5, 6]  # Multiple GPU
            # self.gpu_num = [4, 5, 6, 7]  # Multiple GPU
            self.gpu_num = [1, 2, 3]  # Multiple GPU
        else:
            self.gpu_num = "1"  # Single GPU

        if self.if_warmup:
            # warmup
            self.base_lr = 0.0001  # 基础学习率
            self.warmup_lr = 0.00001
            self.lr_schedule = {  # 学习率调整策略
                # 2: self.base_lr,
                # # 5: self.base_lr,
                10: self.base_lr * 0.1,
                50: self.base_lr * 0.01,
                # 2: self.base_lr * 0.5,
                # 12: self.base_lr * 0.3,
                # 15: self.base_lr * 0.1,
                # 25: self.base_lr * 0.01,
                # 35: self.base_lr * 0.05,
                # 70: self.base_lr * 0.001,
            }
        else:
            self.warmup_lr = 0.0001
            self.lr_schedule = {  # 学习率调整策略
                20: self.warmup_lr * 0.1,
                50: self.warmup_lr * 0.01,
                80: self.warmup_lr * 0.001
            }

# 0.01第二轮就飞了，test_acc=0.1
# 0.0001 第一轮93, 第二轮95, 第三轮96


class TaskConfig(Config):
    def __init__(self):
        super(TaskConfig, self).__init__()
        self.train_data_root = "./dataset/cifar10/train"
        self.val_data_root = "./dataset/cifar10/val"
        self.training_name = "cifar10_efficientnet_b0_224_224"
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
        ])


if __name__ == "__main__":
    config = TaskConfig()
    print(config.training_name)


# Albumentations Memo

# A.Rotate(limit=(-90,-90), p=1, border_mode=cv2.BORDER_REFLECT_101)
# limit里面是旋转的角度, 正数代表逆时针旋转, 负数是正时针
# border_mode是旋转后的填充方式：cv2.BORDER_CONSTANT 会用黑色填充空出来的部分但是不会缩小
#                                cv2.BORDER_REPLICATE 会用复制边界像素的方式去填充空白处, 但是不会缩小
#                                cv2.BORDER_WRAP 会用原本的图像去填充空白处
#                                cv2.BORDER_REFLECT_101 会用原本的镜像去填充空白处
