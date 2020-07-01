class Config:
    def __init__(self):

        # ---------------------------   Hyper-Parameter  ------------------------------#
        
        self.model_name = "efficientnet-b0" # "resnet50"/ SeNet / "efficientnet-b0"
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

        # 单机多卡的 开关
        self.use_multi_gpu = True

        # 是否有验证集的 开关
        self.if_has_val_data = False

        self.if_warmup = False

        # -------------------------------   选择  ----------------------------------#

        # 优化器的 选择
        self.which_optimizer = "adam"  # "adam" or "sgd"
        # 保存模型策略的 选择
        self.save_strategy = "best"  # "best" or "last"
        # 显卡的 选择
        if self.use_multi_gpu:
            # self.gpu_num = [3, 4, 5, 6]  # 多卡
            # self.gpu_num = [4, 5, 6, 7]  # 多卡
            self.gpu_num = [1, 2, 3]  # 多卡
        else:
            self.gpu_num = "1"  # 单卡

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
        
if __name__ == "__main__":
    config = TaskConfig()
    print(config.lr_schedule)