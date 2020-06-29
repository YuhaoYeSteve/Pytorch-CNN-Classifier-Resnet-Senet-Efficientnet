class Config:
    def __init__(self, sku_list, model_id):
        # 试验台方向
        # # -------------------------------   超参  ----------------------------------#
        # self.model_name = "efficientnet-b0"
        # # self.model_name = "resnet50"
        # self.sku_list = sku_list
        # self.class_num = len(self.sku_list)
        # self.model_id = model_id
        # self.if_test = True

        # if self.if_test:
        #     self.train_epoch = 3
        #     self.start_eval_epoch = 1  # 大于等于start_eval_epoch才开始在全部训练集上进行训练集的精度测试
        #     self.model_ready_train_acc = 0.99
        #     self.model_ready_epoch = 3  # 大于等于model_ready_epoch并且满足大于等于model_ready_train_acc, 即可停止训练
        # else:
        #     if self.class_num <= 5:
        #         self.train_epoch = 35
        #     elif 5 < self.class_num <= 10:
        #         self.train_epoch = 25
        #     elif 10 < self.class_num <= 20:
        #         self.train_epoch = 20
        #     elif 20 < self.class_num <= 40:
        #         self.train_epoch = 15
        #     elif self.class_num > 40:
        #         self.train_epoch = 15
        #     self.start_eval_epoch = 5  # 大于等于start_eval_epoch才开始在全部训练集上进行训练集的精度测试
        #     self.model_ready_train_acc = 0.996
        #     self.model_ready_epoch = 5  # 大于等于model_ready_epoch并且满足大于等于model_ready_train_acc, 即可停止训练

        # self.best_acc = 0.0
        # self.task_name = "lab_direction"
        # self.batch_size = 256
        # self.input_size = 64  # self.input_size = EfficientNet.get_image_size(self.model_name)
        # self.mean = [0.408, 0.447, 0.47]
        # self.std = [0.289, 0.274, 0.278]
        # self.mapfile_and_trt_model_out_root = "./train_out/"  # mapfile: 网络输出类别和实际数据类别名称的对照的json文件
        # # self.data_and_anno_root = "/data/yyh/Orcas/fenlei/lab_direction/"
        # self.data_and_anno_root = "/data/yyh/classify/all_train_data/"
        # self.mix_up_alpha = 0.2
        # self.print_loss_interval = 100
        # self.print_loss_remainder = 99
        # if self.if_test:
        #     self.dataLoader_num_worker = 32
        # else:
        #     self.dataLoader_num_worker = 32
        # self.class_weight = []
        # self.dataset_specified_pretrain_model_path = ""

        # # -------------------------------   开关  ----------------------------------#
        # # mix up的 开关
        # self.use_mix_up = True

        # # label_smoothing loss的 开关
        # self.use_label_smoothing = True

        # # 加载针对数据集的预训练模型的 开关
        # self.load_dataset_specified_pre_train = False

        # # 使用Apex混合精度训练的 开关
        # self.use_apex_amp_mix_precision = False

        # # cudnn加速的 开关
        # self.use_cudnn_accelerate = True

        # # 设置随机种子的 开关
        # self.set_seed = True

        # # 单机多卡的 开关
        # self.use_multi_gpu = False

        # # 是否有验证集的 开关
        # self.if_has_val_data = False

        # # -------------------------------   选择  ----------------------------------#

        # # 优化器的 选择
        # self.which_optimizer = "adam"  # "adam" or "sgd"
        # # 保存模型策略的 选择
        # self.save_strategy = "best"  # "best" or "last"
        # # 显卡的 选择
        # if self.use_multi_gpu:
        #     self.gpu_num = [1, 2]  # 多卡
        # else:
        #     self.gpu_num = "7"  # 单卡
        # self.base_lr = 0.001  # 基础学习率
        # self.lr_schedule = {  # 学习率调整策略
        #     8: self.base_lr * 0.5,
        #     12: self.base_lr * 0.3,
        #     15: self.base_lr * 0.1,
        #     18: self.base_lr * 0.01,
        # }



        # #熊猫公交年龄
        # # -------------------------------   超参  ----------------------------------#
        # self.model_name = "efficientnet-b0"
        # # self.model_name = "resnet50"
        # self.sku_list = sku_list
        # self.class_num = len(self.sku_list)
        # self.model_id = model_id
        # self.if_test = True

        # if self.if_test:
        #     self.train_epoch = 20
        #     self.start_eval_epoch = 1  # 大于等于start_eval_epoch才开始在全部训练集上进行训练集的精度测试
        #     self.model_ready_train_acc = 0.99
        #     self.model_ready_epoch = 10  # 大于等于model_ready_epoch并且满足大于等于model_ready_train_acc, 即可停止训练
        # else:
        #     if self.class_num <= 5:
        #         self.train_epoch = 35
        #     elif 5 < self.class_num <= 10:
        #         self.train_epoch = 25
        #     elif 10 < self.class_num <= 20:
        #         self.train_epoch = 20
        #     elif 20 < self.class_num <= 40:
        #         self.train_epoch = 15
        #     elif self.class_num > 40:
        #         self.train_epoch = 15
        #     self.start_eval_epoch = 5  # 大于等于start_eval_epoch才开始在全部训练集上进行训练集的精度测试
        #     self.model_ready_train_acc = 0.996
        #     self.model_ready_epoch = 5  # 大于等于model_ready_epoch并且满足大于等于model_ready_train_acc, 即可停止训练

        # self.best_acc = 0.0
        # self.task_name = "lab_direction"
        # self.batch_size = 256
        # self.input_size = 64  # self.input_size = EfficientNet.get_image_size(self.model_name)
        # self.mean = [0.408, 0.447, 0.47]
        # self.std = [0.289, 0.274, 0.278]
        # self.mapfile_and_trt_model_out_root = "./train_out/"  # mapfile: 网络输出类别和实际数据类别名称的对照的json文件
        # # self.data_and_anno_root = "/data/yyh/Orcas/fenlei/lab_direction/"
        # self.data_and_anno_root = "/data/yyh/classify/data/age/train"
        # self.mix_up_alpha = 0.2
        # self.print_loss_interval = 100
        # self.print_loss_remainder = 99
        # if self.if_test:
        #     self.dataLoader_num_worker = 32
        # else:
        #     self.dataLoader_num_worker = 32
        # self.class_weight = []
        # self.dataset_specified_pretrain_model_path = ""

        # # -------------------------------   开关  ----------------------------------#
        # # mix up的 开关
        # self.use_mix_up = True

        # # label_smoothing loss的 开关
        # self.use_label_smoothing = True

        # # 加载针对数据集的预训练模型的 开关
        # self.load_dataset_specified_pre_train = False

        # # 使用Apex混合精度训练的 开关
        # self.use_apex_amp_mix_precision = False

        # # cudnn加速的 开关
        # self.use_cudnn_accelerate = False

        # # 设置随机种子的 开关
        # self.set_seed = True

        # # 单机多卡的 开关
        # self.use_multi_gpu = False

        # # 是否有验证集的 开关
        # self.if_has_val_data = False

        # # -------------------------------   选择  ----------------------------------#

        # # 优化器的 选择
        # self.which_optimizer = "adam"  # "adam" or "sgd"
        # # 保存模型策略的 选择
        # self.save_strategy = "best"  # "best" or "last"
        # # 显卡的 选择
        # if self.use_multi_gpu:
        #     self.gpu_num = [1, 2]  # 多卡
        # else:
        #     self.gpu_num = "1"  # 单卡
        # self.base_lr = 0.001  # 基础学习率
        # self.lr_schedule = {  # 学习率调整策略
        #     8: self.base_lr * 0.5,
        #     12: self.base_lr * 0.3,
        #     15: self.base_lr * 0.1,
        #     18: self.base_lr * 0.01,
        # }



        #Cifar-10
        # -------------------------------   超参  ----------------------------------#
        self.model_name = "efficientnet-b0"
        # self.model_name = "resnet50"
        self.sku_list = sku_list
        self.class_num = len(self.sku_list)
        self.model_id = model_id
        self.if_test = True

        if self.if_test:
            self.train_epoch = 100
            self.start_eval_epoch = 1  # 大于等于start_eval_epoch才开始在全部训练集上进行训练集的精度测试
            self.model_ready_train_acc = 0.999
            self.model_ready_epoch = 10  # 大于等于model_ready_epoch并且满足大于等于model_ready_train_acc, 即可停止训练
        else:
            if self.class_num <= 5:
                self.train_epoch = 35
            elif 5 < self.class_num <= 10:
                self.train_epoch = 25
            elif 10 < self.class_num <= 20:
                self.train_epoch = 20
            elif 20 < self.class_num <= 40:
                self.train_epoch = 15
            elif self.class_num > 40:
                self.train_epoch = 15
            self.start_eval_epoch = 5  # 大于等于start_eval_epoch才开始在全部训练集上进行训练集的精度测试
            self.model_ready_train_acc = 0.996
            self.model_ready_epoch = 5  # 大于等于model_ready_epoch并且满足大于等于model_ready_train_acc, 即可停止训练

        self.best_acc = 0.0
        self.task_name = "cifar10"
        self.batch_size = 256
        self.input_size = 224  # self.input_size = EfficientNet.get_image_size(self.model_name)
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        self.mapfile_and_trt_model_out_root = "./train_out/"  # mapfile: 网络输出类别和实际数据类别名称的对照的json文件
        # self.data_and_anno_root = "/data/yyh/Orcas/fenlei/lab_direction/"
        self.data_and_anno_root = "/data/yyh/classify/data/age/train"
        self.mix_up_alpha = 0.2
        self.print_loss_interval = 100
        self.print_loss_remainder = 99
        if self.if_test:
            self.dataLoader_num_worker = 32
        else:
            self.dataLoader_num_worker = 32
        self.class_weight = []
        self.dataset_specified_pretrain_model_path = ""

        # -------------------------------   开关  ----------------------------------#
        # mix up的 开关
        self.use_mix_up = False

        # label_smoothing loss的 开关
        self.use_label_smoothing = False

        # 加载针对数据集的预训练模型的 开关
        self.load_dataset_specified_pre_train = False

        # 使用Apex混合精度训练的 开关
        self.use_apex_amp_mix_precision = False

        # cudnn加速的 开关
        self.use_cudnn_accelerate = False

        # 设置随机种子的 开关
        self.set_seed = True

        # 单机多卡的 开关
        self.use_multi_gpu = True

        # 是否有验证集的 开关
        self.if_has_val_data = False

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
        self.base_lr = 0.001  # 基础学习率
        self.lr_schedule = {  # 学习率调整策略
            10: self.base_lr * 0.1,
            20: self.base_lr * 0.01,
            # 2: self.base_lr * 0.5,
            # 12: self.base_lr * 0.3,
            # 15: self.base_lr * 0.1,
            # 25: self.base_lr * 0.01,
            # 35: self.base_lr * 0.05,
            # 70: self.base_lr * 0.001,
        }