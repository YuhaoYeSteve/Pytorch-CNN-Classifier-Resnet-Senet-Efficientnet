import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
# from my_resnet import resnet50, resnet50_with_softmax
import os
import onnx_to_trt as transform_trt
from efficientnet_pytorch import EfficientNet

if_muti_gpu_training = True

# pytorch_model_name = "2020-03-31_10-55-35_size224_last_epoch9_batch16_resnet_50_original_pretrain_no_label_smooting_mix_up_testacc1.0_loss0.56448_big_blue_six_direction_test.pth"
pytorch_model_name = "lab_direction_efficientnet_b0_64_64.pth"
fp32_fp16_int8 = "fp32"


# *********************************************** Pytorch Info*********************************************** #
pytorch_model_root = "./pytorch_models/"
pytorch_model_path = os.path.join(pytorch_model_root, pytorch_model_name)

# *********************************************** Pytorch Info*********************************************** #
out_onnx_model_root = "./onnx_models/"
out_onnx_model_name = pytorch_model_name[:-4] + ".onnx"
out_onnx_model_path = os.path.join(out_onnx_model_root, out_onnx_model_name)

# ************************************************* Trt Info   *********************************************** #
out_trt_model_root = "./trt_models/"
out_trt_model_name = pytorch_model_name[:-4] + ".fp32.trtmodel"
trt_model_out_path = os.path.join(out_trt_model_root, out_trt_model_name)
with open("./trt_path.txt", "w") as f:
    f.write(trt_model_out_path)



# **********************************************Pytorch To Onnx*********************************************** #
print("#"*60)
print("Load Pytorch Model: ", pytorch_model_path)
net = torch.load(pytorch_model_path)
# net = resnet50_with_softmax(pytorch_model_path, if_muti_gpu_training=if_muti_gpu_training)
net = net.cuda()
net.eval()
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
torch.onnx.export(net, dummy_input, out_onnx_model_path, verbose=True)
print("Finish Transfering from pytorch pth to onnx model!!!!!")
print("#"*60)

if not os.path.exists(out_onnx_model_path):
    raise Exception("Can not find trasnfered onnx model, Pytorch To Onnx!!!!!!!")

# **********************************************Onnx To TensorRT*********************************************** #
transform_trt_info = []


transform_trt_info.append(out_onnx_model_path)
transform_trt_info.append(trt_model_out_path)
results = transform_trt.onnx_to_trt(transform_trt_info)
print("results: ", results)

if not os.path.exists(trt_model_out_path):
    raise Exception("Can not find trasnfered trt model, Onnx To TensorRT!!!!!!!")

