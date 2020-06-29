# Pytorch-CNN-Classify-Resnet-Senet-Efficientnet
Customized Image Classifier based on Pytorch with visdom visualization Support customized dataset, augmentation and SOTA CNN(Resnet, Senet, EfficientNet....))

## Highlights

- **Automatic Mixed Precision Training:** Support FP16 training based on NVIDIA-Apex(https://github.com/NVIDIA/apex) which can help you training with 2x batch size as well as speed up training time.

- **Multi-GPU Training:** Support single server multi-GPU training based on Pytorch nn.DataParallel module.

- **ONNX And TensorRT Transfer Included:** Support transfer from trained *.pth* model to ONNX model which will be transfered to TensorRT *.trt* model; Support C++ inference code.





## References

1. lukemelas, [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch): https://github.com/lukemelas/EfficientNet-PyTorch

2. Mingxing Tan, Quoc V. Le EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks：https://arxiv.org/pdf/1905.11946.pdf



