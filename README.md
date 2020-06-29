# Pytorch-CNN-Classify-Resnet-Senet-Efficientnet
Customized Image Classifier based on Pytorch with visdom visualization Support customized dataset, augmentation and SOTA CNN(Resnet, Senet, EfficientNet....))

## Highlights

- **Automatic Mixed Precision Training:** Support.

- **Multi-GPU Training:** The same framework works for object detection, 3d bounding box estimation, and multi-person pose estimation with minor modification.

- **Fast:** The whole process in a single network feedforward. No NMS post processing is needed. Our DLA-34 model runs at *52* FPS with *37.4* COCO AP.

- **Strong**: Our best single model achieves *45.1*AP on COCO test-dev.

- **Easy to use:** We provide user friendly testing API and webcam demos.



## References

1. lukemelas, EfficientNet-PyTorch: https://github.com/lukemelas/EfficientNet-PyTorch

2. Mingxing Tan, Quoc V. Le EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks：https://arxiv.org/pdf/1905.11946.pdf



