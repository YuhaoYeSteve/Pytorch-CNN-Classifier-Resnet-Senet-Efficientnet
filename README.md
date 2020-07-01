# Pytorch-CNN-Classify-Resnet-Senet-Efficientnet
Customized Image Classifier based on Pytorch with visdom visualization Support customized dataset, augmentation and SOTA CNN(Resnet, Senet, EfficientNet....))

## Highlights

- **Automatic Mixed Precision Training:** Support FP16 training based on [NVIDIA-Apex](https://github.com/NVIDIA/apex) which can help you training with 2x batch size as well as speed up training time.

- **Multi-GPU Training:** Support single server multi-GPU training based on Pytorch nn.DataParallel module.

- **Training Process Visualization:** Support visualize augmentation result and prediction result in browser based on [visdom]().

- **[ONNX](https://github.com/onnx/onnx) And [TensorRT](https://github.com/NVIDIA/TensorRT) Transfer Included:** Support transfer from trained *.pth* model to ONNX model which will be transfered to TensorRT *.trt* model; Support C++ inference code.

## Training

### 1. Prepare your dataset

    # your dataset structure should be like this
    ./dataset/
        -your_project_name/
            -train/
                -0 (class from 0 to num of class of your dataset)
                   -*.jpg 
                -1
                   -*.jpg
                -2
                   -*.jpg
                -3
                   -*.jpg
                - .....
                   -*.jpg
            -val/
                -0
                   -*.jpg 
                -1
                   -*.jpg
                -2
                   -*.jpg
                -3
                   -*.jpg
                - .....
                   -*.jpg
            
    
    # for example: cifar10(unziped)
    ./dataset/
        -cifar10/
            -train/
                -0 (cifar10 has 10 class in total 0-9) 
                   -000000000001.jpg
                   -000000000002.jpg
                   -000000000003.jpg
                   .... 

                - 1-8

                -9
                   -000000000001.jpg
                   -000000000002.jpg
                   -000000000003.jpg
                   .... 
            -val/
                -0 
                   -000000000001.jpg
                   -000000000002.jpg
                   -000000000003.jpg
                   .... 

                - 1-8

                -9
                   -000000000001.jpg
                   -000000000002.jpg
                   -000000000003.jpg
                   .... 
            




## References

1. lukemelas, EfficientNet-PyTorch: https://github.com/lukemelas/EfficientNet-PyTorch

2. Mingxing Tan, Quoc V. Le EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks：https://arxiv.org/pdf/1905.11946.pdf



