from torchvision import transforms
import matplotlib.pyplot as plt
import os
# from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
from PIL import Image, ImageOps, ImageEnhance
import torch
print(torch.__version__)
import sys
print(sys.version)
print()
# try:
#     import accimage
# except ImportError:
#     accimage = None
import numpy as np
import torch
import collections
import sys
import cv2
import math
import random


def pil_to_cv2(PIL_image):
    CV2_image = cv2.cvtColor(np.asarray(PIL_image), cv2.COLOR_RGB2BGR)
    return CV2_image


def pil_to_cv2_without_cvt(PIL_image):
    CV2_image = np.asarray(PIL_image)
    return CV2_image


def cv2_to_pil_without_cvt(CV2_image):
    PIL_image = Image.fromarray(CV2_image)
    return PIL_image


def cv2_to_pil(CV2_image):
    PIL_image = Image.fromarray(cv2.cvtColor(CV2_image, cv2.COLOR_BGR2RGB))
    return PIL_image


def tensor_to_cv2(pytorch_Tensor):
    # CV2_image = cv2.cvtColor(pytorch_Tensor.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    PIL_image = transforms.ToPILImage()(pytorch_Tensor)
    CV2_image = pil_to_cv2(PIL_image)
    return CV2_image


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_pil_image(img):
    # if accimage is not None:
    #     return isinstance(img, (Image.Image, accimage.Image))
    # else:
    #     return isinstance(img, Image.Image)\
    return isinstance(img, Image.Image)


# 随机Crop
def RandomCrop(pil_img):
    if not _is_pil_image(pil_img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(pil_img)))

    origin_width, origin_height = pil_img.size
    crop_x_center = random.randint(0, origin_width)
    crop_y_center = random.randint(0, origin_height)
    max_crop_width = origin_width - crop_x_center
    max_crop_height = origin_height - crop_y_center
    crop_width = random.randint(int(max_crop_width / 3), max_crop_width)
    crop_height = random.randint(int(max_crop_height / 3), max_crop_height)

    return pil_img.crop((crop_x_center, crop_y_center, crop_x_center + crop_width, crop_y_center + crop_height))


def zuoyou_flip(img):
    """Horizontally flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def shangxia_flip(img):
    """Vertically flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)


def rotate(angle, img, resample=False, expand=True, center=None):
    """Rotate the image by angle.


    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.rotate(angle, resample, expand, center)


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor


def adjust_brightness(brightness_factor, img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not (0.5 <= brightness_factor <= 1.5):
        raise ValueError('brightness_factor is not in [0.5, 1.5].'.format(brightness_factor))
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(contrast_factor, img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not (0.5 <= contrast_factor <= 3.5):
        raise ValueError('brightness_factor is not in [0.5, 1.5].'.format(contrast_factor))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(saturation_factor, img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not (0.5 <= saturation_factor <= 1.5):
        raise ValueError('brightness_factor is not in [0.5, 1.5].'.format(saturation_factor))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(hue_factor, img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img
    h, s, v = img.convert('HSV').split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def gaussian_noise(PIL_img):
    """添加高斯噪声"""
    image = pil_to_cv2(PIL_img)
    h, w, c = image.shape
    noise = np.random.normal(0, 13, (h, w))
    image[:, :, 0] = np.clip((image[:, :, 0] + noise), 0, 255)
    image[:, :, 1] = np.clip((image[:, :, 1] + noise), 0, 255)
    image[:, :, 2] = np.clip((image[:, :, 2] + noise), 0, 255)
    image = cv2_to_pil(image)
    return image


# https://www.cnblogs.com/arkenstone/p/8480759.html
def motion_blur(image, degree=5, angle=5):
    image = pil_to_cv2(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    blurred = cv2_to_pil(blurred)
    return blurred


# https://blog.csdn.net/keith_bb/article/details/54412493
def blur(pil_img):
    image = pil_to_cv2(pil_img)
    gaussian_kernel_size = random.randint(1, 3)
    gaussian_kernel_size = gaussian_kernel_size * 2 + 1
    image = cv2.GaussianBlur(image, ksize=(gaussian_kernel_size, gaussian_kernel_size), sigmaX=0, sigmaY=0)
    image = cv2_to_pil(image)
    return image


# 随机擦除
def random_erasing(im, prob=0.2, sl=0.01, sh=0.03, r1=0.5, mean=[127.5, 127.5, 127.5]):
    if random.uniform(0, 1) > prob:
        return im
    im = pil_to_cv2(im)
    img = im.copy()

    for attempt in range(100):

        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[0] and h < img.shape[1]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                img[x1:x1 + h, y1:y1 + w, 2] = mean[2]
            else:
                img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
            img = cv2_to_pil(img)
            return img


# 透视变换
def perspective_transfor(PIL_img, max=0.3):
    img = pil_to_cv2(PIL_img)
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    pts2 = np.float32([[cols * random.uniform(0., max), rows * random.uniform(0., max)],
                       [cols * (1 - random.uniform(0., max)), rows * random.uniform(0., max)],
                       [cols * random.uniform(0., max), rows * (1 - random.uniform(0., max))],
                       [cols * (1 - random.uniform(0., max)), rows * (1 - random.uniform(0., max))]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    PIL_img = cv2_to_pil(dst)

    return PIL_img
