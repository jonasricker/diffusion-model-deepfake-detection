import cv2
import numpy as np
from PIL import Image


class Perturbation:
    def __init__(self, func, p=0.5):
        self.func = func
        self.p = p

    def __call__(self, x):
        x = np.asarray(x)
        x = self.func(x)
        return Image.fromarray(x)

    def __repr__(self):
        return self.func.__name__.capitalize()


class CombinedPerturbation:
    def __init__(self, funcs):
        self.funcs = funcs
        self.i = 0

    def __call__(self, x):
        x = np.asarray(x)
        x = self.funcs[self.i](x)
        self.i = (self.i + 1) % len(self.funcs)
        return Image.fromarray(x)


# perturbations adapted from https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/create_perturbed_imagedata.py


def noise(image):
    # variance from U[5.0,20.0]
    variance = np.random.uniform(low=5.0, high=20.0)
    image = np.copy(image).astype(np.float64)
    noise = variance * np.random.randn(*image.shape)
    image += noise
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def blur(image):
    # kernel size from [1, 3, 5, 7, 9]
    kernel_size = np.random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=cv2.BORDER_DEFAULT
    )


def jpeg(image):
    # quality factor sampled from U[10, 75]
    factor = np.random.randint(low=10, high=75)
    _, image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), factor])
    return cv2.imdecode(image, 1)


def cropping(image):
    # crop between 5% and 20%
    percentage = np.random.uniform(low=0.05, high=0.2)
    x, y, _ = image.shape
    x_crop = int(x * percentage * 0.5)
    y_crop = int(y * percentage * 0.5)
    cropped = image[x_crop:-x_crop, y_crop:-y_crop]
    resized = cv2.resize(cropped, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    return resized
