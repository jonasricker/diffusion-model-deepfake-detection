# MODIFIED to extract feature representations.

import argparse
import csv
import glob
import os

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
)
from tqdm import tqdm

from networks.resnet_features import resnet50


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dir", type=str, default="data")
parser.add_argument("-m", "--model_path", type=str, default="weights/blur_jpg_prob0.5.pth")
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-j", "--workers", type=int, default=4, help="number of workers")
parser.add_argument("-c", "--crop", type=int, default=None, help="by default, do not crop. specify crop size")
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--size_only", action="store_true", help="only look at sizes of images in dataset")
parser.add_argument("--output_root", help="Output directory")

opt = parser.parse_args()

torch.cuda.set_device(0)

# Load model
if not opt.size_only:
    net = resnet50(num_classes=1)
    if opt.model_path is not None:
        if os.path.isdir(opt.model_path):
            state_dict = torch.load(os.path.join(opt.model_path, "model_epoch_best.pth"), map_location="cpu")
        else:
            state_dict = torch.load(opt.model_path, map_location="cpu")
    net.load_state_dict(state_dict["model"])
    net.eval()
    if not opt.use_cpu:
        net.cuda()

# Transform
trans_init = []
if opt.crop is not None:
    trans_init = [
        transforms.CenterCrop(opt.crop),
    ]
    print("Cropping to [%i]" % opt.crop)
else:
    print("Not cropping")
trans = transforms.Compose(
    trans_init
    + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if os.path.isdir(opt.model_path):
    model_type = "_".join(opt.model_path.split("/")[-2:])
else:
    model_type = opt.model_path.split("/")[-1][:-4]
save_dir = f"{opt.output_root}/features/wang2020/{model_type}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# loop over all datasets
all_models = [
    "Real",
    "ADM",
    "DDPM",
    "IDDPM",
    "LDM",
    "PNDM",
    "Diff-StyleGAN2",
    "Diff-ProjectedGAN",
    "ProGAN",
    "ProjectedGAN",
    "StyleGAN",
]

for model in tqdm(all_models):
    input_folder = f"{opt.dir}/{model}"
    print(f"Loading data set in {input_folder}")

    list_files = sorted(
        sum(
            [glob.glob(os.path.join(input_folder, "*." + x)) for x in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]],
            list(),
        )
    )
    num_files = len(list_files)

    x_inter_list = []

    with torch.no_grad():
        for filename in tqdm(list_files):
            # load the image
            img = Image.open(filename).convert("RGB")
            img.load()
            # apply predefined transforms
            data = trans(img)

            # unsqueeze to mirror batch size of 1 (to match required size)
            data = torch.unsqueeze(data, dim=0)

            if not opt.size_only:
                if not opt.use_cpu:
                    data = data.cuda()
                y_pred, x_inter = net(data)
                # print(x_inter.shape)
                x_inter_list.append(x_inter.cpu().numpy())

    x_inter = np.vstack(x_inter_list)
    np.save(f"{save_dir}/features_{model}", x_inter)
    print("\nDone.")
