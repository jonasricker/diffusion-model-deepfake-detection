# MODIFIED: Extract feature space representation at last layer of detector.

# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

import argparse
import glob
import os

import numpy as np
from PIL import Image
from resnet50nodown_features import resnet50nodown

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This script extracts features representation of images in an image folder and saves these representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights-path", help="Root of model directory.")
    parser.add_argument("--img-root", help="Root of image directory.")
    parser.add_argument("--output-root", help="Output directory.")
    config = parser.parse_args()
    weights_path = config.weights_path
    img_root = config.img_root
    out_path = config.output_root

    detector_type = weights_path.rsplit("_")[-1][:-4]  # extract detector type
    save_dir = f"{out_path}/features/gragnaniello2021/{detector_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    from torch.cuda import is_available as is_available_cuda

    device = "cuda:0" if is_available_cuda() else "cpu"
    net = resnet50nodown(device, weights_path)

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

    for model in all_models:
        input_folder = f"{img_root}/{model}"

        list_files = sorted(
            sum(
                [glob.glob(os.path.join(input_folder, "*." + x)) for x in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]],
                list(),
            )
        )
        num_files = len(list_files)

        print(f"Feature Extraction for {model}")
        print("Start...")

        # Extract feature representations for this model
        x_inter_list = []
        for index, filename in enumerate(list_files):
            print("%5d/%d" % (index, num_files), end="\r")
            img = Image.open(filename).convert("RGB")
            img.load()
            logit, x_inter = net.apply(img)
            x_inter_list.append(x_inter)

        x_inter = np.vstack(x_inter_list)

        np.save(f"{save_dir}/features_{model}", x_inter)

        print("\nDone.")
