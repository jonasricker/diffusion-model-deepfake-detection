# Script to load intermediate representations of the data
# and calculate MMD between representations of real and model-generated images

# MMD implementation adapted from https://github.com/TorchDrift/TorchDrift/blob/master/notebooks/note_on_mmd.ipynb

import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


def MMD(x, y, sigma):
    """
    Maximum Mean Discrepancy with Gaussian kernel (RBF).
    Compare kernel MMD paper and code:
    A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    """
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp(
        (-1 / (2 * sigma**2)) * dists**2
    )  # + torch.eye(n+m).to(device) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = (
        k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    )
    return mmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This script loads pre-computed representations of images of DMs and GANs, "
            "                   and computes MMD between real and fake for all given"
            " models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_input_folder",
        "-b",
        type=str,
        default="./features",
        help="base input folder with subfolders containing feature representations",
    )

    config = parser.parse_args()
    base_input_folder = config.base_input_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dirs = []
    for dir in os.listdir(base_input_folder):
        if os.path.exists(dir + "features_Real.npy"):
            features_dirs.append(base_input_folder + "/" + dir)
        else:
            for subdir in os.listdir(base_input_folder + "/" + dir):
                if os.path.exists(
                    base_input_folder + "/" + dir + "/" + subdir + "/features_Real.npy"
                ):
                    features_dirs.append(base_input_folder + "/" + dir + "/" + subdir)
    print(
        f"Found {len(features_dirs)} features directories in root directory:\n",
        features_dirs,
    )

    for feature_dir in tqdm(features_dirs):
        print(f"Calculate & save MMDs for features in {feature_dir}...\n")
        # load intermediate representations
        # loop over all datasets
        all_DMs = ["ADM", "DDPM", "IDDPM", "LDM", "PNDM"]
        all_GANs = [
            "Diff-StyleGAN2",
            "Diff-ProjectedGAN",
            "ProGAN",
            "ProjectedGAN",
            "StyleGAN",
        ]

        X_real = np.load(feature_dir + "/features_Real.npy")
        X_real = torch.from_numpy(X_real)

        print("Calculate MMDs for Diffusion Models...")
        mmds_DM = {}
        for DM in tqdm(all_DMs):
            X_DM = np.load(feature_dir + f"/features_{DM}.npy")
            X_DM = torch.from_numpy(X_DM)

            dists = torch.pdist(torch.cat([X_real, X_DM], dim=0))
            sigma = dists[:100].median() / 2
            mmds_DM[f"{DM}"] = MMD(X_real, X_DM, sigma)

        print(mmds_DM)
        with open(feature_dir + "/mmds_DM.pkl", "wb") as f:
            pickle.dump(mmds_DM, f)

        print("Calculate MMDs for GANs...")
        mmds_GAN = {}
        for GAN in tqdm(all_GANs):
            X_GAN = np.load(feature_dir + f"/features_{GAN}.npy")
            X_GAN = torch.from_numpy(X_GAN)

            dists = torch.pdist(torch.cat([X_real, X_GAN], dim=0))
            sigma = dists[:100].median() / 2
            mmds_GAN[f"{GAN}"] = MMD(X_real, X_GAN, sigma)

        print(mmds_GAN)
        with open(feature_dir + "/mmds_GAN.pkl", "wb") as f:
            pickle.dump(mmds_GAN, f)
