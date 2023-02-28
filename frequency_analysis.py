"""Analyze frequency characteristics of images."""
import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Tuple

import cytoolz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data import apply_to_imgdir
from src.image import dct, fft, spectral_density
from src.optimization import parallel_map
from src.paper_utils import configure_matplotlib, get_figsize, get_filename
from src.visualization import plot_power_spectrum, plot_spectra

configure_matplotlib()


def get_mean_std(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(array, axis=0), np.std(array, axis=0)


TRANSFORMS = {
    "dct": cytoolz.compose_left(dct, get_mean_std),
    "fft": cytoolz.compose_left(fft, get_mean_std),
    "fft_hp": cytoolz.compose_left(
        partial(fft, hp_filter=True, hp_filter_size=3), get_mean_std
    ),
    "density": cytoolz.compose_left(spectral_density, get_mean_std),
}


def main(
    image_root: Path,
    output_root: Path,
    transform: str,
    plot_subdir: Path,
    moment: str,
    log: bool,
    vmin: float,
    vmax: float,
    img_dirs: List[str],
    overwrite: bool,
    num_workers: int,
    experiment: str,
    fraction: float,
    zoom: bool,
    diff: bool,
    fixed_height: int,
):
    output_dir = output_root / "frequency_analysis"
    plot_dir = output_dir / "plots"
    if plot_subdir is not None:
        plot_dir = plot_dir / plot_subdir
    plot_dir.mkdir(exist_ok=True, parents=True)

    # compute mean and std of transform
    mean_std = parallel_map(
        apply_to_imgdir,
        [image_root / dirname for dirname in img_dirs],
        num_workers=num_workers,
        mode="multiprocessing",
        func_kwargs=dict(
            func=TRANSFORMS[transform],
            grayscale=True,
            cache_dir=output_dir / "cache",
            overwrite=overwrite,
        ),
    )
    means, stds = zip(*mean_std)
    df = pd.DataFrame({"mean": means, "std": stds}, index=img_dirs)

    # plot
    labels = img_dirs
    if (image_root / "labels.json").exists():
        with open(image_root / "labels.json") as f:
            label_updates = json.load(f)
        labels = [
            label_updates[label] if label in label_updates else label
            for label in labels
        ]

    data = np.stack(df[moment])
    if transform == "density":
        plt.figure(figsize=get_figsize(fraction=fraction))
        if diff:
            data = data[1:] / data[0] - 1
            plot_power_spectrum(
                data=data, labels=labels[1:], log=log, zoom=zoom, first_black=False
            )
            plt.ylabel("Spectral Density Error")
            plt.ylim(-1, 1)
        else:
            plot_power_spectrum(data=data, labels=labels, log=log, zoom=zoom)
            plt.ylim(10**-5, 10**3)
    else:
        plot_spectra(
            data=np.abs(data),
            labels=labels,
            width=get_figsize(fraction=fraction)[0],
            log=log,
            vmin=vmin,
            vmax=vmax,
            fixed_height=fixed_height,
        )

    filename = get_filename(
        file_format="pdf",
        kind=transform,
        variant=moment,
        data="_".join(img_dirs),
        experiment=experiment,
        identifiers="diff" if diff else None,
    )
    plt.savefig(plot_dir / filename)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_root", type=Path, help="Root of image directory.")
    parser.add_argument("output_root", type=Path, help="Output directory.")
    parser.add_argument("transform", choices=TRANSFORMS, help="Transform to apply.")
    parser.add_argument("--plot-subdir", type=Path)
    parser.add_argument(
        "--moment",
        choices=["mean", "std"],
        default="mean",
        help="Whether to plot mean or std.",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Whether to plot difference between real and other dirs.",
    )
    parser.add_argument(
        "--img-dirs",
        nargs="+",
        required=True,
        help="Image directories to analyze, order is maintained.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute instead of using existing data.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers (default: 4)."
    )
    parser.add_argument(
        "--experiment",
        default="default",
        help="Custom experiment name to use for output files.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1,
        help="Fraction of paper width to use for plot.",
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether to plot in log scale."
    )
    parser.add_argument("--vmin", type=float, help="Argument vmin for imshow.")
    parser.add_argument("--vmax", type=float, help="Argument vmax for imshow.")
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Whether to show zoomed in are at highest frequencies.",
    )
    parser.add_argument(
        "--fixed-height",
        type=int,
        help="Adjust height as if there were this many spectra.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
