"""Analyze frequency evolution of diffusion models."""
import argparse
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cytoolz import functoolz
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data import apply_to_imgdir, array_from_imgdir
from src.image import dct, fft, log_scale, spectral_density
from src.paper_utils import configure_matplotlib, get_figsize, get_filename
from src.visualization import plot_evolution, plot_evolution_mean


def _get_dirs(root_dir: Path, img_dirs: List[str]):
    data = {}
    for img_dir in sorted(root_dir.iterdir()):
        if img_dirs is None or img_dir.name in img_dirs:
            timesteps = int(img_dir.name.split("_")[-1])
            data[timesteps] = img_dir

    return {k: v for k, v in sorted(data.items())}


def spectrum_vs_steps(args):
    configure_matplotlib(
        rc={"figure.constrained_layout.use": False, "figure.dpi": 3000}
    )
    output_dir = args.output_root / spectrum_vs_steps.__name__
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    reference = np.load(args.reference)

    evolution, timesteps = [], []
    for ts, img_dir in tqdm(_get_dirs(args.image_root, args.img_dirs).items()):
        timesteps.append(ts)
        evolution.append(
            apply_to_imgdir(
                img_dir=img_dir,
                func=spectral_density,
                grayscale=True,
                cache_dir=output_dir / "cache",
            ).mean(axis=0)
        )
    evolution = np.array(evolution) / reference - 1

    # plot evolution
    if args.all_labels:
        y = None
        yticklabels = timesteps
    else:
        y = timesteps
        yticklabels = None

    plt.figure(figsize=get_figsize(fraction=args.fraction))
    plot_evolution(data=evolution, ylabel=args.ylabel, y=y, yticklabels=yticklabels)
    plt.tight_layout(pad=0.3)

    plt.savefig(
        plot_dir
        / get_filename(
            file_format="pdf",
            kind="spectrum_evolution",
            data=args.image_root.name,
            experiment=args.experiment,
        )
    )
    plt.close()

    # plot average
    configure_matplotlib()
    plt.figure(figsize=get_figsize(fraction=0.49))
    plot_evolution_mean(
        data=abs(evolution), x=timesteps, xlabel=args.ylabel, show_best=False
    )
    plt.savefig(
        plot_dir
        / get_filename(
            file_format="pdf",
            kind="spectrum_evolution_mean",
            data=args.image_root.name,
            experiment=args.experiment,
            overwrite=args.overwrite,
        )
    )
    plt.close()
    print(
        "Timestep with minimum error:",
        timesteps[np.argmin(abs(evolution).mean(axis=1))],
    )


def denoising_vs_diffusion(args):
    configure_matplotlib(
        rc={"figure.constrained_layout.use": False, "figure.dpi": 3000}
    )
    output_dir = args.output_root / spectrum_vs_steps.__name__
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    denoising, denoising_timesteps = [], []
    for ts, img_dir in tqdm(_get_dirs(args.image_root, args.img_dirs).items()):
        denoising_timesteps.append(ts)
        denoising.append(
            apply_to_imgdir(
                img_dir=img_dir,
                func=spectral_density,
                grayscale=True,
                cache_dir=output_dir / "cache",
                overwrite=args.overwrite,
            ).mean(axis=0)
        )

    diffusion, diffusion_timesteps = [], []
    for ts, img_dir in tqdm(_get_dirs(args.diffusion_root, args.img_dirs).items()):
        diffusion_timesteps.append(ts)
        diffusion.append(
            apply_to_imgdir(
                img_dir=img_dir,
                func=spectral_density,
                grayscale=True,
                cache_dir=output_dir / "cache",
            ).mean(axis=0)
        )

    if diffusion_timesteps != denoising_timesteps:
        raise ValueError("Denoising and diffusion have different timesteps.")

    evolution = np.array(denoising) / np.array(diffusion) - 1

    y = denoising_timesteps
    yticklabels = None

    plt.figure(figsize=get_figsize(fraction=args.fraction))
    plot_evolution(
        data=np.array(evolution), ylabel=args.ylabel, y=y, yticklabels=yticklabels
    )
    plt.tight_layout(pad=0.3)

    plt.savefig(
        plot_dir
        / get_filename(
            file_format="pdf",
            kind="diffusion_vs_denoising",
            data=args.image_root.name,
            experiment=args.experiment,
        )
    )
    plt.close()

    # plot average
    configure_matplotlib()
    plt.figure(figsize=get_figsize(fraction=0.49))
    plot_evolution_mean(
        data=abs(evolution), x=denoising_timesteps, xlabel=args.ylabel, show_best=False
    )
    plt.savefig(
        plot_dir
        / get_filename(
            file_format="pdf",
            kind="diffusion_vs_denoising_mean",
            data=args.image_root.name,
            experiment=args.experiment,
        )
    )
    plt.close()
    print(
        "Timestep with minimum error:",
        denoising_timesteps[np.argmin(abs(evolution).mean(axis=1))],
    )


def compute_reference(args):
    output_dir = args.output_root / spectrum_vs_steps.__name__
    output_dir.mkdir(parents=True, exist_ok=True)

    array = array_from_imgdir(
        directory=args.image_root, grayscale=True, num_samples=args.num_samples
    )
    ref = spectral_density(array).mean(axis=0)
    np.save(output_dir / f"{args.identifier}-{args.num_samples}_samples.npy", ref)


def classification(args):
    output_dir = args.output_root / spectrum_vs_steps.__name__
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    transforms = {
        "Pixel": lambda x: x,
        "DCT": dct,
        "log(DCT)": functoolz.compose_left(dct, log_scale),
        "FFT": functoolz.compose_left(fft, np.abs),
        "log(FFT)": functoolz.compose_left(fft, log_scale),
    }

    cache_file = output_dir / "cache" / f"LR_{args.image_root.name}.pickle"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            output = pickle.load(f)
    else:
        real = array_from_imgdir(
            directory=args.real_path,
            crop_size=64,
            grayscale=True,
            num_samples=args.num_samples,
        )
        output = []
        for ts, img_dir in tqdm(_get_dirs(args.image_root, args.img_dirs).items()):
            fake = array_from_imgdir(
                directory=img_dir, crop_size=64, grayscale=True, num_samples=512
            )
            x = np.concatenate([real, fake])
            y = np.array([0] * len(real) + [1] * len(fake))

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=0, stratify=y
            )

            for tf_name, tf_func in transforms.items():
                x_train_tf = tf_func(x_train).reshape((x_train.shape[0], -1))
                x_test_tf = tf_func(x_test).reshape((x_test.shape[0], -1))

                scaler = StandardScaler()
                x_train_tf = scaler.fit_transform(x_train_tf)
                x_test_tf = scaler.transform(x_test_tf)

                clf = LogisticRegressionCV(
                    Cs=[
                        1 / 1e4,
                        1 / 1e3,
                        1 / 1e2,
                        1 / 1e1,
                        1 / 1e0,
                        1 / 1e-1,
                        1 / 1e-2,
                        1 / 1e-3,
                        1 / 1e-4,
                    ],
                    max_iter=1000,
                    n_jobs=args.num_workers,
                )
                clf.fit(x_train_tf, y_train)
                train_score = clf.score(x_train_tf, y_train)
                test_score = clf.score(x_test_tf, y_test)
                output.append((ts, tf_name, train_score, test_score))
                print((ts, tf_name, train_score, test_score))
        with open(cache_file, "wb") as f:
            pickle.dump(output, f)

    df = (
        pd.DataFrame(output, columns=["Step", "Transform", "Train", "Test"])
        .set_index("Step")
        .pivot(columns="Transform", values=["Train", "Test"])
    )

    # plot
    configure_matplotlib()
    plt.figure(figsize=get_figsize(fraction=args.fraction))

    if args.all_labels:
        y = range(len(df.index))
        yticklabels = df.index
        plt.gca().set_yticks(y, labels=yticklabels)
    else:
        y = df.index

    plt.plot(df[("Test", "Pixel")], y, label="Pixel")
    plt.plot(df[("Test", "DCT")], y, label="DCT")
    plt.plot(df[("Test", "log(DCT)")], y, label="log(DCT)")
    plt.plot(df[("Test", "FFT")], y, label="FFT")
    plt.plot(df[("Test", "log(FFT)")], y, label="log(FFT)")

    plt.ylabel(args.ylabel)
    plt.xlabel("Accuracy")
    plt.legend()

    filename = get_filename(file_format="pdf", kind="lr_acc", data=args.image_root.name)
    plt.savefig(plot_dir / filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        default="default",
        help="Custom experiment name to use for output files.",
    )

    plot = parser.add_argument_group("plot")
    plot.add_argument(
        "--fraction", type=float, default=1, help="Fraction of plot size."
    )
    plot.add_argument("--ylabel", type=str, default="$t$", help="Y-Label")
    plot.add_argument("--all-labels", action="store_true", help="Plot all labels.")

    data = parser.add_argument_group("data")
    data.add_argument("image_root", type=Path, help="Root of image directory.")
    data.add_argument("output_root", type=Path, help="Output directory.")
    data.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute instead of using existing data.",
    )
    data.add_argument(
        "--img-dirs",
        nargs="+",
        type=str,
        help="Subset of img-dirs in image_root to use.",
    )

    subparsers = parser.add_subparsers(title="subcommands", help="additional help")

    p_spectrum_vs_steps = subparsers.add_parser(spectrum_vs_steps.__name__)
    p_spectrum_vs_steps.set_defaults(func=spectrum_vs_steps)
    p_spectrum_vs_steps.add_argument(
        "reference", type=Path, help="Path to reference file."
    )

    p_denoising_vs_diffusion = subparsers.add_parser(denoising_vs_diffusion.__name__)
    p_denoising_vs_diffusion.set_defaults(func=denoising_vs_diffusion)
    p_denoising_vs_diffusion.add_argument(
        "diffusion_root", type=Path, help="Path to diffusion root."
    )

    p_compute_reference = subparsers.add_parser(compute_reference.__name__)
    p_compute_reference.set_defaults(func=compute_reference)
    p_compute_reference.add_argument("identifier", help="Name of reference.")
    p_compute_reference.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples (default: 10000).",
    )

    p_classification = subparsers.add_parser(classification.__name__)
    p_classification.set_defaults(func=classification)
    p_classification.add_argument("real_path", type=Path, help="Path to real images.")
    p_classification.add_argument(
        "--num_samples", type=int, default=512, help="Number of samples (default: 512)."
    )
    p_classification.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers (default: 4)."
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    parse_args()
