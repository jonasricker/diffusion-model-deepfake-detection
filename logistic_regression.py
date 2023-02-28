"""Perform logistic regression on different image transforms."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from cytoolz import functoolz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from src.data import array_from_imgdir
from src.image import dct, fft, log_scale

TRANSFORMS = {
    "Pixel": lambda x: x,
    "DCT": dct,
    "log(DCT)": functoolz.compose_left(dct, log_scale),
    "FFT": functoolz.compose_left(fft, np.abs),
    "log(FFT)": functoolz.compose_left(fft, log_scale),
}


def main(args):
    output_dir = args.output_root / "logistic_regression"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result_cache_file = (
        cache_dir / f"{args.crop_size}_{args.num_train}_{args.num_val}.pickle"
    )
    if result_cache_file.exists() and not args.overwrite:
        with open(result_cache_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = []

        real_test = array_from_imgdir(
            args.image_root / "test" / "Real",
            crop_size=args.crop_size,
            grayscale=True,
            num_samples=args.num_test,
            num_workers=args.num_workers,
        )

        for img_dir in args.img_dirs:
            real_train = array_from_imgdir(
                args.image_root / "train" / img_dir / "0_real",
                crop_size=args.crop_size,
                grayscale=True,
                num_samples=args.num_train,
                num_workers=args.num_workers,
            )
            fake_train = array_from_imgdir(
                args.image_root / "train" / img_dir / "1_fake",
                crop_size=args.crop_size,
                grayscale=True,
                num_samples=args.num_train,
                num_workers=args.num_workers,
            )
            x_train = np.concatenate([real_train, fake_train])
            y_train = np.array([0] * len(real_train) + [1] * len(fake_train))
            x_train, y_train = shuffle(x_train, y_train, random_state=0)
            del real_train, fake_train

            real_val = array_from_imgdir(
                args.image_root / "val" / img_dir / "0_real",
                crop_size=args.crop_size,
                grayscale=True,
                num_samples=args.num_val,
                num_workers=args.num_workers,
            )
            fake_val = array_from_imgdir(
                args.image_root / "val" / img_dir / "1_fake",
                crop_size=args.crop_size,
                grayscale=True,
                num_samples=args.num_val,
                num_workers=args.num_workers,
            )
            x_val = np.concatenate([real_val, fake_val])
            y_val = np.array([0] * len(real_val) + [1] * len(fake_val))
            x_val, y_val = shuffle(x_val, y_val, random_state=0)
            del real_val, fake_val

            fake_test = array_from_imgdir(
                args.image_root / "test" / img_dir,
                crop_size=args.crop_size,
                grayscale=True,
                num_samples=args.num_test,
                num_workers=args.num_workers,
            )
            x_test = np.concatenate([real_test, fake_test])
            y_test = np.array([0] * len(real_test) + [1] * len(fake_test))
            del fake_test

            print("Finished data loading")

            for tf_name, tf_func in TRANSFORMS.items():
                x_train_tf, x_val_tf, x_test_tf = map(tf_func, [x_train, x_val, x_test])
                x_train_tf, x_val_tf, x_test_tf = (
                    x_train_tf.reshape((len(x_train_tf), -1)),
                    x_val_tf.reshape((len(x_val_tf), -1)),
                    x_test_tf.reshape((len(x_test_tf), -1)),
                )
                print("Finished transform")

                scaler = StandardScaler()
                x_train_tf = scaler.fit_transform(x_train_tf)
                x_val_tf, x_test_tf = map(scaler.transform, [x_val_tf, x_test_tf])
                print("Finished scaling")

                cache_file = (
                    cache_dir
                    / f"{img_dir}_{tf_name}_{args.crop_size}_{args.num_train}_{args.num_val}.pickle"
                )
                if cache_file.exists() and not args.overwrite:
                    with open(cache_file, "rb") as f:
                        best_model = pickle.load(f)
                else:
                    best_model = None
                    for lmbda in [1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]:
                        clf = LogisticRegression(
                            C=1 / lmbda,
                            max_iter=1000,
                        )
                        clf.fit(x_train_tf, y_train)
                        val_score = clf.score(x_val_tf, y_val)
                        if best_model is None or best_model[2] < val_score:
                            best_model = (lmbda, clf, val_score)
                    with open(cache_file, "wb") as f:
                        pickle.dump(best_model, f)
                train_score = best_model[1].score(x_train_tf, y_train)
                test_score = best_model[1].score(x_test_tf, y_test)

                results.append(
                    (img_dir, tf_name, best_model[0], train_score, test_score)
                )
        with open(result_cache_file, "wb") as f:
            pickle.dump(results, f)
    df = pd.DataFrame(
        results, columns=["Dataset", "Transform", "Lambda", "Train Score", "Test Score"]
    ).set_index("Dataset")
    df.to_csv(table_dir / f"{args.crop_size}_{args.num_train}_{args.num_val}.csv")

    df = df.drop(["Lambda", "Train Score"], axis=1)
    df = df.pivot(columns="Transform").droplevel(0, axis=1)
    df = df.reindex(index=args.img_dirs, columns=TRANSFORMS)
    if args.columns is not None:
        df = df[args.columns]

    # add gain columns
    gain_columns = []
    for i in range(df.shape[1] - 1, 0, -1):
        gain = df.iloc[:, i] - df["Pixel"]
        df.insert(i + 1, column=f"Gain_{i}", value=gain, allow_duplicates=True)
        gain_columns.append(f"Gain_{i}")

    # change style
    df *= 100
    s = df.style
    s = s.format(precision=1)
    s = s.format(formatter="{:+.1f}", subset=gain_columns)
    s = s.set_properties(subset=gain_columns, color="{gray}")
    s = s.highlight_max(
        subset=[col for col in df if col not in gain_columns],
        axis=1,
        props="textbf:--rwrap;",
    )
    s = s.format_index(lambda label: "" if label.startswith("Gain") else label, axis=1)
    s = s.hide(names=True, axis=0).hide(names=True, axis=1)

    filename = table_dir / f"{args.crop_size}_{args.num_train}_{args.num_val}.tex"
    s.to_latex(filename, hrules=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_root",
        type=Path,
        help="Root of image directory containing 'train', 'val', and test.",
    )
    parser.add_argument("output_root", type=Path, help="Output directory.")
    parser.add_argument(
        "--img-dirs",
        nargs="+",
        required=True,
        help="Names of directories in 'train' and 'val'.",
    )
    parser.add_argument(
        "--crop-size", type=int, default=64, help="Size the image will be cropped to."
    )
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=10000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute instead of using existing data.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of workers (default: 8)."
    )
    parser.add_argument(
        "--experiment",
        default="default",
        help="Custom experiment name to use for output files.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["Pixel", "DCT", "log(DCT)", "FFT", "log(FFT)"],
        help="Columns to print.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
