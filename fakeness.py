"""Show example images considered more or less fake by a detector."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.paper_utils import configure_matplotlib, get_figsize, get_filename


def load_data(path: Path) -> pd.DataFrame:
    frames = []
    for file in path.iterdir():
        frames.append(pd.read_pickle(file))
    return pd.concat(frames, axis=1)


def main(args):
    output_dir = args.output_root / "fakeness"
    output_dir.mkdir(exist_ok=True)

    for img_dir in args.img_dirs:
        df = load_data(args.clf_dir / img_dir)
        if args.detector == "all":
            ranking = df.transform(pd.Series.rank).mean(axis=1).sort_values()
        else:
            ranking = df[args.detector].rank().sort_values().dropna()

        configure_matplotlib(
            rc={
                "figure.dpi": 300,
                "figure.constrained_layout.wspace": 0,
                "figure.constrained_layout.hspace": 0,
            }
        )
        fig = plt.figure(figsize=(get_figsize()[0], get_figsize()[1] / 3))
        subfigs = fig.subfigures(nrows=1, ncols=len(args.percentiles), wspace=0.1)
        for p, percentile in enumerate(args.percentiles):
            axs = subfigs[p].subplots(2, 2).flatten()
            start = int(percentile * len(ranking))
            if start + 4 > len(ranking):
                start = len(ranking) - 4
            for i in range(4):
                image = np.asarray(
                    Image.open(args.image_dir / img_dir / ranking.index[start + i][1])
                )
                axs[i].imshow(image)
                axs[i].set_axis_off()
                axs[i].grid()
            subfigs[p].suptitle(f"{int(100*percentile)}th Percentile")
        filename = get_filename(
            file_format="pdf",
            kind="fakeness",
            variant=args.detector,
            data=f"{args.image_dir.name}_{str(img_dir)}",
        )
        plt.savefig(output_dir / filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("clf_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--percentiles", nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--img-dirs", nargs="+", type=Path)
    parser.add_argument(
        "--detector", default="gragnaniello2021_gandetection_resnet50nodown_progan"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
