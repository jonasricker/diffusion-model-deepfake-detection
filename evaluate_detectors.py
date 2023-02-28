"""Evaluate deepfake detectors."""
import argparse
import copy
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
from boltons.strutils import MultiReplace
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torchvision.transforms import Compose

from src.image_perturbations import Perturbation, blur, cropping, jpeg, noise
from src.metrics import pd_at
from src.optimization import parallel_map
from src.paper_utils import (
    combine_columns,
    configure_matplotlib,
    configure_pandas,
    get_figsize,
    get_filename,
    DISPLAY_NAMES
)
from src.wrappers import (
    PredictorWrapper,
    gragnaniello2021_pred,
    gragnaniello2021_setup,
    mandelli2022_pred,
    mandelli2022_setup,
    wang2020_pred,
    wang2020_setup,
)

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# DEFINITIONS ==========================================================================================================

PREDICTORS = {
    "wang2020": PredictorWrapper(
        name="wang2020",
        model_path="wang2020/blur_jpg_prob0.1.pth",
        setup_func=wang2020_setup,
        pred_func=wang2020_pred,
    ),
    "gragnaniello2021": PredictorWrapper(
        name="gragnaniello2021",
        model_path="gragnaniello2021/gandetection_resnet50nodown_stylegan2.pth",
        setup_func=gragnaniello2021_setup,
        pred_func=gragnaniello2021_pred,
    ),
    "mandelli2022": PredictorWrapper(
        name="mandelli2022",
        model_path="mandelli2022",
        setup_func=mandelli2022_setup,
        pred_func=mandelli2022_pred,
    ),
}

THRESHOLDS = {"wang2020": 0.5, "gragnaniello2021": 0, "mandelli2022": 0}

METRICS = {
    "Acc": accuracy_score,
    "AUROC": roc_auc_score,
    "AP": average_precision_score,
    "PD@10%": partial(pd_at, x=0.1),
    "PD@5%": partial(pd_at, x=0.05),
    "PD@1%": partial(pd_at, x=0.01),
}

PERTURBATIONS = {
    "noise": Perturbation(noise),
    "blur": Perturbation(blur),
    "jpeg": Perturbation(jpeg),
    "cropping": Perturbation(cropping),
}

# MAIN LOGIC ===========================================================================================================


def compute_prediction(wrapper: PredictorWrapper, output_dir: Path, args) -> pd.DataFrame:
    """Compute predictions for a nested image directory."""
    if wrapper.model_path.stem == "model_epoch_best":
        predictor_id = f"{wrapper.name}_{wrapper.model_path.parents[1].name}_{wrapper.model_path.parents[0].name}"
    else:
        predictor_id = f"{wrapper.name}_{wrapper.model_path.stem}"

    frames = []
    for img_dir in args.img_dirs:
        logger.info(f"Predicting {img_dir} with {predictor_id}.")
        perturbation_suffix = f"_{args.perturbation}" if args.perturbation is not None else ""
        output_path = output_dir / "cache" / (img_dir + perturbation_suffix) / f"{predictor_id}.pickle"
        if output_path.exists() and not args.overwrite:
            df = pd.read_pickle(output_path)
        else:
            if wrapper.model is None:
                wrapper.setup()
                if args.perturbation is not None:
                    wrapper.transform = Compose([PERTURBATIONS[args.perturbation], wrapper.transform])
            scores, paths = wrapper.predict_dir(
                args.img_root / img_dir, num_workers=args.num_workers // len(args.gpus), batch_size=args.batch_size
            )
            df = pd.DataFrame(
                scores,
                columns=[predictor_id],
                index=pd.MultiIndex.from_product([[img_dir], [Path(path).name for path in paths]]),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(output_path)
        frames.append(df)
    del wrapper.model
    torch.cuda.empty_cache()
    return pd.concat(frames)


def apply_metrics_to_df(df: pd.DataFrame, real_df: pd.DataFrame, metrics: Dict[str, Callable]) -> pd.Series:
    frames = []
    for predictor in df.columns:
        real = real_df[predictor].to_numpy()
        fake = df[predictor].to_numpy()
        y_score = np.concatenate([real, fake])
        y_true = np.array([0] * len(real) + [1] * len(fake))

        out = {}
        for metric, func in metrics.items():
            if metric == "Acc":
                out[metric] = func(y_true=y_true, y_pred=y_score > THRESHOLDS[predictor.split("_")[0]])
            else:
                out[metric] = func(y_true=y_true, y_score=y_score)
        frames.append(pd.DataFrame(out, index=[predictor]))
    return pd.concat(frames)


def evaluate(args):
    """Main function."""
    output_dir = args.output_root / evaluate.__name__
    table_dir = output_dir / "tables"
    table_dir.mkdir(exist_ok=True, parents=True)

    predictors = []
    for name in args.predictors:
        model_paths = vars(args)[f"{name}_model_path"]
        model_paths = model_paths if isinstance(model_paths, list) else [model_paths]
        for model_path in model_paths:
            p = copy.deepcopy(PREDICTORS[name])
            p.model_path = args.model_root / model_path
            if p.name == "wang2020" and p.model_path.is_dir():
                p.model_path = p.model_path / "model_epoch_best.pth"
            predictors.append(p)
    results = parallel_map(
        compute_prediction,
        predictors,
        func_kwargs={"output_dir": output_dir, "args": args},
        gpu_ids=args.gpus,
    )
    prediction_df = pd.concat(results, axis=1)
    real_df = prediction_df.loc[[args.real_dir]]
    prediction_df.drop(args.real_dir, inplace=True)

    metric_df = (
        prediction_df.groupby(level=0, sort=False)
        .apply(apply_metrics_to_df, real_df, METRICS)
        .unstack()
        .reorder_levels([1, 0], axis=1)
    )
    metric_df *= 100

    if args.output == "table":
        configure_pandas()

        if len(args.metric) == 1:
            args.metric = args.metric[0]

        if isinstance(args.metric, list):
            df = combine_columns(metric_df, columns=args.metric)
        else:
            df = metric_df.xs(args.metric, axis=1, level=-1)

        df.columns = pd.MultiIndex.from_tuples(
            [s.split("_", maxsplit=1) for s in df.columns], names=["detector", "variant"]
        )

        rep = MultiReplace(DISPLAY_NAMES)
        s = df.style
        s = s.format(precision=1)
        s = s.format_index(rep.sub, axis=1)
        s = s.hide(axis=0, names=True)
        s = s.hide(axis=1, names=True)
        if isinstance(args.metric, str):
            s = s.format(precision=1)
            s = s.highlight_max(axis=1, props="textbf:--rwrap;")

        filename = get_filename(
            file_format="tex",
            kind="clf_metrics",
            variant=args.metric if isinstance(args.metric, str) else "_".join(args.metric),
            experiment=args.experiment,
        )
        s.to_latex(
            table_dir / filename,
            hrules=True,
            # column_format="l" + "c" * len(out_df.columns),
            multicol_align="c",
        )
    else:
        if len(args.metric) != 1:
            raise ValueError("Confusion matrix only support single metric.")
        configure_matplotlib(
            rc={
                "xtick.labelbottom": False,
                "xtick.bottom": False,
                "xtick.labeltop": True,
                "ytick.left": False,
                "figure.constrained_layout.use": False,
                "savefig.pad_inches": 0.01,
                "axes.grid": False,
            }
        )
        fig, ax = plt.subplots(1, 1, figsize=get_figsize(fraction=0.88))
        metric_df = metric_df.loc[:, (slice(None), args.metric[0])].droplevel(1, axis=1)
        sns.heatmap(
            metric_df,
            xticklabels=[label.split("_")[-1] for label in metric_df.columns],
            annot=True,
            annot_kws={"fontsize": 4.5},
            fmt=".1f",
            cbar=False,
            square=True,
            linewidths=1,
            ax=ax,
            vmin=0,
            vmax=100,
        )
        plt.xticks(rotation=45)
        plt.xlabel("Fine-tuned on")
        ax.xaxis.set_label_position("top")
        plt.ylabel("Tested on")

        ax.vlines([5, 10], ymin=0.04, ymax=9.96, colors=["k"], linestyles=["solid"], linewidths=[0.5])
        ax.hlines([5], xmin=0.04, xmax=12.96, colors=["k"], linestyles=["solid"], linewidths=[0.5])

        filename = get_filename(
            file_format="pdf",
            kind="clf_metrics",
            variant=args.metric[0],
            experiment=args.experiment,
        )
        plt.savefig(table_dir / filename)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", help="Custom experiment name to use for output files.")

    data = parser.add_argument_group("data")
    data.add_argument("img_root", type=Path, help="Root of image directory.")
    data.add_argument("model_root", type=Path, help="Root of model directory.")
    data.add_argument("output_root", type=Path, help="Output directory.")
    parser.add_argument(
        "--img-dirs", nargs="+", required=True, help="Image directories to analyze, order is maintained."
    )
    data.add_argument("--overwrite", action="store_true", help="Recompute instead of using existing data.")

    technical = parser.add_argument_group("technical")
    technical.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers, will be split across GPUs (default: 4)."
    )
    technical.add_argument(
        "--gpus",
        default="0",
        type=lambda s: list(map(int, s.split(","))),
        help="Comma-separated list of GPUs to use (default: 0).",
    )
    technical.add_argument("--batch-size", type=int, help="Batch size to use, by default is determined automatically.")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--metric", nargs="+", choices=METRICS, default=["AUROC"], help="One or more metrics to use."
    )
    evaluation.add_argument(
        "--predictors",
        nargs="+",
        choices=PREDICTORS.keys(),
        default=list(PREDICTORS.keys()),
        help="Predictors to evaluate (default: all).",
    )
    for name, predictor in PREDICTORS.items():
        evaluation.add_argument(
            f"--{name}-model-path",
            type=Path,
            nargs="+",
            default=predictor.model_path,
            help=f"One or multiple paths to model for {name}, relative to model_root (default: {predictor.model_path}).",
        )
    evaluation.add_argument("--output", choices=["table", "cfm"], default="table", help="Output to create")
    evaluation.add_argument("--real-dir", default="Real", help="Name of directory containing real images.")
    evaluation.add_argument(
        "--perturbation",
        choices=PERTURBATIONS.keys(),
        help="Perturbation to apply, use 'combined' to apply all combined.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
