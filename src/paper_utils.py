from __future__ import annotations

import re
import sys
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

DISPLAY_NAMES = {
    "mandelli2022": "Mandelli2022",
    "wang2020": "Wang2020",
    "gragnaniello2021": "Gragnaniello2021",
    "blur_jpg_prob": "",
    "gandetection_resnet50nodown_": "",
    "progan": "ProGAN",
    "stylegan2": "StyleGAN2",
}


def sanitize(s: str) -> str:
    """Sanitize filename by removing illegal characters."""
    s = s.replace(" ", "_")
    return re.sub("[^A-Za-z0-9._-]", "", s)


def get_filename(
    file_format: str,
    kind: str,
    variant: Optional[str] = None,
    experiment: Optional[str] = None,
    data: Optional[str] = None,
    identifiers: Optional[Union[str, List[str]]] = None,
) -> str:
    """
    Return standardized filename with dashes to separate entities. Raises an error if dashes are used within an entity.

    :param file_format: File format.
    :param kind: Kind of output.
    :param variant: Variant for this kind of output.
    :param experiment: Name of the experiment.
    :param data: String identifying the used data.
    :param identifiers: One or multiple additional identifiers put at the end of the filename.
    :return: Composed filename.
    """
    if not isinstance(identifiers, list):
        identifiers = [identifiers]
    parts = []
    for part in [kind, variant, experiment, data] + identifiers:
        if part is not None:
            part.replace("-", "_")
            parts.append(sanitize(part))
    return "-".join(parts) + f".{file_format.lower().lstrip('.')}"


# MATPLOTLIB ====================================


def get_figsize(
    width: Union[int, float, str] = "iclr",
    ratio: float = (5**0.5 - 1) / 2,  # - 0.2, TODO attention!
    fraction: float = 1.0,
    nrows: int = 1,
    ncols: int = 1,
) -> Tuple[float, float]:
    """
    Return width and height of figure in inches.

    :param width: Width of figure in pt or name of conference template.
    :param ratio: Ratio between width and height (height = width * ratio). Defaults to golden ratio (~0.618).
    :param fraction: Scaling factor for both width and height.
    :param nrows, ncols: Number of rows/columns if subplots are used.
    :return: Tuple containing width and height in inches.
    """
    conference_lut = {"iclr": 397.48499}
    if isinstance(width, str):
        if width not in conference_lut:
            raise ValueError(
                f"Available width keys are: {list(conference_lut.keys())}, got:"
                f" {width}."
            )
        width = conference_lut[width]
    height = width * ratio * (nrows / ncols)
    factor = 1 / 72.72 * fraction
    return width * factor, height * factor


def configure_matplotlib(
    context: str = "paper",
    style: str = "whitegrid",
    palette: str = "deep",
    rc: Optional[Dict] = None,
):
    """
    Configure matplotlib using rcParams based on seaborn styles.

    :param context: One of {paper, notebook, talk, poster}.
    :param style: One of {darkgrid, whitegrid, dark, white, ticks}.
    :param palette: One of {deep, muted, bright, pastel, dark, colorblind}.
    :param rc: Dictionary of rc parameter mappings to override.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from seaborn._oldcore import unique_dashes

    dashes = unique_dashes(len(sns.color_palette("deep")))
    dashes[0] = "-"

    # modify rcParams
    params = {
        # font
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath}",
        # "font.family": "serif",
        # "font.serif": "Times",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.title_fontsize": 7,
        "legend.fontsize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        # style
        "lines.linewidth": 0.75,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.linewidth": 0.25,
        "grid.linestyle": "--",
        "grid.color": "black",
        "axes.prop_cycle": mpl.cycler(color=sns.color_palette("deep"))
        + mpl.cycler(linestyle=dashes),
        "axes.labelpad": 0.5,
        "xtick.major.pad": 1,
        "xtick.major.size": 2,
        "xtick.major.width": 0.5,
        "xtick.minor.size": 1,
        "xtick.minor.width": 0.3,
        "xtick.direction": "in",
        "ytick.major.pad": 1,
        "ytick.major.size": 2,
        "ytick.major.width": 0.5,
        "ytick.minor.size": 1,
        "ytick.minor.width": 0.3,
        "ytick.minor.pad": 1,
        "ytick.direction": "in",
        # figure
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.01,
        "figure.constrained_layout.h_pad": 0.01,
        "figure.constrained_layout.wspace": 0.05,
        "figure.constrained_layout.hspace": 0.05,
        "figure.dpi": 300,
        # legend
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.5,
        "legend.borderpad": 0.25,
        # saving
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
    mpl.rcParams.update(params)

    # apply custom rcParams
    if rc is not None:
        mpl.rcParams.update(rc)

    # patch plt.savefig to include script that created the figure
    plt.savefig = partial(plt.savefig, metadata={"Creator": _get_creation_string()})


# PANDAS ========================================


def configure_pandas() -> None:
    """
    Configure the following pandas properties:
    - Display options for printing tables to the console
    - Formatting options for exporting tables to .tex files.
    - Tables saved to .tex files include the script that was used to create them.
    """
    import pandas as pd
    from pandas._typing import FilePath, WriteBuffer
    from pandas.io.formats.format import get_buffer

    # display options
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 20
    pd.options.display.show_dimensions = True

    # .tex options
    pd.options.styler.latex.hrules = True

    # patch save function to include creating script
    def patched_save_to_buffer(
        string: str,
        buf: FilePath | WriteBuffer[str] | None = None,
        encoding: str | None = None,
    ) -> str | None:
        """
        Perform serialization. Write to buf or return as string if buf is None.
        """
        with get_buffer(buf, encoding=encoding) as f:
            if buf is not None and str(buf).lower().endswith(".tex"):
                f.write(f"% {_get_creation_string()}\n")
            f.write(string)
            if buf is None:
                return f.getvalue()
            return None

    pd.io.formats.format.save_to_buffer = patched_save_to_buffer


def combine_columns(
    df: pd.DataFrame, columns: List[str], separator: str = " / "
) -> pd.DataFrame:
    """Combine multiple columns by concatenating their string representations and adding appropriate padding."""

    def _pad_percentage(f: float) -> str:
        return "{:5.1f}".format(f).replace(" ", "\phantom{0}")

    out_df = df.xs(columns[0], axis=1, level=-1).applymap(_pad_percentage)
    for col in columns[1:]:
        out_df += separator + df.xs(col, axis=1, level=-1).applymap(_pad_percentage)

    return out_df


def _get_creation_string() -> str:
    argv = sys.argv
    argv[0] = argv[0].split("/")[-1]
    return "python " + " ".join(argv)
