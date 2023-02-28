from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable


def plot_evolution(
    data: np.ndarray,
    ylabel: str,
    y: Optional[List] = None,
    yticklabels: Optional[List] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create two-dimensional evolution plot going from the bottom to the top. Range of colormap goes from -1 to 1.

    :param data: Two-dimensional array containing the data. First row is plotted at the bottom, last at the top.
    :param yticklabels: Alternative labels to plot on y-axis.
    :param ax: Axes to use, if None use currently-active Axes.
    :return: Axes with plot.
    """
    if ax is None:
        ax = plt.gca()

    # plot
    im = ax.imshow(
        X=data,
        vmin=-1,
        vmax=1,
        origin="lower",
        cmap=sns.color_palette("vlag", as_cmap=True),
        aspect="auto",
        interpolation="none",
        extent=(
            None if y is None else (-0.5, data.shape[1] - 0.5, y[0] - 0.5, y[-1] + 0.5)
        ),
    )

    # labels
    ax.set_xticks(np.linspace(0, data.shape[1] - 1, 3), labels=[0, 0.5, 1.0])
    ax.set_xlabel("$f/f_{nyq}$")
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)), labels=yticklabels)
    ax.set_ylabel(ylabel)

    # style
    ax.grid(False)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set_rasterization_zorder(2)

    return ax


def plot_evolution_mean(
    data: np.ndarray,
    x: List,
    xlabel: str,
    show_best: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot average of spectrum evolution."""
    avg = data.mean(axis=1)

    if ax is None:
        ax = plt.gca()

    ax.plot(x, avg)
    ax.set_ylabel("Spectral Density Error")
    ax.set_xlabel(xlabel)

    if show_best:
        best_x = np.argmin(avg)
        best_y = avg[best_x]
        ax.annotate(f"({best_x}, {best_y})", (best_x, best_y))

    return ax


def plot_power_spectrum(
    data: np.ndarray,
    labels: List[str],
    log: bool = True,
    zoom: bool = False,
    first_black: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot reduced spectra."""
    # validation
    if len(data.shape) != 2:
        raise ValueError(
            f"Array 'data' should have shape (datasets, frequencies), got {data.shape}."
        )
    if data.shape[0] != len(labels):
        raise ValueError(
            f"Mismatch between first dimension of 'data' ({data.shape[0]}) and number"
            f" of labels ({len(labels)})."
        )

    if ax is None:
        ax = plt.gca()

    # plot
    if first_black:
        plt.plot(data[0], "k-", label=labels[0])
    for i in range(first_black, data.shape[0]):
        plt.plot(data[i], label=labels[i])
    if zoom:
        axins = ax.inset_axes([0.6, 0.4, 0.375, 0.55])
        if first_black:
            axins.plot(data[0], "k-")
        for i in range(first_black, data.shape[0]):
            axins.plot(data[i])

        x_min, x_max = int(0.9 * data.shape[1]), data.shape[1]
        axins.set_xlim(x_min, x_max - 0.75)
        axins.set_ylim(0.8e-4, 0.7e-2)

        if log:
            axins.set_yscale("log")
        axins.get_xaxis().set_visible(False)
        for edge in ["bottom", "top", "right", "left"]:
            axins.spines[edge].set_color("black")

        rectangle_path, connector_lines = ax.indicate_inset_zoom(
            axins, edgecolor="black", alpha=1, linewidth=0.5
        )
        for line in connector_lines:
            line.set_linewidth(0.5)

    # axes
    ax.set_xticks(np.linspace(0, data.shape[1] - 1, 3), labels=[0, 0.5, 1.0])
    ax.set_xlabel("$f/f_{nyq}$")
    ax.set_ylabel("Spectral Density")
    if log:
        ax.set_yscale("log")

    # style
    ax.legend(ncol=3)

    return ax


@mpl.rc_context({"figure.dpi": 1000, "figure.constrained_layout.use": False})
def plot_spectra(
    data: np.ndarray,
    labels: List[str],
    width: float,
    log: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fixed_height: int = None,
) -> plt.Figure:
    """Plot 2D frequency spectra."""
    # get vim/vmax
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    # figure setup
    fig = plt.figure(
        figsize=(
            width,
            width / (len(labels) if fixed_height is None else fixed_height) + 0.1,
        )
    )
    grid = AxesGrid(
        fig, 111, nrows_ncols=(1, len(labels)), cbar_mode="edge", cbar_size="10%"
    )
    for ax in grid:
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    # plot
    if log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin, vmax = None, None
    else:
        norm = None

    for i, (array, label) in enumerate(zip(data, labels)):
        im = grid[i].imshow(
            array,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            cmap=sns.color_palette("mako", as_cmap=True),
        )
        if len(data) != 1:
            grid[i].set_title(label, pad=2)

        if i == len(labels) - 1:
            grid[-1].cax.colorbar(im)

    plt.tight_layout(pad=0.5)
    return fig
