from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

from sdafe.r import el, py2rpy

from rpy2.robjects.vectors import ListVector
from rpy2.robjects.packages import importr

mgcv = importr('mgcv')


def plot_gam(
        fit: ListVector,
        data: pd.DataFrame,
        axs: Sequence[matplotlib.axes.Axes],
        n: int = 100,
        se_mult: float = 2.0
) -> None:
    """A stripped-down version of plot.gam from the mgcv package in R

    Parameters
    ----------
    fit: ListVector
        the result of calling the `gam` function from the `mgcv` package via rpy2
    data: pd.DataFrame
        the data used to fit the GAM model
    axs: Sequence[matplotlib.axes.Axes]
        the subplots to plot on
    n: int
        the number of points to use for each plot. Default: 100
    se_mult: float
        the multiplier for the standard error to use for confidence intervals
    """
    Vp = np.array(el(fit, 'Vp'))

    for i in range(len(el(fit, 'smooth'))):
        smooth = el(fit, 'smooth')[i]
        name = el(smooth, 'term')[0]

        raw = data[name]
        xx = np.linspace(np.min(raw), np.max(raw), n)

        X = np.array(mgcv.PredictMat(smooth, py2rpy(pd.DataFrame({name: xx}))))

        first_param_idx = int(el(smooth, 'first.para')[0])
        last_param_idx = int(el(smooth, 'last.para')[0])
        p = el(fit, 'coefficients')[first_param_idx - 1:last_param_idx]

        Vp_i = Vp[first_param_idx - 1:last_param_idx, first_param_idx - 1:last_param_idx]
        se = np.sqrt(np.maximum(0, np.sum(X @ Vp_i * X, axis=1)))

        y = X @ p
        ul = y + se * se_mult
        ll = y - se * se_mult

        axs[i].fill_between(xx, ll, ul, alpha=0.2)
        axs[i].plot(xx, ul, '--', color='gray', linewidth=1)
        axs[i].plot(xx, ll, '--', color='gray', linewidth=1)
        axs[i].plot(xx, y)
        axs[i].set_title(name)
        axs[i].set_xlabel(name)
        axs[i].set_ylabel(f's({name})')
