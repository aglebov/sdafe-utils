from typing import Tuple

import numpy as np


def hill_curve(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate values for the Hill plot

    Parameters
    ----------
    vals: np.ndarray
        the input sample

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        a tuple of three arrays: the values of `c`, `n(c)` and `a(c)` in Eq. (19.32) in SDAFE
    """
    vals = np.sort(vals)
    n_neg = np.sum(vals < 0)
    cumlog = np.cumsum(np.log(-vals[:n_neg]))
    nc = np.arange(1, n_neg + 1)
    c = (vals[:-1] + vals[1:]) / 2
    a = nc / (cumlog[nc - 1] - nc * np.log(-c[:n_neg]))
    return c, nc, a


def hill_curve2(
        vals: np.ndarray,
        lower_q: float = 0.025,
        upper_q: float = 0.25,
        npoints: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate values for the Hill plot using equally spaced threshold values

    Parameters
    ----------
    vals: np.ndarray
        the input sample
    lower_q: float
        the quantile of the sample to use as the smallest threshold value. Default: 0.025
    upper_q: float
        the quantile of the sample to use as the largest threshold value. Default: 0.25
    npoints: int
        the number of threshold values to use

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        a tuple of three arrays: the values of `c`, `n(c)` and `a(c)` in Eq. (19.32) in SDAFE
    """
    n_tail = np.sum(vals <= np.quantile(vals, upper_q))
    c = np.linspace(*np.quantile(vals, [lower_q, upper_q]), npoints)
    vals = np.sort(vals)[:n_tail].reshape(-1, 1)
    nc = np.sum(vals <= c, axis=0)
    cumlog = np.cumsum(np.log(-vals))
    a = nc / (cumlog[nc - 1] - nc * np.log(-c))
    return c, nc, a
