from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult
import scipy.stats as stats


def silverman_bw(vals: np.ndarray | pd.Series) -> float:
    """Select bandwidth for a kernel density estimator using Silverman's rule of thumb

    Parameters
    ----------
    vals: np.ndarray | pd.Series
        input data

    Returns
    -------
    float
        the selected bandwidth value

    Notes
    -----
    See https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth
    """
    return 0.9 * min(np.std(vals, ddof=1), stats.iqr(vals) / 1.34) * len(vals) ** (-0.2)


def loglik_t(x: np.ndarray | pd.Series, loc: float, scale: float, df: int) -> float:
    """Log-likelihood of a sample under t-distribution

    Parameters
    ----------
    x: np.ndarray | pd.Series
        the sample to use
    loc: float
        the location parameter of the t-distribution
    scale: float
        the scale parameter of the t-distribution
    df: int
        the degrees of freedom of the t-distribution

    Returns
    -------
    float
        the log-likelihood of the sample under the t-distribution with the given parameters
    """
    return np.sum(-stats.t.logpdf(x, loc=loc, scale=scale, df=df))


def fit_t_distr(x: np.ndarray | pd.Series) -> OptimizeResult:
    """Fit t-distribution to the input data using the maximum likelihood method

    Parameters
    ----------
    x: np.ndarray | pd.Series
        the sample to use

    Returns
    -------
    OptimizeResult
        the result of optimisation, as returned by `scipy.optimize.minimize`
    """
    start = np.array([np.mean(x), np.std(x, ddof=1), 2])
    bounds = [(None, None), (0.001, None), (1, None)]
    return minimize(lambda beta: loglik_t(x, *beta), start, method='L-BFGS-B', bounds=bounds)
