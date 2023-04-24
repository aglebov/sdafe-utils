from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult
import scipy.stats as stats


def loglik_t(x: Union[np.array, pd.Series], loc: float, scale: float, df: int) -> float:
    """Log-likelihood of a sample under t-distribution

    Parameters
    ----------
    x: Union[np.array, pd.Series]
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


def fit_t_distr(x: Union[np.array, pd.Series]) -> OptimizeResult:
    """Fit t-distribution to the input data using the maximum likelihood method

    Parameters
    ----------
    x: Union[np.array, pd.Series]
        the sample to use

    Returns
    -------
    OptimizeResult
        the result of optimisation, as returned by `scipy.optimize.minimize`
    """
    start = np.array([np.mean(x), np.std(x, ddof=1), 2])
    bounds = [(None, None), (0.001, None), (1, None)]
    return minimize(lambda beta: loglik_t(x, *beta), start, method='L-BFGS-B', bounds=bounds)
