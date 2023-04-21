from typing import Tuple

import numpy as np
import pandas as pd


def bootstrap_corr(
        data: pd.DataFrame,
        n_boot: int = 10_000,
        alpha: float = 0.95,
        rng: np.random.Generator = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap estimates of confidence intervals for correlation

    Parameters
    ----------
    data: pd.DataFrame
        the input dataframe: rows are observations, columns are variables
    n_boot: int
        the number of resamples to use. Default: 10000
    alpha: float
        the confidence level of the confidence intervals to estimate.
        Default: 0.95
    rng: np.random.Generator
        the random number generator to use. If not provided, the default
        numpy generator is used.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        a tuple of three dataframes:

           * the correlation estimate,
           * the lower bounds of the confidence intervals,
           * the upper bounds of the confidence intervals.
    """
    if rng is None:
        rng = np.random.default_rng()

    var_names = data.columns
    data = data.values
    n, p = data.shape

    # the estimate
    estimate = np.corrcoef(data, rowvar=False)

    # bootstrap for confidence intervals
    corrs = np.zeros((n_boot, p, p))
    for i in range(n_boot):
        corrs[i] = np.corrcoef(data[rng.choice(n, size=n), :], rowvar=False)
    # use basic bootstrap interval
    confints = 2 * estimate - np.quantile(corrs, [(1 + alpha) / 2, (1 - alpha) / 2], axis=0)

    def to_df(a):
        return pd.DataFrame(a, index=var_names, columns=var_names)

    return to_df(estimate), to_df(confints[0]), to_df(confints[1])
