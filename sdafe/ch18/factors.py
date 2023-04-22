from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

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


def estimate_corr(data: pd.DataFrame, alpha: float = 0.95) -> pd.DataFrame:
    """Estimate correlations, confidence intervals and p-values for pairs of variables

    Parameters
    ----------
    data: pd.DataFrame
        the input data: rows are observations, columns are variables
    alpha: float
        the confidence for the confidence interval. Default: 0.95

    Returns
    -------
    pd.DataFrame
        a dataframe of results with rows for every pair of variables in the input data
        and the following columns
           * Variable1: the name of the first variable
           * Variable2: the name of the second variable
           * Correlation: the estimate of the correlation between the two variables
           * Lower bound: the lower bound of the confidence interval
           * Upper bound: the upper bound of the confidence interval
           * p-value: the p-value for the hypothesis that the true correlation is 0
    """
    temp = []
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            res = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])
            confint = res.confidence_interval(alpha)
            temp.append([data.columns[i], data.columns[j], res.statistic, confint[0], confint[1], res.pvalue])
    return pd.DataFrame(
        temp,
        columns=['Variable1', 'Variable2', 'Correlation', 'Lower bound', 'Upper bound', 'p-value']
    )
