from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats


def VaR(sample: np.ndarray | pd.DataFrame, alpha: Optional[float] = 0.05, s: Optional[float] = 1.0) -> float:
    """Non-parametric VaR estimate

    Parameters
    ----------
    sample: np.array | pd.DataFrame
        the sample of return values to estimate the VaR from
    alpha: Optional[float]
        the confidence parameter for the VaR, e.g. 0.05 to obtain 95% VaR. Default: 0.05
    s: Optional[float]
        the position size. Default: 1.0

    Returns
    -------
    float
        the estimated value of VaR for a given confidence parameter and position size
    """
    return -s * np.quantile(sample, alpha)


def ES(sample: np.ndarray | pd.DataFrame, alpha: Optional[float] = 0.05, s: Optional[float] = 1.0) -> float:
    """Non-parametric expected shortfall estimate

    Parameters
    ----------
    sample: np.array | pd.DataFrame
        the sample of return values to estimate the ES from
    alpha: Optional[float]
        the confidence parameter for the ES, e.g. 0.05 to obtain 95% VaR. Default: 0.05
    s: Optional[float]
        the position size. Default: 1.0

    Returns
    -------
    float
        the value of expected shortfall for a given confidence parameter and position size
    """
    return -s * np.mean(sample[sample <= -VaR(sample, alpha)])


def VaR_norm(
        loc: float,
        scale: float,
        alpha: Optional[float] = 0.05,
        s: Optional[float] = 1.0,
) -> float:
    """VaR for a normally distributed loss

    Parameters
    ----------
    loc: float
        the location parameter of the normal distribution
    scale: float
        the scale parameter of the normal distribution
    alpha: float
        the confidence parameter for the VaR, e.g. 0.05 to obtain 95% VaR. Default: 0.05
    s: float
        the position size. Default: 1.0

    Returns
    -------
    float
        the value of VaR for a given confidence parameter and position size
    """
    return -s * stats.norm.ppf(alpha, loc=loc, scale=scale)


def ES_norm(
        loc: float,
        scale: float,
        alpha: Optional[float] = 0.05,
        s: Optional[float] = 1.0
) -> float:
    """Expected shortfall for a normally distributed loss

    Parameters
    ----------
    loc: float
        the location parameter of the t-distribution
    scale: float
        the scale parameter of the t-distribution
    alpha: float
        the confidence parameter for the expected shortfall,
        e.g. 0.05 to obtain 95% ES. Default: 0.05
    s: float
        the position size. Default: 1.0

    Returns
    -------
    float
        the value of expected shortfall for a given confidence parameter and position size
    """
    p = stats.norm.pdf(stats.norm.ppf(alpha))
    return s * (-loc + scale * p / alpha)


def VaR_t(
        df: int,
        loc: float,
        scale: float,
        alpha: Optional[float] = 0.05,
        s: Optional[float] = 1.0,
) -> float:
    """VaR for a t-distributed loss

    Parameters
    ----------
    df: int
        the degrees of freedom parameter of the t-distribution
    loc: float
        the location parameter of the t-distribution
    scale: float
        the scale parameter of the t-distribution
    alpha: float
        the confidence parameter for the VaR, e.g. 0.05 to obtain 95% VaR. Default: 0.05
    s: float
        the position size. Default: 1.0

    Returns
    -------
    float
        the value of VaR for a given confidence parameter and position size
    """
    return -s * stats.t.ppf(alpha, loc=loc, scale=scale, df=df)


def ES_t(
        df: int,
        loc: float,
        scale: float,
        alpha: Optional[float] = 0.05,
        s: Optional[float] = 1.0
) -> float:
    """Expected shortfall for a t-distributed loss

    Parameters
    ----------
    df: int
        the degrees of freedom parameter of the t-distribution
    loc: float
        the location parameter of the t-distribution
    scale: float
        the scale parameter of the t-distribution
    alpha: float
        the confidence parameter for the expected shortfall,
        e.g. 0.05 to obtain 95% ES. Default: 0.05
    s: float
        the position size. Default: 1.0

    Returns
    -------
    float
        the value of expected shortfall for a given confidence parameter and position size
    """
    q = stats.t.ppf(alpha, df=df)
    p = stats.t.pdf(q, df=df)
    return s * (-loc + scale * p / alpha * (df + q ** 2) / (df - 1))
