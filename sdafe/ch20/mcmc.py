from typing import Sequence

import arviz
import matplotlib.figure
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.linalg import solve_triangular

from sdafe.linalg import cov_rows


def max_eigenv_sym(A: np.ndarray, B: np.ndarray) -> float:
    """Find the maximum eigenvalue of matrix X that solves AX = B given A is symmetric positive semi-definite"""
    C = np.linalg.cholesky(A)
    w, _ = np.linalg.eig(solve_triangular(C, solve_triangular(C, B, lower=True).T, lower=True))
    return float(np.max(w))


def gelman_diag(
        x: np.ndarray, confidence: float = 0.95
) -> tuple[np.ndarray, float] | np.ndarray:
    """Estimate potential scale reduction factor based on the results of an MCMC simulation

    This is a direct translation to Python of the `gelman.diag` function from the `coda` package in R:
    https://github.com/cran/coda/blob/master/R/gelman.R

    Parameters
    ----------
    x: np.ndarray
        a 3D array of shape(`n_iter`, `n_var`, `n_chain`) of samples simulation, where
        `n_iter` is the number of draws in each chain, `n_var` is the number of variables,
        `n_chain` is the number of chains
    confidence: float
        the confidence level for the upper bound estimate. Default: 0.95

    Returns
    -------
    tuple[np.ndarray, np.float] | np.ndarray
        an array containing point estimates (first column) and upper bounds (second column)
        for potential scale reduction factor for each variable. If there is more than one variable,
        then a tuple contain the array of estimates and the multivariate psrf value.
    """
    n_iter, n_var, n_chain = x.shape

    # covariances for each chain
    S2 = np.stack([np.cov(x[:, :, i_chain], rowvar=False, ddof=1) for i_chain in range(n_chain)], axis=-1)
    # within-chain covariances (averaging across chains)
    W = np.mean(S2, axis=2)
    # averages over iterations
    xbar = np.mean(x, axis=0)
    # averages over iterations and chains, i.e. the resulting parameter estimates
    muhat = np.mean(xbar, axis=1)
    # between-chain covariances
    B = n_iter * np.cov(xbar, ddof=1)

    def mpsrf():
        """Multivariate potential scale reduction factor"""
        return np.sqrt((1 - 1 / n_iter) + (1 + 1 / n_var) * max_eigenv_sym(W, B) / n_iter)

    w = np.diag(W)
    b = np.diag(B)

    # variances for each chain
    s2 = np.stack([np.diag(S2[:, :, i_chain]) for i_chain in range(n_chain)], axis=-1)

    var_w = np.var(s2, axis=1, ddof=1) / n_chain
    var_b = 2 * b ** 2 / (n_chain - 1)
    cov_wb = n_iter / n_chain * np.diag(cov_rows(s2, xbar ** 2, ddof=1) - 2 * muhat * cov_rows(s2, xbar, ddof=1))

    V = (n_iter - 1) * w / n_iter + (1 + 1 / n_chain) * b / n_iter
    var_V = ((n_iter - 1) ** 2 * var_w + (1 + 1 / n_chain) ** 2 * var_b + 2 * (n_iter - 1) * (1 + 1 / n_chain) * cov_wb) / n_iter ** 2
    df_V = 2 * V ** 2 / var_V

    df_adj = (df_V + 3) / (df_V + 1)
    B_df = n_chain - 1
    W_df = 2 * w ** 2 / var_w

    R2_fixed = (n_iter - 1) / n_iter
    R2_random = (1 + 1 / n_chain) / n_iter * b / w

    R2_est = R2_fixed + R2_random
    R2_upper = R2_fixed + stats.f.ppf((1 + confidence) / 2, B_df, W_df) * R2_random

    R2 = np.sqrt(np.vstack([R2_est, R2_upper]) * df_adj).T

    if n_var > 1:
        return R2, mpsrf()
    else:
        return R2


def gelman_diag_arviz(data: arviz.InferenceData, var_names: Sequence[str]) -> tuple[pd.DataFrame, float]:
    """Summary of Gelman diagnostics for an `arviz.InferenceData` instance

    Parameters
    ----------
    data: arviz.InferenceData
        the results of a simulation
    var_names: Sequence[str]
        a sequence of variable names present in the provided inference data for which to calculate diagnostics

    Returns
    -------
    tuple[pd.DataFrame, float]
        a tuple with two elements: a dataframe containing point estimates of the potential scale reduction factor
        and the upper bounds of the confidence intervals, and the multivariate potential scale reduction factor
        value
    """
    psrf, mpsrf = gelman_diag(np.stack([data.posterior[v].T for v in var_names], axis=1))
    return pd.DataFrame(psrf, columns=['Point est.', 'Upper C.I.'], index=var_names), mpsrf


def gelman_plot(
        fig: matplotlib.figure.Figure,
        data: arviz.InferenceData,
        var_names: Sequence[str],
        confidence: float = 0.95,
        max_bins: int = 50,
        guideline_level: float | None = None
) -> matplotlib.figure.Figure:
    """Plot of the evolution of the potential scale reduction factor as the number of iterations increases

    This is done by recalculating the shrink factor for an expanding window of draws taken from the results
    of the simulation.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        the figure to draw on
    data: arviz.InferenceData
        the results of a simulation
    var_names: Sequence[str]
        a sequence of variable names present in the provided inference data for which to calculate diagnostics
    confidence: float
        the confidence level for the upper bound estimate. Default: 0.95
    max_bins: int
        the number of intervals to use when evaluating the shrink factors. Default: 50
    guideline_level: float
        if provided, a horizontal line is plotted to mark the level of the shrink factor that is considered acceptable

    Returns
    -------
    matplotlib.figure.Figure
        the figure object that was passed to this function
    """
    x = np.stack([data.posterior[v].T for v in var_names], axis=1)
    n_iter = x.shape[0]
    n_bin = min(n_iter, max_bins)
    draws = (np.arange(n_bin) + 1) * (n_iter // n_bin)
    rhat = np.stack([gelman_diag(x[:d, :, :])[0] for d in draws], axis=-1)

    axs = fig.axes
    for i, var in enumerate(var_names):
        axs[i].plot(draws, rhat[i][0], label='median')
        axs[i].plot(draws, rhat[i][1], '--', label=f'{(1 + confidence)*50}%')
        axs[i].axhline(1.0, color='black', linewidth=0.5)
        if guideline_level is not None:
            axs[i].axhline(guideline_level, color='red', linestyle='--', linewidth=0.5)
        axs[i].set_xlabel('draw')
        axs[i].set_ylabel('$\\hat{R}$')
        axs[i].set_title(var)
        axs[i].legend()

    return fig
