import numpy as np
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
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray]:
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
    tuple[np.ndarray, np.ndarray, np.float] | tuple[np.ndarray, np.ndarray]
        a tuple containing estimates and upper bounds for potential scale reduction factor for each variable.
        If there is more than one variable, then additionally returns a multivariate psrf.
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

    R2 = np.sqrt(np.vstack([R2_est, R2_upper]) * df_adj)

    if n_var > 1:
        return R2[0, :], R2[1, :], mpsrf()
    else:
        return R2[0, :], R2[1, :]
