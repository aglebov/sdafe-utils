from typing import Union

import numpy as np
import pandas as pd


class PCAResult:
    """Summary of results of a PCA"""

    def __init__(self, n_obs: int, center: np.ndarray, sdev: np.ndarray, loadings: np.ndarray,
                 scores: np.ndarray):
        """Initialise PCA results

        Parameters
        ----------
        n_obs: int
            the number of observations in the input data
        center: np.ndarray
            the means of the input variables
        sdev: np.ndarray
            the standard deviations of the principal components
            (the square roots of the eigenvalues)
        loadings: np.ndarray
            the component loadings (the eigenvectors)
        scores: np.ndarray
            the projections of input data on the principal components
        """
        self.n_obs = n_obs
        self.center = center
        self.sdev = sdev
        self.loadings = loadings
        self.scores = scores

    def summary(self) -> pd.DataFrame:
        """Returns a component importance summary"""
        var = self.sdev ** 2
        var_proportion = var / np.sum(var)
        return pd.DataFrame([
            self.sdev,
            var_proportion,
            np.cumsum(var_proportion),
        ], columns=[f'Comp.{i + 1}' for i in range(len(self.sdev))],
            index=['Standard deviation', 'Proportion of Variance', 'Cumulative Proportion'])


def princomp(X: np.ndarray | pd.DataFrame, cor: bool = False, compat: bool = True):
    """Performs a PCA and returns a summary of the results

    Parameters
    ----------
    X: np.ndarray | pd.DataFrame
        the data to use: rows are observations, columns are variables
    cor: bool
        if True, perform a PCA of the correlation matrix of the data,
        otherwise of the covariance matrix. Default: False
    compat: bool
        if True, use the population rather than sample covariance/correlation
        estimate for compatibility with `princomp` in R

    Returns
    -------
    PCAResult
        a summary of the results of PCA
    """
    means = np.mean(X, axis=0)
    X = X - means
    if cor:
        cov = np.corrcoef(X, rowvar=False, ddof=1 if not compat else 0)
    else:
        cov = np.cov(X, rowvar=False, ddof=1 if not compat else 0)

    w, v = np.linalg.eig(cov)
    order = np.flip(np.argsort(w))
    w = w[order]
    v = v[:, order]

    return PCAResult(
        n_obs=X.shape[0],
        center=means,
        sdev=np.sqrt(w),
        loadings=v,
        scores=X @ v,
    )
