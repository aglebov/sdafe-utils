import numpy as np


def matrix_sqrt_inv(A: np.ndarray) -> np.ndarray:
    """Calculate the inverse square root of a matrix if defined

    Parameters
    ----------
    A: np.ndarray
        input matrix

    Returns
    -------
    np.ndarray
        the inverse square root matrix

    Notes
    -----
    See https://people.orie.cornell.edu/davidr/SDAFE2/Rscripts/SDAFE2.R
    """
    u, s, vh = np.linalg.svd(A)
    if np.min(s) >= 0:
        return (u / np.sqrt(s)) @ vh
    else:
        raise ValueError('Matrix square root is not defined')


def cov_rows(x: np.ndarray, y: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Covariances between rows of `x` and `y`

    Parameters
    ----------
    x: np.ndarray
        a 2D array of shape (m, k)
    y: np.ndarray
        a 2D array of shape (n, l)
    ddof: int
        the delta degrees of freedom argument to use in the divisor

    Returns
    -------
    np.ndarray
        a 2D array of shape (m, n), where an element at (i, j) is the covariance
        of the i-th row of `x` and the j-th row of `y`
    """
    r1, _ = x.shape
    r2, _ = y.shape
    return np.cov(x, y, ddof=ddof)[:r1, r2:]
