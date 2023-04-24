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
