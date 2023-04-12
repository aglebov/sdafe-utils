import numpy as np

def matrix_sqrt_inv(A: np.array) -> np.array:
    """Calculate the inverse square root of a matrix if defined

    Parameters
    ----------
    A: np.array
        input matrix

    Returns
    -------
    np.array
        the inverse square root matrix
    """
    u, s, vh = np.linalg.svd(A)
    if np.min(s) >= 0:
        return (u / np.sqrt(s)) @ vh
    else:
        raise ValueError('Matrix square root is not defined')
