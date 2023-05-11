from typing import Callable

import numpy as np


def bootstrap(
        sample: np.ndarray,
        resample_fun: Callable[[np.ndarray], np.ndarray],
        n_resamples: int,
        estimator_fun: Callable[[np.ndarray], float]
) -> np.ndarray:
    """Boostrap an estimate from the provided sample using the resampling function

    Using a resampling function offers an ability to handler model-free and model-based bootstrap uniformly.

    Parameters
    ----------
    sample: np.ndarray
        a 1D array of sample values
    resample_fun: Callable[[np.ndarray], np.ndarray]
        the function to generate the next random resample from the original sample
    n_resamples: int
        the number of resamples to use for estimation
    estimator_fun: Callable[[np.ndarray], float]
        the function to calculate the estimate for a resample passed to it

    Returns
    -------
    np.ndarray
        an array of estimates from each of the resamples

    Notes
    -----
    See `scipy.stats.bootstrap` for a library implementation, which does not provide the ability to choose
    a custom resampling function.
    """
    return np.fromiter(
        (estimator_fun(resample_fun(sample)) for _ in range(n_resamples)),
        float
    )
