from typing import Any, Callable, Optional

import numpy as np
from statsmodels.tools.numdiff import approx_hess1


def mle_se(x: Any, loglik_f: Callable[[Any], float], epsilon: Optional[float] = 1e-6) -> np.ndarray:
    """Calculate the standard errors for MLE parameter estimates using Fisher information

    Parameters
    ----------
    x: Any
        the coordinates of the optimum of the log-likelihood function
    loglik_f: Callable[[Any], float]
        the log-likelihood function
    epsilon: float
        the epsilon to use when evaluating the derivatives. Default: 1e-6

    Returns
    -------
    np.ndarray
        the array of standard errors
    """
    return np.sqrt(np.diag(np.linalg.inv(approx_hess1(x, loglik_f, epsilon=epsilon))))
