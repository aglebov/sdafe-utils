from typing import Any, Callable, Optional

import numpy as np
import scipy.optimize
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


def aic(fit: scipy.optimize.OptimizeResult) -> float:
    """AIC estimate based on the results of an MLE fit

    Parameters
    ----------
    fit: scipy.optimize.OptimizeResult
        the result of minimizing the negative log-likelihood

    Returns
    -------
    float
        the AIC estimate of the fit
    """
    return 2 * fit.fun + 2 * len(fit.x)


def bic(fit: scipy.optimize.OptimizeResult, n: int) -> float:
    """BIC estimate based on the results of an MLE fit

    Parameters
    ----------
    fit: scipy.optimize.OptimizeResult
        the result of minimizing the negative log-likelihood
    n: int
        the number of observations used in the fit

    Returns
    -------
    float
        the BIC estimate of the fit
    """
    return 2 * fit.fun + np.log(n) * len(fit.x)
