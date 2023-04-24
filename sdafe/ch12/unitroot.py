import math

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller


def adf_test(sample: np.ndarray | pd.Series):
    """Perform the augmented Dickey-Fuller test using the default from the R's adf.test function

    Parameters
    ----------
    sample: np.ndarray | pd.Series
        the sample to perform the test on

    Returns
    -------
    Tuple[float, float, int, int, Dict[str, float]]
        the results of the test:
           * the test statistic
           * the p-value
           * the number of lags used
           * the number of observations
           * the critical values
    """
    n = math.floor((len(sample) - 1) ** (1/3))
    return adfuller(sample, maxlag=n, autolag=None, regression='ct')
