import math

from statsmodels.tsa.stattools import adfuller


def adf_test(x):
    """Reproduces the defaults used in the adf.test function from the tseries package in R"""
    n = math.floor((len(x) - 1) ** (1/3))
    return adfuller(x, maxlag=n, autolag=None, regression='ct')
