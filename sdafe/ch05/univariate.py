import numpy as np
import scipy.stats as stats
from scipy.special import gamma, beta


def _dged(x: float, nu: float = 2.0) -> float:
    """Density function of a generalised error distribution, as defined in section 5.6 of SDAFE

    With the default shape parameter of 2.0, this becomes the standard normal distribution.

    Parameters
    ----------
    x: float
        the point at which to evaluate the density function
    nu: float
        the shape parameter of the distribution. Default: 2.0

    Returns
    -------
    float
        the value of the density function at `x`
    """
    lnu = np.sqrt(2 ** (-2 / nu) * gamma(1 / nu) / gamma(3 / nu))
    k = nu / lnu / 2 ** (1 + 1 / nu) / gamma(1 / nu)
    return k * np.exp(-1 / 2 * np.abs(x / lnu) ** nu)


def dged(x: float, loc: float = 0.0, scale: float = 1.0, nu: float = 2.0) -> float:
    """Density function of a generalised error distribution with location and scale

    With the default shape parameter of 2.0, this becomes the standard normal distribution.

    Parameters
    ----------
    x: float
        the point at which to evaluate the density function
    loc: float
        the location parameter of the distribution. Default: 0.0
    scale: float
        the scale parameter of the distribution. Default: 1.0
    nu: float
        the shape parameter of the distribution. Default: 2.0

    Returns
    -------
    float
        the value of the density function at `x`
    """
    return _dged((x - loc) / scale, nu) / scale


def _sged_params(nu: float, xi: float) -> tuple[float, float, float]:
    """Calculate parameters of a symmetric GED for transforming it into a skewed GED

    Parameters
    ----------
    nu: float
        the shape parameter that determines the tail weight
    xi: float
        the shape parameter that determines the skewness

    Returns
    -------
    tuple[float, float, float]
        the location, scale and normalising constant
    """
    assert xi > 0, 'xi must be greater than 0'

    lnu = np.sqrt(2 ** (-2 / nu) * gamma(1 / nu) / gamma(3 / nu))

    m1 = 2 ** (1 / nu) * lnu * gamma(2 / nu) / gamma(1 / nu)
    mu = m1 * (xi - 1 / xi)
    sigma = np.sqrt((1 - m1 ** 2) * (xi ** 2 + 1 / xi ** 2) + 2 * m1 ** 2 - 1)
    g = 2 / (xi + 1 / xi)

    return mu, sigma, g


def _dsged(x: float, nu: float = 2.0, xi: float = 1.0) -> float:
    """Density function of a skewed generalised error distribution

    This is a Python implementation of the code in https://rdrr.io/cran/fGarch/src/R/dist-sged.R

    Under the default values of the parameters, this becomes the density of the standard normal distribution.

    Parameters
    ----------
    x: float
        the point at which to evaluate the density function
    nu: float
        the shape parameter that determines the tail weight: higher values of `nu` result in lighter tails.
        Default: 2.0
    xi: float
        the shape parameter that determines the skewness: value of `xi` above 1 result in right-skewness,
        values between 0 and 1 result in left-skewness. Default: 1.0

    Returns
    -------
    float
        the value of the density function at `x`
    """
    mu, sigma, g = _sged_params(nu, xi)

    z = x * sigma + mu
    Xi = xi ** np.sign(z)
    Density = g * dged(z / Xi, nu=nu)

    return Density * sigma


def dsged(x: float, loc: float = 0.0, scale: float = 1.0, nu: float = 2.0, xi: float = 1.0) -> float:
    """Density function of a skewed generalised error distribution with location and scale parameters

    This is a Python implementation of the code in https://rdrr.io/cran/fGarch/src/R/dist-sged.R

    Under the default values of the parameters, this becomes the density of the standard normal distribution.

    Parameters
    ----------
    x: float
        the point at which to evaluate the density function
    loc: float
        the location parameter. Default: 0.0
    scale: float
        the scale parameter. Default: 1.0
    nu: float
        the shape parameter that determines the tail weight: higher values of `nu` result in lighter tails.
        Default: 2.0
    xi: float
        the shape parameter that determines the skewness: value of `xi` above 1 result in right-skewness,
        values between 0 and 1 result in left-skewness. Default: 1.0

    Returns
    -------
    float
        the value of the density function at `x`
    """
    return _dsged((x - loc) / scale, nu=nu, xi=xi) / scale


def _sstd_params(nu: float, xi: float) -> tuple[float, float, float]:
    """Calculate parameters of a symmetric t-distribution for transforming it into a skewed t-distribution

    Parameters
    ----------
    nu: float
        the shape parameter that determines the tail weight
    xi: float
        the shape parameter that determines the skewness

    Returns
    -------
    tuple[float, float, float]
        the location, scale and normalising constant
    """
    assert nu >= 2, 'nu must be greater or equal to 2'
    assert xi > 0, 'xi must be greater than 0'
    m1 = 2 * np.sqrt(nu - 2) / (nu - 1) / beta(0.5, nu / 2)
    mu = m1 * (xi - 1 / xi)
    sigma = np.sqrt((1 - m1 ** 2) * (xi ** 2 + 1 / xi ** 2) + 2 * m1 ** 2 - 1)
    g = 2 / (xi + 1 / xi)
    return mu, sigma, g


def _dsstd(x: float, nu: float, xi: float) -> float:
    """Density function of a skewed t-distribution

    This is a Python implementation of the code in https://github.com/cran/fGarch/blob/master/R/dist-sstd.R

    The transformation to the skewed distribution is only defined when `nu` >= 2. The scale of the resulting
    distribution is adjusted for the degrees of freedom so that its variance is always 1.

    Parameter
    ---------
    x: float
        the point at which to evaluate the density function
    nu: float
        the shape parameter that determines the tail weight
    xi: float
        the shape parameter that determines the skewness

    Returns
    -------
    float
        the value of the density function at `x`
    """
    mu, sigma, g = _sstd_params(nu, xi)

    z = x * sigma + mu
    Xi = xi ** np.sign(z)
    Density = g * stats.t.pdf(z / Xi, scale=np.sqrt((nu - 2) / nu), df=nu)

    return Density * sigma


def dsstd(x: float, mean: float, sd: float, nu: float, xi: float) -> float:
    """Density function of a skewed t-distribution with location and scale parameters

    This is a Python implementation of the code in https://github.com/cran/fGarch/blob/master/R/dist-sstd.R

    The transformation to the skewed distribution is only defined when `nu` >= 2. The scale of the resulting
    distribution is adjusted for the degrees of freedom so that its variance is always 1.

    Parameter
    ---------
    x: float
        the point at which to evaluate the density function
    mean: float
        the location parameter
    sd: float
        the scale parameter
    nu: float
        the shape parameter that determines the tail weight
    xi: float
        the shape parameter that determines the skewness

    Returns
    -------
    float
        the value of the density function at `x`
    """
    return _dsstd(x=(x - mean) / sd, nu=nu, xi=xi) / sd


def _qsstd(p: float, nu: float, xi: float) -> float:
    """Quantile function of a skewed t-distribution

    This is a Python implementation of the code in https://github.com/cran/fGarch/blob/master/R/dist-sstd.R

    The transformation to the skewed distribution is only defined when `nu` >= 2. The scale of the resulting
    distribution is adjusted for the degrees of freedom so that its variance is always 1.

    Parameter
    ---------
    p: float
        the point at which to evaluate the quantile function
    nu: float
        the shape parameter that determines the tail weight
    xi: float
        the shape parameter that determines the skewness

    Returns
    -------
    float
        the quantile corresponding to probability `p`
    """
    mu, sigma, g = _sstd_params(nu, xi)

    pxi = p - (1 / (1 + xi ** 2))
    sig = np.sign(pxi)
    Xi = xi ** sig
    p = (np.heaviside(pxi, 0.5) - sig * p) / (g * Xi)

    # the quantile of the standardised skewed t-distribution
    return (-sig * stats.t.ppf(p, scale=Xi * np.sqrt((nu - 2) / nu), df=nu) - mu) / sigma


def qsstd(p: float, mean: float, sd: float, nu: float, xi: float) -> float:
    """Quantile function of a skewed t-distribution with location and scale parameters

    This is a Python implementation of the code in https://github.com/cran/fGarch/blob/master/R/dist-sstd.R

    The transformation to the skewed distribution is only defined when `nu` >= 2. The scale of the resulting
    distribution is adjusted for the degrees of freedom so that its variance is always 1.

    Parameter
    ---------
    p: float
        the point at which to evaluate the quantile function
    mean: float
        the location parameter
    sd: float
        the scale parameter
    nu: float
        the shape parameter that determines the tail weight
    xi: float
        the shape parameter that determines the skewness

    Returns
    -------
    float
        the quantile corresponding to probability `p`
    """
    return _qsstd(p, nu, xi) * sd + mean


def rsstd(
        size: int | tuple[int, ...],
        mean: float,
        sd: float,
        nu: float,
        xi: float,
        random_state: np.random.Generator | None = None
) -> np.ndarray:
    """Sample from a skewed t-distribution

    This is a Python implementation of the code in https://github.com/cran/fGarch/blob/master/R/dist-sstd.R

    Parameter
    ---------
    size: int | tuple[int, ...]
        the size (shape) of the sample to draw from this distribution
    mean: float
        the location parameter
    sd: float
        the scale parameter
    nu: float
        the shape parameter that determines the tail weight
    xi: float
        the shape parameter that determines the skewness
    random_state: np.random.Generator | None
        the random number generator to use. If not provided, use the default numpy generator.

    Returns
    -------
    np.ndarray
        the array of the requested shape containing the samples drawn from this distribution
    """
    # Generate Random Deviates:
    weight = xi / (xi + 1 / xi)
    z = stats.uniform.rvs(size=size, loc=-weight, scale=1, random_state=random_state)
    Xi = xi ** np.sign(z)
    r = -np.abs(stats.t.rvs(size=size, scale=np.sqrt((nu - 2) / nu), df=nu, random_state=random_state)) / Xi * np.sign(z)

    # Scale:
    mu, sigma, _ = _sstd_params(nu, xi)
    return (r - mu) / sigma * sd + mean
