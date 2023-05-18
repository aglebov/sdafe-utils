import numpy as np

import scipy.stats as stats

from statsmodels.graphics.tsaplots import _plot_corr, plot_acf, _prepare_data_corr_plot
from statsmodels.graphics import utils
from statsmodels.tools.validation.validation import array_like, bool_like
from statsmodels.tsa.stattools import ccovf


def ccf(x, y, adjusted=True, fft=True, *, nlags=None, alpha=None):
    """
    The cross-correlation function.

    Parameters
    ----------
    x, y : array_like
        The time series data to use in the calculation.
    adjusted : bool
        If True, then denominators for cross-correlation are n-k, otherwise n.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    nlags : int, optional
        Number of lags to return cross-correlations for. If not provided,
        the number of lags equals len(x).
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x)).

    Returns
    -------
    ndarray
        The cross-correlation function of x and y: the element at index k
        is the correlation between {x[k], x[k+1], ..., x[n]} and {y[0], y[1], ..., y[m-k]},
        where n and m are the lengths of x and y, respectively.
    confint : ndarray, optional
        Confidence intervals for the CCF at lags 0, 1, ..., nlags-1. Shape
        (nlags, 2). Returned if alpha is not None.

    Notes
    -----
    If adjusted is True, the denominator for the cross-correlation is adjusted.
    """
    x = array_like(x, "x")
    y = array_like(y, "y")
    adjusted = bool_like(adjusted, "adjusted")
    fft = bool_like(fft, "fft", optional=False)

    cvf = ccovf(x, y, adjusted=adjusted, demean=True, fft=fft)
    ret = cvf / (np.std(x) * np.std(y))
    ret = ret[:nlags]

    if alpha is not None:
        interval = stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(len(x))
        confint = np.vstack([ret - interval, ret + interval]).T
        return ret, confint
    else:
        return ret


def plot_ccf(
        x,
        y,
        *,
        ax=None,
        lags=None,
        invert_lags=False,
        alpha=0.05,
        use_vlines=True,
        adjusted=False,
        fft=False,
        title="Cross-correlation",
        auto_ylims=False,
        vlines_kwargs=None,
        **kwargs,
):
    """
    Plot the cross-correlation function

    Plots lags on the horizontal and the correlations on vertical axis.

    Parameters
    ----------
    x, y : array_like
        Arrays of time-series values
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    invert_lags: bool, optional
        If True, use negative values of lags when plotting.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned. If None, no confidence intervals are plotted.
    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    adjusted : bool
        If True, then denominators for cross-covariance are n-k, otherwise n
    fft : bool, optional
        If True, computes the CCF via FFT.
    title : str, optional
        Title to place on plot.  Default is 'Autocorrelation'
    auto_ylims : bool, optional
        If True, adjusts automatically the y-axis limits to ACF values.
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise, `fig` is returned.

    See Also
    --------
    See notes and references for statsmodels.graphics.tsaplots.plot_acf
    """
    fig, ax = utils.create_mpl_ax(ax)

    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, True)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs

    if invert_lags:
        lags = -lags

    ccf_res = ccf(x, y, adjusted=adjusted, fft=fft, alpha=alpha, nlags=nlags + 1)
    ccf_xy = ccf_res[0]
    confint = ccf_res[1] if len(ccf_res) > 1 else None

    _plot_corr(
        ax,
        title,
        ccf_xy,
        confint,
        lags,
        irregular,
        use_vlines,
        vlines_kwargs,
        auto_ylims=auto_ylims,
        **kwargs,
    )

    return fig


def plot_accf(
        x,
        *,
        fig=None,
        lags=None,
        alpha=0.05,
        use_vlines=True,
        adjusted=False,
        fft=False,
        missing="none",
        zero=True,
        auto_ylims=False,
        bartlett_confint=True,
        vlines_kwargs=None,
        **kwargs,
):
    """
    Plot matrix of auto- and cross-correlations

    Plots lags on the horizontal and the correlations on vertical axis.

    Parameters
    ----------
    x : array_like
        2D array of time-series values: rows are observations, columns are variables
    fig : Matplotlib figure instance, optional
        If given, this figure is used to plot in instead of a new figure being
        created.
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals .
        If None, no confidence intervals are plotted.
    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    adjusted : bool
        If True, then denominators for autocovariance are n-k, otherwise n
    fft : bool, optional
        If True, computes the ACF via FFT.
    missing : str, optional
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how
        the NaNs are to be treated.
    zero : bool, optional
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.
    auto_ylims : bool, optional
        If True, adjusts automatically the y-axis limits to ACF values.
    bartlett_confint : bool, default True
        If True, use Bartlett's formula to calculate confidence intervals
        in autocorrelation plots. See the description of ``plot_acf`` for
        details.
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise, `fig` is returned.

    See Also
    --------
    See notes and references for statsmodels.graphics.tsaplots
    """
    from statsmodels.tools.data import _is_using_pandas

    array_like(x, "x", ndim=2)
    m = x.shape[1]

    fig = utils.create_mpl_fig(fig)
    gs = fig.add_gridspec(m, m)

    if _is_using_pandas(x, None):
        varnames = list(x.columns)

        def get_var(i):
            return x.iloc[:, i]
    else:
        varnames = [f'x[{i}]' for i in range(m)]

        def get_var(i):
            return x[:, i]

    for i in range(m):
        for j in range(m):
            ax = fig.add_subplot(gs[i, j])
            if i == j:
                plot_acf(
                    get_var(i),
                    ax=ax,
                    title=f'ACF({varnames[i]})',
                    lags=lags,
                    alpha=alpha,
                    use_vlines=use_vlines,
                    adjusted=adjusted,
                    fft=fft,
                    missing=missing,
                    zero=zero,
                    auto_ylims=auto_ylims,
                    bartlett_confint=bartlett_confint,
                    vlines_kwargs=vlines_kwargs,
                    **kwargs,
                )
            else:
                plot_ccf(
                    get_var(i),
                    get_var(j),
                    ax=ax,
                    title=f'CCF({varnames[i]}, {varnames[j]})',
                    lags=lags,
                    invert_lags=i > j,
                    alpha=alpha,
                    use_vlines=use_vlines,
                    adjusted=adjusted,
                    fft=fft,
                    auto_ylims=auto_ylims,
                    vlines_kwargs=vlines_kwargs,
                    **kwargs,
                )

    return fig
