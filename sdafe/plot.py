from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def abs_diff_heatmap(df1: pd.DataFrame, df2: pd.DataFrame, ax: Optional[plt.Axes] = None):
    """Plot heatmap of element-wise absolute differences between two dataframes

    Parameters
    ----------
    df1: pd.DataFrame
        the first dataframe
    df2: pd.DataFrame
        the second dataframe
    ax: Optional[plt.Axes]
        the subplot to use, if this is not provided, a new figure will be created
    """
    pd.testing.assert_index_equal(df1.index, df2.index)
    pd.testing.assert_index_equal(df1.columns, df2.columns)

    abs_diff = (df1 - df2).abs()

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(abs_diff)

    ax.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
    ax.set_yticks(np.arange(len(df1.index)), labels=df1.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(df1.index)):
        for j in range(len(df1.columns)):
            ax.text(j, i, f'{abs_diff.iloc[i, j]:.2f}', ha="center", va="center", color="w")


def plot_qq_norm(ax: plt.Axes, sample: np.ndarray | pd.Series) -> None:
    """QQ plot with a regression line through the 25% and 75% quantiles

    Parameters
    ----------
    ax: plt.Axes
        the subplot to use
    sample: np.ndarray | pd.Series
        the sample to plot
    """
    qs = np.array([0.25, 0.75])

    stats.probplot(sample, dist="norm", plot=ax, fit=False)

    # draw a regression line through 0.25 and 0.75 quantiles
    coord = lambda q: (stats.norm.ppf(q), np.quantile(sample, q))
    ax.axline(coord(qs[0]), coord(qs[1]), color='red')


def plot_qq_t(ax: plt.Axes, sample: np.ndarray | pd.Series, df: int) -> None:
    """QQ plot with a regression line through the 25% and 75% quantiles

    Parameters
    ----------
    ax: plt.Axes
        the subplot to use
    sample: np.ndarray | pd.Series
        the sample to plot
    df: int
        the degrees of freedom of the t-distribution to use
    """
    qs = np.array([0.25, 0.75])

    stats.probplot(sample, dist="t", plot=ax, fit=False, sparams=(df,))

    # draw a regression line through 0.25 and 0.75 quantiles
    coord = lambda q: (stats.t.ppf(q, df=df), np.quantile(sample, q))
    ax.axline(coord(qs[0]), coord(qs[1]), color='red')
