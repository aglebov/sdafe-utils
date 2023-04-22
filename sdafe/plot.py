from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
