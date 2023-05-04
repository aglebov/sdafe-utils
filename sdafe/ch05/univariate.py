import numpy as np
import pandas as pd
import scipy.stats as stats


def silverman_bw(vals: np.ndarray | pd.Series) -> float:
    """Select bandwidth for a kernel density estimator using Silverman's rule of thumb

    Parameters
    ----------
    vals: np.ndarray | pd.Series
        input data

    Returns
    -------
    float
        the selected bandwidth value

    Notes
    -----
    See https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth
    """
    return 0.9 * min(np.std(vals, ddof=1), stats.iqr(vals) / 1.34) * len(vals) ** (-0.2)
