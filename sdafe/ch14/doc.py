import numpy as np
import pandas as pd
import scipy.stats as stats


def doc_test(A: np.ndarray, m: int) -> pd.DataFrame:
    """Test for dynamic orthogonal components

    Parameters
    ----------
    A: np.ndarray
        the matrix of time series: rows are observations, columns are variables
    m: int
        the number of lags to test

    Returns
    -------
    pd.DataFrame
        a dataframe with three columns:

           - `Q(m)`: the test statistic
           - `d.f.`: the number of degrees of freedom
           - `p-value`: the p-value

    Notes
    -----
    See https://people.orie.cornell.edu/davidr/SDAFE2/Rscripts/SDAFE2.R
    """
    n, k = A.shape
    res = []

    q = n * np.sum(np.tril(np.corrcoef(A, rowvar=False), k=-1) ** 2)
    df = k * (k - 1) / 2  # lower triangular excluding the diagonal
    p = 1 - stats.chi2.cdf(q, df=df)
    res.append((q, df, p))

    for j in range(1, m + 1):
        # corrcoef gives us correlation between all combinations of variables in x and y,
        # but we only need the correlations of variables in x with variables in y,
        # hence select a submatrix
        ccf = np.corrcoef(A[j:, :], A[:-j, :], rowvar=False)[:k, k:]
        q += n * (n + 2) * np.sum((ccf - np.diag(np.diag(ccf))) ** 2) / (n - j)  # zero the diagonal elements
        df += k * (k - 1)  # excluding the diagonal elements
        p = 1 - stats.chi2.cdf(q, df=df)
        res.append((q, df, p))

    return pd.DataFrame(res, columns=['Q(m)', 'd.f.', 'p-value'], index=pd.Index(range(m + 1), name='m'))
