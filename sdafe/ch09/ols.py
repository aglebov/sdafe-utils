import builtins
from typing import Callable

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults

def reduce_model(
        x: pd.DataFrame,
        y: pd.Series,
        fit_f: Callable[[pd.DataFrame, pd.Series], RegressionResults],
        print_progress: bool = True,
) -> list[str]:
    """A simplistic version of the stepAIC function from R's MASS package

    Parameters
    ----------
    x: pd.DataFrame
        a dataframe of observations of exogenous variables
    y: pd.Series
        a series of observations of the endogenous response variable
    fit_f: Callable[[pd.DataFrame, pd.Series], RegressionResults]
        the function to use for fitting
    print_progress: bool
        whether to print the models considered during the search. Default: True

    Returns
    -------
    list[str]
        the names of the columns of `x` to retain after reducing the model
    """
    def print(*args, **kwargs):
        if print_progress:
            builtins.print(*args, **kwargs)

    while len(x.columns) > 0:
        baseline_fit = fit_f(sm.add_constant(x), y)
        print(f'Variables: {list(x.columns)}')

        res = []
        res.append(baseline_fit.aic)

        for col in x.columns:
            reduced_x = x.drop(col, axis=1)
            fit = fit_f(sm.add_constant(reduced_x), y)
            res.append(fit.aic)

        res = pd.DataFrame(res, columns=['AIC'], index=['<none>'] + list(x.columns))
        res.sort_values(by='AIC', inplace=True)
        print(res)

        if res.iloc[0, 0] < baseline_fit.aic:
            print(f'Dropping variable: {res.index[0]}\n')
            x = x.drop(res.index[0], axis=1)
        else:
            print('Stopping')
            print(f'Remaining variables: {list(x.columns)}')
            return list(x.columns)

    print('No variables left in the model - stopping')
    return list(x.columns)
