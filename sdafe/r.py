"""Convenience utilities for using R functions from Python """
from typing import Any

import numpy as np
import pandas as pd

import rpy2.rlike.container as rlc
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri, default_converter


np_cv_rules = default_converter + numpy2ri.converter + pandas2ri.converter


def tl(d: Any = None, **kwargs):
    """Resursively converts a value to a `rpy2.rlike.container.TaggedList`

    Examples
    --------
    >>> l = tl({'mean': 1.2, 'sd': 2.5})
    >>> l
    [1.2, 2.5]
    >>> l.tags
    ('mean', 'sd')
    >>> l = tl([2.0, 5.0])
    >>> l
    [2.0, 5.0]
    >>> l.tags
    (None, None)
    >>> l = tl([{'first': 1.0, 'second': 2.0}, {'third': 3.0}])
    >>> l
    [[1.0, 2.0], [3.0]]
    >>> l[0].tags
    ('first', 'second')
    >>> l[1].tags
    ('third')
    """
    if d is None:
        return tl(kwargs)
    elif isinstance(d, dict):
        tags = list(d.keys())
        vals = [tl(d[k]) for k in tags]
        return rlc.TaggedList(vals, tags=tags)
    elif isinstance(d, list):
        return rlc.TaggedList([tl(v) for v in d])
    elif isinstance(d, np.float64):
        return float(d)
    else:
        return d


def sv(*vs) -> robjects.StrVector:
    """Converts a list of strings to a `rpy2.robjects.StrVector`"""
    return robjects.StrVector(vs)


def fv(*vs):
    """Coverts a list of numbers to a `rpy2.robjects.FloatVector`"""
    return robjects.FloatVector(vs)


def el(l: rlc.TaggedList, name: str) -> Any:
    """Extract an element from a `rpy2.rlike.container.TaggedList`

    Parameters
    ----------
    l: rlc.TaggedList
        a TaggedList of values
    name: str
        the name of the element to extract

    Returns
    -------
    Any
        the value from the list corresponding to the given tag
    """
    return l[l.names.index(name)]


def py2rpy(obj: pd.DataFrame | np.ndarray):
    with np_cv_rules.context():
        return np_cv_rules.py2rpy(obj)
