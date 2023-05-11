import numpy as np
import pandas as pd
import pytest

from sdafe.ch20.mcmc import gelman_diag


@pytest.fixture
def gelman_diag_data():
    return np.stack([pd.read_csv(f'data/x{i}.csv', index_col=0).values for i in range(3)], axis=-1)


def test_gelman_diag(gelman_diag_data):
    est, upper, mpsrf = gelman_diag(gelman_diag_data)
    est_expected = np.array([1.00693873, 1.00132574, 1.00059572, 1.00796062])
    upper_expected = np.array([1.02764179, 1.00515841, 1.00312387, 1.02482746])
    mpsrf_expected = 1.0112809021691374
    np.testing.assert_array_almost_equal(est, est_expected, decimal=8)
    np.testing.assert_array_almost_equal(upper, upper_expected, decimal=8)
    assert np.abs(mpsrf - mpsrf_expected) < 1e-9
