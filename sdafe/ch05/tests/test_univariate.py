import numpy as np
import pytest
import scipy.stats as stats

from sdafe.ch05.univariate import _dged, dged, _dsged, dsged, _dsstd, dsstd, _qsstd, qsstd, rsstd


def test_dged():
    # with the default value of nu=2.0, the generalised error distribution becomes the normal distribution
    xs = np.linspace(-10, 10, 100)
    np.testing.assert_almost_equal(_dged(xs), stats.norm.pdf(xs))

    # varying shape parameter
    xs = np.array([-5., -2., -1., 0., 1., 2., 5.])
    np.testing.assert_almost_equal(
        _dged(xs, nu=1.0),
        np.array([0.000600564, 0.041794074, 0.171909492, 0.707106781, 0.171909492, 0.041794074, 0.000600564]),
    )
    np.testing.assert_almost_equal(
        _dged(xs, nu=1.5),
        np.array([6.448863e-5, 5.000549e-2, 2.145872e-1, 4.759667e-1, 2.145872e-1, 5.000549e-2, 6.448863e-5]),
    )
    np.testing.assert_almost_equal(
        _dged(xs, nu=2.5),
        np.array([3.152869e-9, 5.542415e-2, 2.601287e-1, 3.625618e-1, 2.601287e-1, 5.542415e-2, 3.152869e-9]),
    )
    np.testing.assert_almost_equal(
        _dged(xs, nu=5),
        np.array([1.432540e-82, 4.557235e-2, 2.921119e-1, 3.101535e-1, 2.921119e-1, 4.557235e-2, 1.432540e-82]),
    )

    # location and scale
    xs = np.linspace(-10, 10, 100)
    nus = np.array([1.0, 1.5, 2.0, 2.5, 5.0])
    locs = np.linspace(-10, 10, 10)
    scales = np.linspace(0.5, 10, 10)
    for nu in nus:
        for loc in locs:
            for scale in scales:
                np.testing.assert_almost_equal(dged(xs, loc, scale, nu), _dged((xs - loc) / scale, nu) / scale)


def test_dsged():
    xs = np.linspace(-10, 10, 100)

    with pytest.raises(AssertionError):
        _dsged(xs, 2.0, 0.0)

    with pytest.raises(AssertionError):
        _dsged(xs, 2.0, -1.0)

    # with the default values of nu=2.0 and xi=1, the skewed GED becomes the normal distribution
    np.testing.assert_almost_equal(_dsged(xs), stats.norm.pdf(xs))

    # with the default value of xi=1, the skewed GED becomes the symmetric GED
    nus = np.array([1.0, 1.5, 2.0, 2.5, 5.0])
    for nu in nus:
        np.testing.assert_almost_equal(_dsged(xs, nu), _dged(xs, nu))

    # varying shape parameters
    xs = np.array([-5., -2., -1., 0., 1., 2., 5.])
    np.testing.assert_almost_equal(
        _dsged(xs, nu=1.0, xi=0.5),
        np.array([2.250252e-03, 4.956928e-02, 1.389547e-01, 3.895234e-01, 2.682229e-01, 4.343640e-03, 1.844709e-08]),
    )
    np.testing.assert_almost_equal(
        _dsged(xs, nu=1.0, xi=0.8),
        np.array([0.0013342685, 0.0470015459, 0.1540760419, 0.5050775719, 0.1987622258, 0.0310937398, 0.0001190393]),
    )
    np.testing.assert_almost_equal(
        _dsged(xs, nu=1.0, xi=1.2),
        np.array([0.0001740068, 0.0334765394, 0.1932555558, 0.5293376143, 0.1566710327, 0.0463708073, 0.0012022991]),
    )
    np.testing.assert_almost_equal(
        _dsged(xs, nu=1.0, xi=1.5),
        np.array([1.190364e-05, 1.921579e-02, 2.254159e-01, 4.346781e-01, 1.455163e-01, 4.871421e-02, 1.827627e-03]),
    )
    np.testing.assert_almost_equal(
        _dsged(xs, nu=3.0, xi=0.5),
        np.array([5.310534e-07, 6.977945e-02, 2.232375e-01, 3.388971e-01, 3.576802e-01, 3.139871e-03, 5.872924e-119]),
    )
    np.testing.assert_almost_equal(
        _dsged(xs, nu=3.0, xi=0.8),
        np.array([9.181066e-10, 6.375712e-02, 2.492290e-01, 3.421811e-01, 3.030185e-01, 4.000457e-02, 2.203414e-22]),
    )
    np.testing.assert_almost_equal(
        _dsged(xs, nu=3.0, xi=1.2),
        np.array([3.880654e-20, 4.329940e-02, 2.969849e-01, 3.422869e-01, 2.528447e-01, 6.259477e-02, 2.942021e-10]),
    )

    # location and scale
    xs = np.linspace(-10, 10, 100)
    nus = np.array([1.0, 1.5, 2.0, 2.5, 5.0])
    xis = np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    locs = np.linspace(-10, 10, 10)
    scales = np.linspace(0.5, 10, 10)
    for nu in nus:
        for xi in xis:
            for loc in locs:
                for scale in scales:
                    np.testing.assert_almost_equal(
                        dsged(xs, loc, scale, nu, xi),
                        _dsged((xs - loc) / scale, nu, xi) / scale
                    )


def test_dsstd():
    xs = np.linspace(-10, 10, 100)

    with pytest.raises(AssertionError):
        _dsstd(xs, 2.5, 0.0)

    with pytest.raises(AssertionError):
        _dsstd(xs, 2.5, -1.0)

    with pytest.raises(AssertionError):
        _dsstd(xs, 1.9, 0.5)

    with pytest.raises(AssertionError):
        _dsstd(xs, 1.9, 0.5)

    # with xi=1, the skewed distribution becomes the symmetric distribution with variance normalised to 1
    np.testing.assert_almost_equal(_dsstd(xs, nu=2.5, xi=1.0), stats.t.pdf(xs, df=2.5, scale=np.sqrt(0.2)))

    # varying shape parameters
    xs = np.array([-5., -2., -1., 0., 1., 2., 5.])
    np.testing.assert_almost_equal(
        _dsstd(xs, nu=2.5, xi=0.5),
        np.array([0.0016184995, 0.0217649030, 0.0951235224, 0.6353118907, 0.0434652450, 0.0012095409, 0.0000266416]),
    )
    np.testing.assert_almost_equal(
        _dsstd(xs, nu=2.5, xi=0.8),
        np.array([0.0012709874, 0.0209752710, 0.1100265431, 0.7432366803, 0.1142374578, 0.0105461174, 0.0003744146]),
    )
    np.testing.assert_almost_equal(
        _dsstd(xs, nu=2.5, xi=1.2),
        np.array([0.0004459033, 0.0118726551, 0.1166336062, 0.7592020663, 0.1117849351, 0.0205658218, 0.0012038465]),
    )
    np.testing.assert_almost_equal(
        _dsstd(xs, nu=4.0, xi=0.5),
        np.array([2.102933e-03, 4.047229e-02, 1.437115e-01, 4.451009e-01, 3.116875e-01, 3.547366e-03, 1.130007e-05]),
    )
    np.testing.assert_almost_equal(
        _dsstd(xs, nu=4.0, xi=0.8),
        np.array([0.0013807340, 0.0389297571, 0.1680654473, 0.5013550239, 0.2268821455, 0.0240851707, 0.0003043145]),
    )
    np.testing.assert_almost_equal(
        _dsstd(xs, nu=4.0, xi=1.2),
        np.array([0.0003746226, 0.0262232648, 0.2199469498, 0.5086793672, 0.1717574816, 0.0383635279, 0.0012786764]),
    )

    # location and scale
    xs = np.linspace(-10, 10, 100)
    nus = np.array([2.0, 2.5, 3.0, 4.0, 10.0])
    xis = np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    locs = np.linspace(-10, 10, 10)
    scales = np.linspace(0.5, 10, 10)
    for nu in nus:
        for xi in xis:
            for loc in locs:
                for scale in scales:
                    np.testing.assert_almost_equal(
                        dsstd(xs, loc, scale, nu, xi),
                        _dsstd((xs - loc) / scale, nu, xi) / scale
                    )


def test_qsstd():
    xs = np.linspace(0.001, 0.999, 100)

    with pytest.raises(AssertionError):
        _qsstd(xs, 2.5, 0.0)

    with pytest.raises(AssertionError):
        _qsstd(xs, 2.5, -1.0)

    with pytest.raises(AssertionError):
        _qsstd(xs, 1.9, 0.5)

    with pytest.raises(AssertionError):
        _qsstd(xs, 1.9, 0.5)

    # with xi=1, the skewed distribution becomes the symmetric distribution with variance normalised to 1
    np.testing.assert_almost_equal(_qsstd(xs, nu=2.5, xi=1.0), stats.t.ppf(xs, df=2.5, scale=np.sqrt(0.2)))

    # varying shape parameters
    ps = np.array([0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999])
    np.testing.assert_almost_equal(
        _qsstd(ps, nu=2.5, xi=0.5),
        np.array([-8.7736396, -3.1220148, -0.7607985, -0.3255216, 0.1944129, 0.5021759, 0.6111326, 0.9983235, 1.8246346]),
    )
    np.testing.assert_almost_equal(
        _qsstd(ps, nu=2.5, xi=0.8),
        np.array([-7.60128400, -2.81740416, -0.79375786, -0.40587126, 0.09500586, 0.48118640, 0.71706417, 1.83691693, 4.41016031]),
    )
    np.testing.assert_almost_equal(
        _qsstd(ps, nu=2.5, xi=1.2),
        np.array([-4.73060173, -1.93922420, -0.72895409, -0.47727511, -0.07955396, 0.41480084, 0.79315980, 2.75634718, 7.38939082]),
    )
    np.testing.assert_almost_equal(
        _qsstd(ps, nu=5.0, xi=0.5),
        np.array([-6.3322671, -3.3653477, -1.2265018, -0.6399814, 0.2078629, 0.7731250, 0.9704590, 1.4711742, 2.0693399]),
    )
    np.testing.assert_almost_equal(
        _qsstd(ps, nu=5.0, xi=0.8),
        np.array([-5.38880861, -2.97061394, -1.19328578, -0.68714363, 0.09431277, 0.73403886, 1.07591575, 2.17835301, 3.61947147]),
    )
    np.testing.assert_almost_equal(
        _qsstd(ps, nu=5.0, xi=1.2),
        np.array([-3.79159545, -2.25679263, -1.08853987, -0.73029454, -0.07856224, 0.69188963, 1.18605174, 2.91241892, 5.25529060]),
    )

    # location and scale
    ps = np.array([0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999])
    nus = np.array([2.0, 2.5, 3.0, 4.0, 10.0])
    xis = np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    locs = np.linspace(-10, 10, 10)
    scales = np.linspace(0.5, 10, 10)
    for nu in nus:
        for xi in xis:
            for loc in locs:
                for scale in scales:
                    np.testing.assert_almost_equal(
                        qsstd(ps, loc, scale, nu, xi),
                        _qsstd(ps, nu, xi) * scale + loc
                    )


def test_rsstd():
    rng = np.random.default_rng(12345)
    n = 50_000_000
    s = rsstd(n, 1.0, 2.0, 5.0, 2.0, random_state=rng)
    bins = np.linspace(-10, 10, 100)
    hist = np.histogram(s, bins)
    mids = (hist[1][:-1] + hist[1][1:]) / 2
    widths = hist[1][1:] - hist[1][:-1]
    diffs = hist[0] - dsstd(mids, 1, 2, 5, 2) * n * widths
    assert np.max(np.abs(diffs)) / n < 1e-3
