import math

import pytest

from financial_analyst.analysis import quant


def test_return_math_and_log():
    hpr = quant.holding_period_return(100, 112, dividends=1.5)
    assert hpr == pytest.approx(0.135)

    annual = quant.annualized_return(hpr, days=90)
    assert annual == pytest.approx(0.6712398, rel=1e-6)

    assert quant.log_return(hpr) == pytest.approx(math.log1p(0.135))


def test_trimmed_and_winsorized_means():
    values = [1, 2, 100, 3, 4]

    assert quant.trimmed_mean(values, percent=40) == pytest.approx(3.0)
    assert quant.winsorized_mean(values, percent=40) == pytest.approx(3.0)


def test_downside_and_variation():
    returns = [0.1, -0.05, 0.02, -0.03]

    downside = quant.target_downside_deviation(returns, target_return=0.0, sample=False)
    assert downside == pytest.approx(0.041231, rel=1e-3)

    cv = quant.coefficient_of_variation(std_dev=2.0, mean=4.0)
    assert cv == pytest.approx(0.5)


def test_probabilistic_moments_and_bayes():
    outcomes = [(0.1, 0.5), (0.2, 0.5)]

    mu = quant.expected_return(outcomes)
    var = quant.probabilistic_variance(outcomes)
    std = quant.probabilistic_standard_deviation(outcomes)

    assert mu == pytest.approx(0.15)
    assert var == pytest.approx(0.0025)
    assert std == pytest.approx(0.05)

    posterior = quant.bayes_rule(prior=0.4, likelihood=0.7, evidence=0.5)
    assert posterior == pytest.approx(0.56)


def test_normal_std_lookup():
    coverage = quant.normal_std_coverage()
    assert 0.95 in coverage
    assert coverage[0.95] == pytest.approx(1.959963984540054)


def test_standard_error_matches_sample():
    values = [1, 2, 3, 4]
    se = quant.standard_error(values, p_var_known=False)
    assert se == pytest.approx(0.6455, rel=1e-3)


def test_linear_regression_basic_line():
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]

    reg = quant.LinearRegression(x, y)

    assert reg.b0 == pytest.approx(0.0)
    assert reg.b1 == pytest.approx(2.0)
    assert reg.r_squared() == pytest.approx(1.0)
    assert reg.standard_error_regression() == pytest.approx(0.0)
