"""
Quantitative Methods for Financial Analysis
Vincent Doyon | December 2025

Description:
Core quantitative techniques for financial analysis and modeling.
"""

import math
import random
from typing import Dict, Iterable, Tuple

__all__ = [
    "annualized_return",
    "bayes_rule",
    "bootstrap_mean",
    "coefficient_of_variation",
    "correlation",
    "covariance",
    "expected_return",
    "geometric_mean_return",
    "harmonic_mean",
    "holding_period_return",
    "jackknife_mean",
    "LinearRegression",
    "log_return",
    "normal_std_coverage",
    "probabilistic_standard_deviation",
    "probabilistic_variance",
    "roy_safety_first_ratio",
    "standard_deviation",
    "standard_error",
    "target_downside_deviation",
    "trimmed_mean",
    "variance",
    "winsorized_mean",
    "z_score",
]


def holding_period_return(p_start: float, p_end: float, dividends: float = 0.0) -> float:
    """
    Holding period return.
    """
    if p_start <= 0.0:
        raise ValueError("Starting price must be positive.")
    return (p_end - p_start + dividends) / p_start


def annualized_return(hpr: float, days: int, year_basis: int = 365) -> float:
    """
    Annualized return based on a holding period return.
    """
    if days <= 0:
        raise ValueError("Days must be positive.")
    return (1.0 + hpr) ** (year_basis / days) - 1.0


def log_return(hpr: float) -> float:
    """
    Continuously compounded (log) return.
    """
    if hpr <= -1.0:
        raise ValueError("HPR must be greater than -1.")
    return math.log1p(hpr)


def geometric_mean_return(returns: Iterable[float]) -> float:
    """
    Geometric mean of a sequence of simple returns.
    """
    returns = list(returns)
    if not returns:
        raise ValueError("Returns sequence must be non-empty.")

    log_sum = 0.0
    for r in returns:
        if r <= -1.0:
            raise ValueError("Returns must be greater than -1.")
        log_sum += math.log1p(r)

    return math.exp(log_sum / len(returns)) - 1.0


def harmonic_mean(values: Iterable[float]) -> float:
    """
    Harmonic mean of a sequence of positive values.
    """
    values = list(values)
    if not values:
        raise ValueError("Values sequence must be non-empty.")
    if any(v <= 0.0 for v in values):
        raise ValueError("All values must be positive.")

    return len(values) / sum(1.0 / v for v in values)


def trimmed_mean(values: Iterable[float], percent: float) -> float:
    """
    Trimmed mean of a sequence of values.

    Percent is the total percentage of observations to remove,
    split evenly between the lower and upper tails.
    """
    values = sorted(values)
    n = len(values)

    if n == 0:
        raise ValueError("Values sequence must be non-empty.")
    if not (0.0 <= percent < 100.0):
        raise ValueError("Percent must be in the range [0, 100).")

    trim_each_side = int(math.floor((percent / 100.0) * n / 2.0))

    if 2 * trim_each_side >= n:
        raise ValueError("Trimming percentage removes all observations.")

    trimmed = values[trim_each_side : n - trim_each_side]
    return sum(trimmed) / len(trimmed)


def winsorized_mean(values: Iterable[float], percent: float) -> float:
    """
    Winsorized mean of a sequence of values.

    Percent is the total percentage of observations to substitute,
    split evenly between the lower and upper tails.
    """
    values = sorted(values)
    n = len(values)

    if n == 0:
        raise ValueError("Values sequence must be non-empty.")
    if not (0.0 <= percent < 100.0):
        raise ValueError("Percent must be in the range [0, 100).")

    winsor_each_side = int(math.floor((percent / 100.0) * n / 2.0))

    if 2 * winsor_each_side >= n:
        raise ValueError("Winsorization percentage substitutes all observations.")

    low_value = values[winsor_each_side]
    high_value = values[n - winsor_each_side - 1]

    winsorized = (
        [low_value] * winsor_each_side
        + values[winsor_each_side : n - winsor_each_side]
        + [high_value] * winsor_each_side
    )

    return sum(winsorized) / n


def roy_safety_first_ratio(
    expected_return: float,
    target_return: float,
    volatility: float,
) -> float:
    """
    Roy's Safety-First Ratio.

    Measures the number of standard deviations the portfolio's expected
    return exceeds a minimum acceptable return.
    """
    if volatility <= 0.0:
        raise ValueError("Volatility must be positive.")

    return (expected_return - target_return) / volatility


def variance(values: Iterable[float], sample: bool = True) -> float:
    """
    Variance of a sequence of values.

    If sample=True, computes sample variance (ddof=1).
    If sample=False, computes population variance.
    """
    values = list(values)
    n = len(values)

    if n == 0:
        raise ValueError("Values sequence must be non-empty.")
    if sample and n < 2:
        raise ValueError("Sample variance requires at least two observations.")

    mean = sum(values) / n
    ssq = sum((x - mean) ** 2 for x in values)

    return ssq / (n - 1 if sample else n)


def standard_deviation(values: Iterable[float], sample: bool = True) -> float:
    """
    Standard deviation of a sequence of values.
    """
    return math.sqrt(variance(values, sample=sample))


def target_downside_deviation(
    returns: Iterable[float],
    target_return: float = 0.0,
    sample: bool = True,
) -> float:
    """
    Target downside deviation (semi-deviation).

    Measures dispersion of returns falling below a target return.
    """
    returns = list(returns)
    n = len(returns)

    if n == 0:
        raise ValueError("Returns sequence must be non-empty.")

    downside = [(r - target_return) ** 2 for r in returns if r < target_return]

    if not downside:
        return 0.0

    denom = (len(downside) - 1) if sample and len(downside) > 1 else len(downside)
    return math.sqrt(sum(downside) / denom)


def coefficient_of_variation(std_dev: float, mean: float) -> float:
    """
    Coefficient of variation.

    Measures relative dispersion as the ratio of standard deviation
    to the absolute value of the mean.
    """
    if mean == 0.0:
        raise ValueError("Mean must be non-zero to compute coefficient of variation.")

    return std_dev / abs(mean)


def bayes_rule(
    prior: float,
    likelihood: float,
    evidence: float,
) -> float:
    """
    Bayes' rule.

    Computes the posterior probability:
        P(A | B) = P(B | A) * P(A) / P(B)

    Parameters
    ----------
    prior : float
        Prior probability P(A).
    likelihood : float
        Likelihood P(B | A).
    evidence : float
        Marginal probability P(B).

    Returns
    -------
    float
        Posterior probability P(A | B).
    """
    if not (0.0 <= prior <= 1.0):
        raise ValueError("Prior probability must be in [0, 1].")
    if not (0.0 <= likelihood <= 1.0):
        raise ValueError("Likelihood must be in [0, 1].")
    if evidence <= 0.0:
        raise ValueError("Evidence must be positive.")

    posterior = likelihood * prior / evidence

    if posterior < 0.0 or posterior > 1.0:
        raise ValueError("Computed posterior is outside [0, 1]. Check inputs.")

    return posterior


def expected_return(outcomes: Iterable[Tuple[float, float]]) -> float:
    """
    Expected return of a discrete distribution.

    Each outcome is a (return, probability) pair.
    """
    mu = 0.0
    prob_sum = 0.0

    for r, p in outcomes:
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probabilities must be in [0, 1].")
        mu += r * p
        prob_sum += p

    if not math.isclose(prob_sum, 1.0, rel_tol=1e-9):
        raise ValueError("Probabilities must sum to 1.")

    return mu


def probabilistic_variance(outcomes: Iterable[Tuple[float, float]]) -> float:
    """
    Variance of a discrete probabilistic return distribution.
    """
    outcomes = list(outcomes)
    mu = expected_return(outcomes)

    var = 0.0
    for r, p in outcomes:
        var += p * (r - mu) ** 2

    return var


def probabilistic_standard_deviation(
    outcomes: Iterable[Tuple[float, float]]
) -> float:
    """
    Standard deviation of a discrete probabilistic return distribution.
    """
    return math.sqrt(probabilistic_variance(outcomes))


def covariance(
    x: Iterable[float],
    y: Iterable[float],
    sample: bool = True,
) -> float:
    """
    Covariance between two sequences.

    If sample=True, computes sample covariance (ddof=1).
    If sample=False, computes population covariance.
    """
    x = list(x)
    y = list(y)

    if len(x) != len(y):
        raise ValueError("Input sequences must have the same length.")
    n = len(x)

    if n == 0:
        raise ValueError("Input sequences must be non-empty.")
    if sample and n < 2:
        raise ValueError("Sample covariance requires at least two observations.")

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    return cov / (n - 1 if sample else n)


def correlation(
    x: Iterable[float],
    y: Iterable[float],
    sample: bool = True,
) -> float:
    """
    Pearson correlation coefficient between two sequences.
    """
    cov = covariance(x, y, sample=sample)

    std_x = math.sqrt(covariance(x, x, sample=sample))
    std_y = math.sqrt(covariance(y, y, sample=sample))

    if std_x == 0.0 or std_y == 0.0:
        raise ValueError("Correlation is undefined for zero-variance inputs.")

    return cov / (std_x * std_y)


def z_score(value: float, mean: float, std_dev: float) -> float:
    """
    Z-score of a value under a normal distribution.

    Measures how many standard deviations `value` is from the mean.
    """
    if std_dev <= 0.0:
        raise ValueError("Standard deviation must be positive.")

    return (value - mean) / std_dev


def normal_std_coverage() -> Dict[float, float]:
    """
    Standard deviation (z-score) cutoffs for a standard normal distribution.

    Returns a mapping from central probability mass to the corresponding
    two-sided z-score such that:
        P(|Z| <= z) = coverage
    """
    return {
        0.68: 1.0,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }


def standard_error(
    values: Iterable[float],
    population_variance: float | None = None,
    p_var_known: bool = False
) -> float:
    """
    Standard error of the mean.

    Parameters
    ----------
    values : Iterable[float]
        Sample observations.
    p_var_known : bool
        If True, use the supplied population variance.
    population_variance : float | None
        Known population variance (required if p_var_known=True).

    Returns
    -------
    float
        Standard error of the sample mean.
    """
    values = list(values)
    n = len(values)

    if n == 0:
        raise ValueError("Values sequence must be non-empty.")

    if p_var_known:
        if population_variance is None:
            raise ValueError(
                "Population variance must be provided when p_var_known=True."
            )
        if population_variance < 0.0:
            raise ValueError("Population variance must be non-negative.")

        variance = population_variance
    else:
        if n < 2:
            raise ValueError("At least two observations required to estimate variance.")
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)

    return math.sqrt(variance / n)


def jackknife_mean(values: Iterable[float]) -> float:
    """
    Jackknife estimator of the sample mean.

    Computes the leave-one-out means and returns their average.
    """
    values = list(values)
    n = len(values)

    if n < 2:
        raise ValueError("Jackknife requires at least two observations.")

    total = sum(values)
    loo_means = [(total - values[i]) / (n - 1) for i in range(n)]

    return sum(loo_means) / n


def bootstrap_mean(
    values: Iterable[float],
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> float:
    """
    Bootstrap estimator of the sample mean.

    Resamples with replacement and returns the mean of bootstrap means.
    """
    values = list(values)
    n = len(values)

    if n == 0:
        raise ValueError("Values sequence must be non-empty.")
    if n_resamples <= 0:
        raise ValueError("Number of resamples must be positive.")

    rng = random.Random(seed)

    boot_means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    return sum(boot_means) / n_resamples


class LinearRegression:
    """
    Simple Ordinary Least Squares (OLS) regression.

    Model:
        Y = b0 + b1 * X + epsilon
    """

    def __init__(self, x: Iterable[float], y: Iterable[float]):
        self.x = list(x)
        self.y = list(y)

        if len(self.x) != len(self.y):
            raise ValueError("X and Y must have the same number of observations.")
        if len(self.x) < 3:
            raise ValueError("At least three observations are required for OLS inference.")

        self.n = len(self.x)

        # Estimated parameters
        self.b0: float | None = None
        self.b1: float | None = None

        # Fitted values and residuals
        self.y_hat: list[float] | None = None
        self.residuals: list[float] | None = None

        # Fit immediately
        self._fit()

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        x_bar = sum(self.x) / self.n
        y_bar = sum(self.y) / self.n

        s_xx = sum((xi - x_bar) ** 2 for xi in self.x)
        if s_xx == 0.0:
            raise ValueError("X has zero variance; slope is undefined.")

        s_xy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(self.x, self.y))

        self.b1 = s_xy / s_xx
        self.b0 = y_bar - self.b1 * x_bar

        self.y_hat = [self.b0 + self.b1 * xi for xi in self.x]
        self.residuals = [yi - yhi for yi, yhi in zip(self.y, self.y_hat)]

    # ------------------------------------------------------------------
    # Sums of squares
    # ------------------------------------------------------------------

    def sse(self) -> float:
        """Sum of squared errors (residual sum of squares)."""
        return sum(e ** 2 for e in self.residuals)

    def sst(self) -> float:
        """Total sum of squares."""
        y_bar = sum(self.y) / self.n
        return sum((yi - y_bar) ** 2 for yi in self.y)

    def ssr(self) -> float:
        """Regression sum of squares."""
        return self.sst() - self.sse()

    # ------------------------------------------------------------------
    # Goodness of fit
    # ------------------------------------------------------------------

    def r_squared(self) -> float:
        """Coefficient of determination."""
        return self.ssr() / self.sst()

    def adjusted_r_squared(self) -> float:
        """Adjusted R-squared."""
        return 1.0 - (1.0 - self.r_squared()) * (self.n - 1) / (self.n - 2)

    # ------------------------------------------------------------------
    # Variance and standard errors
    # ------------------------------------------------------------------

    def residual_variance(self) -> float:
        """Unbiased estimator of error variance."""
        return self.sse() / (self.n - 2)

    def standard_error_regression(self) -> float:
        """Standard error of the regression (SER)."""
        return math.sqrt(self.residual_variance())

    def standard_error_slope(self) -> float:
        """Standard error of b1."""
        x_bar = sum(self.x) / self.n
        s_xx = sum((xi - x_bar) ** 2 for xi in self.x)
        return math.sqrt(self.residual_variance() / s_xx)

    def standard_error_intercept(self) -> float:
        """Standard error of b0."""
        x_bar = sum(self.x) / self.n
        s_xx = sum((xi - x_bar) ** 2 for xi in self.x)
        return math.sqrt(
            self.residual_variance() * (1.0 / self.n + x_bar ** 2 / s_xx)
        )

    # ------------------------------------------------------------------
    # Hypothesis testing
    # ------------------------------------------------------------------

    def t_stat_slope(self) -> float:
        """t-statistic for H0: b1 = 0."""
        return self.b1 / self.standard_error_slope()

    def t_stat_intercept(self) -> float:
        """t-statistic for H0: b0 = 0."""
        return self.b0 / self.standard_error_intercept()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x_new: float) -> float:
        """Point prediction for a new X value."""
        return self.b0 + self.b1 * x_new
