# financial-analyst

Toolkit for financial analysis and modeling, with reusable quantitative helpers you can drop into research notebooks or production code.

## Features
- Return math: holding-period, annualized, log returns, geometric/harmonic means.
- Robust stats: trimmed/winsorized means, downside deviation, coefficient of variation.
- Probabilistic tools: Bayes' rule, expected return, variance/standard deviation for discrete outcomes.
- Regression: lightweight OLS helper for quick, single-factor fits.

## Installation
From PyPI (when published):
```bash
pip install financial-analyst
```

From source (editable with dev extras):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart
```python
from financial_analyst.analysis import quant

# Holding-period and annualized return
hpr = quant.holding_period_return(100, 112, dividends=1.5)  # 13.5%
annual = quant.annualized_return(hpr, days=90)

# Simple OLS regression
reg = quant.LinearRegression([1, 2, 3], [2, 4, 6])
slope = reg.b1  # ~2.0
r2 = reg.r_squared()
```

## Development
- Requires Python 3.10+.
- Install dev tooling: `pip install -e ".[dev]"`.
- Run tests: `pytest`.
- Lint/format suggestions: `ruff check src tests`.

## License
MIT License © 2025 Vincent Doyon
