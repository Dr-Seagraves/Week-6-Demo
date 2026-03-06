# Regression Diagnostics Demo

**QM 2023 — Statistics II / Data Analytics**  
Week 6: Regression diagnostics, assumption testing, and robust standard errors.

## Contents

| File | Description |
|------|-------------|
| `week6_diagnostics_simulated.ipynb` | Notebook: simulated Firm Size / Firm Age → Revenue data with heteroskedasticity. Step-by-step diagnostics, plots, and fixes. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook week6_diagnostics_simulated.ipynb
# or
jupyter lab week6_diagnostics_simulated.ipynb
```

## What You'll Learn

1. **Diagnostic plots**: Residuals vs Fitted, Q-Q, Scale-Location, Residuals vs Leverage
2. **Breusch-Pagan test** for heteroskedasticity
3. **Robust standard errors (HC3)** when homoskedasticity fails
4. **VIF** for multicollinearity

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, statsmodels (see `requirements.txt`)
