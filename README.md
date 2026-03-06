# Regression Diagnostics Demo

**QM 2023 — Statistics II / Data Analytics**  
Week 6: Regression diagnostics, assumption testing, and robust standard errors.

## Contents

| File | Description |
|------|-------------|
| `week6_diagnostics_simulated.ipynb` | Notebook: simulated Firm Size / Firm Age → Revenue data with heteroskedasticity. Step-by-step diagnostics, plots, and fixes. |

## Setup for GitHub Codespaces

### 1. Install Required Packages

Open a terminal in your Codespace and run:

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, matplotlib, and statsmodels.

### 2. Run the Notebook

Simply open `week6_diagnostics_simulated.ipynb` in VS Code (it will be listed in the file tree). The Jupyter extension is pre-installed in Codespaces.

Click the **Run All** button at the top of the notebook, or run cells individually by clicking the play button next to each code cell.

## What You'll Learn

1. **Diagnostic plots**: Residuals vs Fitted, Q-Q, Scale-Location, Residuals vs Leverage
2. **Breusch-Pagan test** for heteroskedasticity
3. **Robust standard errors (HC3)** when homoskedasticity fails
4. **VIF** for multicollinearity

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, statsmodels (see `requirements.txt`)
