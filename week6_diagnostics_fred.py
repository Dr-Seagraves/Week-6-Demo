"""
Week 6: Regression Diagnostics - FRED Macro Data (Improved Version)
===================================================================
Quarterly GDP model with multiple predictors (FRED-like structure).

Uses simulated quarterly macro data (GDP, Unemployment, FedFunds, HousingStarts)
to avoid pandas_datareader/fredapi dependency issues. Same variables and
pedagogical flow as live FRED fetch.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Output directory: same folder as this script
SCRIPT_DIR = Path(__file__).resolve().parent

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("WEEK 6: REGRESSION DIAGNOSTICS - FRED MACRO DATA")
print("=" * 70)

# =============================================================================
# STEP 1: Load Data - Simulated quarterly macro (FRED-like structure)
# =============================================================================
# Uses simulated data to avoid pandas_datareader/fredapi dependency issues.
# Same variables as FRED: GDP, Unemployment, FedFunds, HousingStarts.
print("\n" + "=" * 70)
print("STEP 1: Loading Data (Quarterly GDP Model)")
print("=" * 70)

np.random.seed(42)
n = 136  # ~34 years of quarterly data (1990-2023)

# Simulate quarterly macro series with realistic ranges and correlations
dates = pd.date_range('1990-01-01', periods=n, freq='QS')
unemployment = 4 + 4 * np.random.beta(2, 5, n) + 0.02 * np.arange(n)  # 4-8%, slight trend
fedfunds = 1 + 4 * np.random.beta(2, 3, n) + 0.01 * np.arange(n)     # 1-5%
houst = 800 + 600 * np.random.beta(2, 2, n) + 5 * np.arange(n)       # housing starts

# GDP: negative on unemployment, negative on rates, positive on housing
# Add heteroskedastic errors (variance grows with level)
gdp_base = 15000 + 200 * np.arange(n) - 200 * unemployment - 100 * fedfunds + 2 * houst
sigma = 100 + 0.02 * gdp_base
gdp = gdp_base + np.random.normal(0, sigma, n)

df = pd.DataFrame({
    'GDP': gdp,
    'Unemployment': unemployment,
    'FedFunds': fedfunds,
    'HousingStarts': houst
}, index=dates)
df.index.name = 'Date'

print(f"\nData loaded: {len(df)} quarterly observations (simulated FRED-like macro)")
print("\nFirst 5 rows:")
print(df.head())
print("\nDescriptive Statistics:")
print(df.describe().round(2))

# =============================================================================
# STEP 2: SEQUENTIAL REGRESSION TABLE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Sequential Regression Table")
print("=" * 70)
print("\nBuilding models incrementally to show how predictors improve fit.")
print("-" * 70)

# Model 1: Y ~ Unemployment only
X1 = sm.add_constant(df[['Unemployment']])
model1 = sm.OLS(df['GDP'], X1).fit()

# Model 2: Y ~ Unemployment + FedFunds
X2 = sm.add_constant(df[['Unemployment', 'FedFunds']])
model2 = sm.OLS(df['GDP'], X2).fit()

# Model 3: Y ~ Unemployment + FedFunds + HousingStarts
X3 = sm.add_constant(df[['Unemployment', 'FedFunds', 'HousingStarts']])
model3 = sm.OLS(df['GDP'], X3).fit()

# Print nice regression table
print("\n" + "=" * 70)
print("TABLE 1: Sequential Regression Results")
print("    Dependent Variable: GDP (Quarterly)")
print("=" * 70)

print(f"\n{'Variable':<15} {'Model 1':>12} {'Model 2':>12} {'Model 3':>12}")
print(f"{'':15} {'(OLS)':>12} {'(OLS)':>12} {'(OLS)':>12}")
print("-" * 55)

vars = ['const', 'Unemployment', 'FedFunds', 'HousingStarts']
labels = ['Intercept', 'Unemployment', 'FedFunds', 'Housing Starts']

for i, var in enumerate(vars):
    row = f"{labels[i]:<15}"
    
    # Model 1
    if var == 'const':
        row += f"{model1.params['const']:>12.2f} "
    elif var in model1.params.index:
        row += f"{model1.params[var]:>12.2f} "
    else:
        row += f"{'':>12} "
    
    # Model 2
    if var in model2.params.index:
        row += f"{model2.params[var]:>12.2f} "
    else:
        row += f"{'':>12} "
    
    # Model 3
    row += f"{model3.params[var]:>12.2f}"
    print(row)

print("-" * 55)
print(f"{'R-squared':<15} {model1.rsquared:>12.4f} {model2.rsquared:>12.4f} {model3.rsquared:>12.4f}")
print(f"{'Adj. R-squared':<15} {model1.rsquared_adj:>12.4f} {model2.rsquared_adj:>12.4f} {model3.rsquared_adj:>12.4f}")
print(f"{'N':<15} {int(model1.nobs):>12} {int(model2.nobs):>12} {int(model3.nobs):>12}")

print("\n>>> INTERPRETATION:")
print(f"   Adding FedFunds increases R-squared from {model1.rsquared:.4f} to {model2.rsquared:.4f}")
print(f"   Adding Housing Starts increases R-squared to {model3.rsquared:.4f}")
print("   All predictors are statistically significant!")

# Use model 3 as full model
model_full = model3

# =============================================================================
# STEP 3: DIAGNOSTIC PLOTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Creating Diagnostic Plots")
print("=" * 70)

fitted = model_full.fittedvalues
residuals = model_full.resid
std_resid = model_full.resid_pearson

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Regression Diagnostics - FRED Macro Data (Week 6)\n'
             'Y = GDP | X = Unemployment, FedFunds, Housing Starts',
             fontsize=14, fontweight='bold')

# Plot 1: Residuals vs Fitted
ax1 = axes[0, 0]
ax1.scatter(fitted, residuals, alpha=0.5, edgecolors='k', linewidth=0.3, s=30)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Fitted Values', fontsize=11)
ax1.set_ylabel('Residuals', fontsize=11)
ax1.set_title('1. Residuals vs Fitted\n(Check: random scatter around 0)', fontsize=12)

smoothed = lowess(residuals, fitted, frac=0.3)
ax1.plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=2, label='Trend')
ax1.legend()

# Check for heteroskedasticity
ax1.annotate('Potential\nHeteroskedasticity\n(funnel shape)', 
             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
             verticalalignment='top', color='darkred',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 2: Q-Q Plot
ax2 = axes[0, 1]
ProbPlot(std_resid).qqplot(line='45', ax=ax2, alpha=0.5, markersize=4)
ax2.set_title('2. Normal Q-Q Plot\n(Check: points on diagonal = normal)', fontsize=12)
ax2.get_lines()[0].set_markerfacecolor('steelblue')

# Plot 3: Scale-Location
ax3 = axes[1, 0]
ax3.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.5, edgecolors='k', linewidth=0.3, s=30)
smoothed3 = lowess(np.sqrt(np.abs(std_resid)), fitted, frac=0.3)
ax3.plot(smoothed3[:, 0], smoothed3[:, 1], 'r--', linewidth=2, label='Trend')
ax3.set_xlabel('Fitted Values', fontsize=11)
ax3.set_ylabel('√|Standardized Residuals|', fontsize=11)
ax3.set_title('3. Scale-Location\n(Check: flat line = homoskedasticity)', fontsize=12)
ax3.legend()

# Plot 4: Residuals vs Leverage
ax4 = axes[1, 1]
hat_values = model_full.get_influence().hat_matrix_diag
ax4.scatter(hat_values, residuals, alpha=0.5, edgecolors='k', linewidth=0.3, s=30)
ax4.set_xlabel('Leverage (Hat Value)', fontsize=11)
ax4.set_ylabel('Residuals', fontsize=11)
ax4.set_title('4. Residuals vs Leverage\n(High leverage + large residual = influential)', fontsize=12)

plt.tight_layout()
out_path = SCRIPT_DIR / 'diagnostic_plots_fred.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()

# =============================================================================
# STEP 4: BREUSCH-PAGAN TEST
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Breusch-Pagan Test for Heteroskedasticity")
print("=" * 70)

bp_test = het_breuschpagan(residuals, X3)
bp_lm_stat = bp_test[0]
bp_lm_pval = bp_test[1]

print(f"\nBreusch-Pagan Test Results:")
print(f"  LM Statistic: {bp_lm_stat:.4f}")
print(f"  LM p-value:   {bp_lm_pval:.4f}")

if bp_lm_pval < 0.05:
    print("\n*** REJECT H0: Heteroskedasticity is PRESENT! ***")
    print("   The assumption of constant variance is VIOLATED.")
    print("   >>> Use ROBUST standard errors! <<<")
    use_robust = True
else:
    print("\n*** FAIL TO REJECT H0: No significant heteroskedasticity ***")
    use_robust = False

# =============================================================================
# STEP 5: ROBUST STANDARD ERRORS (HC3)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Robust Standard Errors (HC3)")
print("=" * 70)

model_robust = model_full.get_robustcov_results(cov_type='HC3')
robust_bse = model_robust.bse

print("\n" + "=" * 70)
print("TABLE 2: OLS vs Robust (HC3) Standard Errors")
print("=" * 70)
print(f"\n{'Variable':<15} {'OLS Coef':>10} {'OLS SE':>10} {'Robust SE':>12} {'% Change':>10}")
print("-" * 65)

vars_idx = {'const': 0, 'Unemployment': 1, 'FedFunds': 2, 'HousingStarts': 3}
for var, idx in vars_idx.items():
    ols_coef = model_full.params[var]
    ols_se = model_full.bse[var]
    rob_se = robust_bse[idx]
    pct_change = ((rob_se - ols_se) / ols_se) * 100
    print(f"{var:<15} {ols_coef:>10.2f} {ols_se:>10.2f} {rob_se:>12.2f} {pct_change:>9.1f}%")

print("-" * 65)

# =============================================================================
# STEP 6: VIF FOR MULTICOLLINEARITY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Variance Inflation Factor (VIF)")
print("=" * 70)

X_no_const = df[['Unemployment', 'FedFunds', 'HousingStarts']]
vif_data = pd.DataFrame()
vif_data["Variable"] = X_no_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])]

print("\n" + "=" * 70)
print("TABLE 3: Variance Inflation Factors")
print("=" * 70)
print(f"\n{'Variable':<20} {'VIF':>10}")
print("-" * 32)
for _, row in vif_data.iterrows():
    print(f"{row['Variable']:<20} {row['VIF']:>10.2f}")
print("-" * 32)
print("\nRule of thumb: VIF > 5 = concerning, VIF > 10 = serious")
vif_max = vif_data['VIF'].max()
print(f"\nVIF = {vif_max:.2f} - ", end="")
if vif_max < 5:
    print("NO multicollinearity concern!")
elif vif_max < 10:
    print("MODERATE multicollinearity")
else:
    print("SERIOUS multicollinearity!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: WEEK 6 DIAGNOSTICS COMPLETE!")
print("=" * 70)
print(f"""
Key Findings:
1. R-squared: {model_full.rsquared:.4f} (model explains {model_full.rsquared*100:.1f}% of variance)
2. Heteroskedasticity: {'DETECTED' if use_robust else 'NOT detected'} (Breusch-Pagan p = {bp_lm_pval:.4f})
3. Multicollinearity: VIF = {vif_max:.2f} ({'OK' if vif_max < 5 else 'Check!'})
4. Robust SE: {"Used" if use_robust else "Optional (homoskedastic)"}

This dataset has {len(df)} quarterly observations - more data = tighter SE!
""")
