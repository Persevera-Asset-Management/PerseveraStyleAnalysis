import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import statsmodels.api as sm

from persevera_style_analysis.core import (
    StepwiseStyleAnalysis, 
    BestSubsetStyleAnalysis
)
from persevera_style_analysis.utils.helpers import extract_betas
from persevera_style_analysis.utils import helpers
from persevera_tools.data import get_series, get_funds_data

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Create timestamped results directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', f'single_fund_analysis_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}")

# CONFIGURATION
# =============
# Specify the fund to analyze
FUND_CNPJ = '44.417.598/0001-94'  # Ibiuna ST Prev FIM
FUND_NAME = 'Persevera Nemesis Total Return FIM'

# Define factors to use
FACTOR_COLS = [
    'br_cdi_index',
    'br_ibovespa',
    'us_sp500',
    'brl_usd',
    'br_pre_2y',
    'br_pre_5y',
    'us_generic_10y',
    'gold',
]

# Analysis parameters
MIN_WINDOW = 10
MAX_WINDOW = 50
THRESHOLD_OUT = 0.05  # For stepwise
VIF_THRESHOLD = 10.0
SELECTION_METRIC = 'adjr2'  # 'aic', 'bic', or 'adjr2'

# DATA LOADING
# ===========
print(f"Loading data for fund: {FUND_NAME}")
fund_data = get_funds_data(cnpjs=[FUND_CNPJ], fields=['fund_nav'])
fund_data.rename(columns={fund_data.columns[0]: FUND_NAME}, inplace=True)

factor_data = get_series(code=FACTOR_COLS)

# Prepare returns data
returns_data = helpers.prepare_data(fund_data, factor_data)
returns_data = returns_data.dropna(how='any')
print(f"Cleaned returns data: {returns_data.shape[0]} rows after removing NaN values")

# ANALYSIS SETUP
# ==============
# Initialize style analysis methods
stepwise_analysis = StepwiseStyleAnalysis(
    returns_data=returns_data,
    fund_cols=[FUND_NAME],
    factor_cols=FACTOR_COLS
)

best_subset_analysis = BestSubsetStyleAnalysis(
    returns_data=returns_data,
    fund_cols=[FUND_NAME],
    factor_cols=FACTOR_COLS
)

# COMPARATIVE ANALYSIS
# ===================
# Run both methods with different window sizes
def run_comparison_with_fixed_window(window_size):
    """Run both methods with a fixed window size and compare results"""
    print(f"\n{'='*60}\nCOMPARISON WITH FIXED WINDOW: {window_size}\n{'='*60}")
    
    # Get the most recent data point
    current_idx = len(returns_data) - 1
    current_date = returns_data.index[current_idx]
    
    # Get excess returns
    y_excess = (returns_data[FUND_NAME].iloc[current_idx - window_size:current_idx] - 
                returns_data['br_cdi_index'].iloc[current_idx - window_size:current_idx])
    
    # Get factor returns (excluding risk-free)
    X_cols = [col for col in FACTOR_COLS if col != 'br_cdi_index']
    X = returns_data[X_cols].iloc[current_idx - window_size:current_idx]
    X = sm.add_constant(X)
    
    # Run best subset regression
    print(f"\n--- BEST SUBSET with window={window_size} ---")
    bs_model, bs_indices = best_subset_analysis._best_subset_regression(X, y_excess, metric=SELECTION_METRIC)
    
    print(f"Best Subset R²: {bs_model.rsquared:.6f}")
    print(f"Best Subset Adj. R²: {bs_model.rsquared_adj:.6f}")
    print("Included factors:")
    included_bs_factors = []
    for idx in bs_indices:
        factor_name = X.columns[idx + 1]  # +1 to skip constant
        included_bs_factors.append(factor_name)
        coef = bs_model.params[f"{factor_name}"]
        pval = bs_model.pvalues[f"{factor_name}"]
        print(f"  - {factor_name}: {coef:.4f} (p-value: {pval:.4f})")
    
    # Run stepwise regression
    print(f"\n--- STEPWISE with window={window_size} ---")
    sw_model, sw_indices = stepwise_analysis._backward_stepwise(X, y_excess, threshold_out=THRESHOLD_OUT)
    
    print(f"Stepwise R²: {sw_model.rsquared:.6f}")
    print(f"Stepwise Adj. R²: {sw_model.rsquared_adj:.6f}")
    print("Included factors:")
    included_sw_factors = []
    for idx in sw_indices:
        factor_name = X.columns[idx]  # Already adjusted for constant in return
        included_sw_factors.append(factor_name)
        coef = sw_model.params[f"{factor_name}"]
        pval = sw_model.pvalues[f"{factor_name}"]
        print(f"  - {factor_name}: {coef:.4f} (p-value: {pval:.4f})")
    
    # Compare results
    print("\n--- COMPARISON ---")
    print(f"Best Subset Adj. R²: {bs_model.rsquared_adj:.6f}, Factors: {len(bs_indices)}")
    print(f"Stepwise Adj. R²:    {sw_model.rsquared_adj:.6f}, Factors: {len(sw_indices)}")
    
    # Identify common and different factors
    common_factors = set(included_bs_factors).intersection(set(included_sw_factors))
    bs_only = set(included_bs_factors) - set(included_sw_factors)
    sw_only = set(included_sw_factors) - set(included_bs_factors)
    
    print(f"Common factors: {common_factors}")
    print(f"Best Subset only: {bs_only}")
    print(f"Stepwise only: {sw_only}")
    
    # Plot comparison
    factors = list(X.columns)[1:]  # Skip constant
    bs_coefs = pd.Series(0.0, index=factors)
    sw_coefs = pd.Series(0.0, index=factors)
    
    # Set Best Subset coefficients
    for idx in bs_indices:
        factor_name = X.columns[idx + 1]
        bs_coefs[factor_name] = bs_model.params[f"{factor_name}"]
    
    # Set Stepwise coefficients
    for idx in sw_indices:
        factor_name = X.columns[idx]
        sw_coefs[factor_name] = sw_model.params[f"{factor_name}"]
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Best Subset': bs_coefs,
        'Stepwise': sw_coefs
    })
    
    # Plot
    plt.figure(figsize=(14, 8))
    comparison.plot(kind='bar')
    plt.title(f'Factor Exposures for {FUND_NAME} with Window={window_size}\n'
              f'Best Subset Adj. R²: {bs_model.rsquared_adj:.4f}, Stepwise Adj. R²: {sw_model.rsquared_adj:.4f}')
    plt.ylabel('Beta Coefficient')
    plt.xlabel('Factors')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, f'window_{window_size}_comparison.png')
    plt.savefig(plot_path)
    
    # Return results for later analysis
    return {
        'window': window_size,
        'bs_rsq': bs_model.rsquared,
        'bs_adj_rsq': bs_model.rsquared_adj,
        'bs_factors': len(bs_indices),
        'bs_included': included_bs_factors,
        'sw_rsq': sw_model.rsquared,
        'sw_adj_rsq': sw_model.rsquared_adj,
        'sw_factors': len(sw_indices),
        'sw_included': included_sw_factors
    }

# Try different window sizes
window_results = []
for window in [10, 12, 18, 24, 36]:
    result = run_comparison_with_fixed_window(window)
    window_results.append(result)

# AUTOMATIC WINDOW SELECTION
# =========================
print(f"\n{'='*60}\nAUTOMATIC WINDOW SELECTION COMPARISON\n{'='*60}")

# Run both methods with automatic window selection
print("\nRunning stepwise analysis with automatic window selection...")
stepwise_results = stepwise_analysis.run_analysis(
    min_window=MIN_WINDOW,
    max_window=MAX_WINDOW,
    vif_threshold=VIF_THRESHOLD,
    threshold_out=THRESHOLD_OUT,
    funds=[FUND_NAME],
    most_recent_only=True
)

print("\nRunning best subset analysis with automatic window selection...")
best_subset_results = best_subset_analysis.run_analysis(
    min_window=MIN_WINDOW,
    max_window=MAX_WINDOW,
    vif_threshold=VIF_THRESHOLD,
    selection_metric=SELECTION_METRIC,
    funds=[FUND_NAME],
    most_recent_only=True
)

# Get the window sizes chosen by each method
best_subset_window = best_subset_results[FUND_NAME].loc['model', 'window']
stepwise_window = stepwise_results[FUND_NAME].loc['model', 'window']

print(f"\nWindow chosen by Best Subset: {best_subset_window}")
print(f"Window chosen by Stepwise: {stepwise_window}")

# Run comparison with the chosen windows
if best_subset_window != stepwise_window:
    print("\nRunning comparison with both chosen windows...")
    run_comparison_with_fixed_window(int(best_subset_window))
    run_comparison_with_fixed_window(int(stepwise_window))

# SUMMARY
# ========
print(f"\n{'='*60}\nANALYSIS SUMMARY\n{'='*60}")

# Create summary table
summary = pd.DataFrame(window_results)
summary = summary.set_index('window')

print("\nPerformance metrics by window size:")
print(summary[['bs_adj_rsq', 'sw_adj_rsq', 'bs_factors', 'sw_factors']])

# Plot R-squared vs Window Size
plt.figure(figsize=(12, 6))
plt.plot(summary.index, summary['bs_adj_rsq'], 'o-', label='Best Subset')
plt.plot(summary.index, summary['sw_adj_rsq'], 's-', label='Stepwise')
plt.axvline(x=best_subset_window, color='blue', linestyle='--', 
           label=f'Best Subset Selected Window: {best_subset_window}')
plt.axvline(x=stepwise_window, color='orange', linestyle='--',
           label=f'Stepwise Selected Window: {stepwise_window}')
plt.title(f'Adjusted R² vs Window Size for {FUND_NAME}')
plt.xlabel('Window Size')
plt.ylabel('Adjusted R²')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
summary_path = os.path.join(results_dir, 'window_size_comparison.png')
plt.savefig(summary_path)

# Save results
summary_csv = os.path.join(results_dir, 'window_summary.csv')
summary.to_csv(summary_csv)

print(f"\nAnalysis completed! Results saved to: {results_dir}") 