import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import statsmodels.api as sm

from persevera_style_analysis.core import (
    SharpeStyleAnalysis, 
    StepwiseStyleAnalysis, 
    BestSubsetStyleAnalysis
)
from persevera_style_analysis.utils.helpers import extract_betas, compute_significance_mask
from persevera_style_analysis.utils import helpers
from persevera_tools.data import get_series, get_funds_data, get_persevera_peers
from persevera_tools.utils.logging import get_logger
from persevera_tools.utils.logging import initialize as _initialize_logging

_initialize_logging()
logger = get_logger(__name__)

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Create timestamped results directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', f'style_analysis_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}")

# Get fund data
peers = {
    '37.829.187/0001-40': 'Legacy Capital Prev PS FIC FIM',
    '41.409.879/0001-07': 'SPX Lancer Plus Prev FIM',
    '41.610.610/0001-94': 'Genoa Capital Cruise Prev FIC FIM',
    '42.479.991/0001-87': 'Vista Macro X FIM',
    '43.860.312/0001-88': 'Kapitalo Kappa 2 Prev XP Seg FIC FIM',
    # '44.603.005/0001-84': 'Ibiuna ST Prev FIM',
    '44.417.598/0001-94': 'Persevera Nemesis Total Return FIM'
}
peers = get_persevera_peers('Nemesis').set_index('fund_cnpj')['short_name'].to_dict()
fund_cnpjs = peers.keys()
fund_data = get_funds_data(cnpjs=fund_cnpjs, fields=['fund_nav'])
fund_data.rename(columns=peers, inplace=True)

# Get factor data
factor_cols = [
    'br_cdi_index',
    # 'anbima_ima_b',
    # 'anbima_ima_b5',
    # 'anbima_ima_b5+',
    'br_ibovespa',
    'us_sp500',
    'brl_usd',
    'br_pre_2y',
    # 'br_pre_5y',
    'us_generic_10y',
    'gold',
    # 'br_gold11',
    # 'crude_oil_wti',
    # 'bitcoin_usd'
]

factor_data = get_series(code=factor_cols)

# Prepare returns data
returns_data = helpers.prepare_data(fund_data, factor_data)

# Handle missing values in the returns data
returns_data = returns_data.dropna(how='any')
print(f"Cleaned returns data: {returns_data.shape[0]} rows after removing NaN values")

# Initialize each style analysis method
print("\nInitializing style analysis models...")
standard_analysis = SharpeStyleAnalysis(
    returns_data=returns_data,
    fund_cols=peers.values(),
    factor_cols=factor_cols
)

stepwise_analysis = StepwiseStyleAnalysis(
    returns_data=returns_data,
    fund_cols=peers.values(),
    factor_cols=factor_cols
)

best_subset_analysis = BestSubsetStyleAnalysis(
    returns_data=returns_data,
    fund_cols=peers.values(),
    factor_cols=factor_cols
)

# Parameters
min_window = 10
max_window = 50
threshold_out = 0.10

# Run analysis for the most recent date
print("\nRunning standard style analysis for the most recent date...")
standard_results = standard_analysis.run_analysis(
    min_window=min_window,
    max_window=max_window,
    most_recent_only=True
)

print("\nRunning stepwise style analysis for the most recent date...")
stepwise_results = stepwise_analysis.run_analysis(
    min_window=min_window,
    max_window=max_window,
    threshold_out=threshold_out,  # Using 10% significance level
    most_recent_only=True
)

print("\nRunning best subset style analysis for the most recent date...")
best_subset_results = best_subset_analysis.run_analysis(
    min_window=min_window,
    max_window=max_window,
    selection_metric='adjr2',  # aic, bic, adjr2
    most_recent_only=True
)

# Save all metrics to a summary CSV file
all_metrics = []

# Compare betas across all three methods for a single fund
def compare_fund_betas(fund_name):
    print(f"\nComparing factor exposures for {fund_name}:")
    
    # Extract beta values from each analysis
    standard_betas = standard_results[fund_name].loc['beta'].drop(['date', 'fund', 'rsquared', 'rsquared_adj', 'window'], errors='ignore')
    stepwise_betas = stepwise_results[fund_name].loc['beta'].drop(['date', 'fund', 'rsquared', 'rsquared_adj', 'window', 'included_factors'], errors='ignore')
    best_subset_betas = best_subset_results[fund_name].loc['beta'].drop(['date', 'fund', 'rsquared', 'rsquared_adj', 'window', 'included_factors', 'selection_metric'], errors='ignore')
    
    # Combine all betas
    comparison = pd.DataFrame({
        'Standard': standard_betas,
        'Stepwise': stepwise_betas,
        'Best Subset': best_subset_betas
    })
    
    # Get R-squared values
    r2_values = {
        'Standard': standard_results[fund_name].loc['model', 'rsquared_adj'],
        'Stepwise': stepwise_results[fund_name].loc['model', 'rsquared_adj'],
        'Best Subset': best_subset_results[fund_name].loc['model', 'rsquared_adj']
    }
    
    # Get included factors count
    included_factors = {
        'Standard': len(standard_betas) - 1,  # Subtract 1 for const
        'Stepwise': stepwise_results[fund_name].loc['model', 'included_factors'],
        'Best Subset': best_subset_results[fund_name].loc['model', 'included_factors']
    }
    
    # Get optimal window sizes
    window_sizes = {
        'Standard': standard_results[fund_name].loc['model', 'window'],
        'Stepwise': stepwise_results[fund_name].loc['model', 'window'],
        'Best Subset': best_subset_results[fund_name].loc['model', 'window']
    }
    
    # Print metrics
    print("\nMetrics Comparison:")
    metrics_df = pd.DataFrame({
        'Adjusted R-squared': r2_values,
        'Included Factors': included_factors,
        'Optimal Window': window_sizes
    })
    print(metrics_df)
    
    # Add to overall metrics
    fund_metrics = metrics_df.copy()
    fund_metrics['Fund'] = fund_name
    all_metrics.append(fund_metrics)
    
    # Create comparison plot
    plt.figure(figsize=(14, 8))
    comparison.plot(kind='bar')
    plt.title(f'Factor Exposures Comparison for {fund_name}\nOptimal Windows: Standard={window_sizes["Standard"]}, Stepwise={window_sizes["Stepwise"]}, Best Subset={window_sizes["Best Subset"]}')
    plt.ylabel('Beta Coefficient')
    plt.xlabel('Factors')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.tight_layout()
    
    # Save plot in the timestamped results directory
    plot_path = os.path.join(results_dir, f'{fund_name.replace(" ", "_")}_comparison.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    return comparison

# Run comparison for each fund
for fund_name in peers.values():
    compare_fund_betas(fund_name)

# Save summary metrics
if all_metrics:
    metrics_summary = pd.concat(all_metrics)
    metrics_path = os.path.join(results_dir, 'metrics_summary.csv')
    metrics_summary.to_csv(metrics_path)
    print(f"\nMetrics summary saved to {metrics_path}")

print("\nAnalysis completed! All results saved in:", results_dir)

# Optional: Save results to database
# standard_analysis.save_results(standard_results)
# stepwise_analysis.save_results(stepwise_results)
# best_subset_analysis.save_results(best_subset_results)

# Debugging comparison of methods for Moat Capital fund

logger.info("\n==================================================")
logger.info("DEBUGGING COMPARISON FOR MOAT CAPITAL EQUITY HEDGE FIC FIM")
logger.info("==================================================\n")

fund_to_debug = 'Ibiuna ST Prev FIM'

# Create a debugging function to run both methods with fixed window sizes
def debug_comparison(fund_name, window_size):
    # Get data for specific window for this fund
    current_idx = len(returns_data) - 1  # Use most recent data
    
    # Get excess returns
    y_excess = (returns_data[fund_name].iloc[current_idx - window_size:current_idx] - 
                returns_data['br_cdi_index'].iloc[current_idx - window_size:current_idx])
    
    # Get factor returns (excluding risk-free)
    X_cols = [col for col in factor_cols if col != 'br_cdi_index']
    X = returns_data[X_cols].iloc[current_idx - window_size:current_idx]
    X = sm.add_constant(X)
    
    # Run best subset regression
    print(f"\n--- BEST SUBSET with window={window_size} ---")
    bs_model, bs_indices = best_subset_analysis._best_subset_regression(X, y_excess, metric='adjr2')
    
    print(f"Best Subset R²: {bs_model.rsquared:.6f}")
    print(f"Best Subset Adj. R²: {bs_model.rsquared_adj:.6f}")
    print("Included factors:")
    for idx in bs_indices:
        print(f"  - {X.columns[idx + 1]}")  # +1 to skip constant
    
    # Run stepwise regression
    print(f"\n--- STEPWISE with window={window_size} ---")
    sw_model, sw_indices = stepwise_analysis._backward_stepwise(X, y_excess, threshold_out=0.05)
    
    print(f"Stepwise R²: {sw_model.rsquared:.6f}")
    print(f"Stepwise Adj. R²: {sw_model.rsquared_adj:.6f}")
    print("Included factors:")
    for idx in sw_indices:
        print(f"  - {X.columns[idx]}")  # Already adjusted for constant in return

# Try different window sizes
logger.info("Comparing methods with different window sizes:")
for window in [12, 24, 36]:
    debug_comparison(fund_to_debug, window)

# Now run with the specific windows selected by each method
logger.info("\n\nComparing with windows chosen by each method:")
# Get the window sizes chosen by each method from the results
best_subset_window = best_subset_results[fund_to_debug].loc['model', 'window']
stepwise_window = stepwise_results[fund_to_debug].loc['model', 'window']

logger.info(f"Window chosen by Best Subset: {best_subset_window}")
logger.info(f"Window chosen by Stepwise: {stepwise_window}")

debug_comparison(fund_to_debug, 10)
debug_comparison(fund_to_debug, 24)
debug_comparison(fund_to_debug, 36)