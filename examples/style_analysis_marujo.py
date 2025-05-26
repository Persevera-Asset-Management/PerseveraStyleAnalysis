from persevera_style_analysis.core.sharpe_style_analysis import SharpeStyleAnalysis
from persevera_style_analysis.utils.helpers import extract_betas, compute_significance_mask
from persevera_style_analysis.utils import helpers
from persevera_tools.data import get_series, get_funds_data, get_persevera_peers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get fund data
peers = {
    '37.829.187/0001-40': 'Legacy Capital Prev PS FIC FIM',
    '41.409.879/0001-07': 'SPX Lancer Plus Prev FIM',
    '41.610.610/0001-94': 'Genoa Capital Cruise Prev FIC FIM',
    '42.479.991/0001-87': 'Vista Macro X FIM',
    '43.860.312/0001-88': 'Kapitalo Kappa 2 Prev XP Seg FIC FIM',
    '44.603.005/0001-84': 'Ibiuna ST Prev FIM'
}
fund_cnpjs = peers.keys()
fund_data = get_funds_data(cnpjs=fund_cnpjs, fields=['fund_nav'])
fund_data.rename(columns=peers, inplace=True)

# Get factor data
factor_cols = [
    'br_cdi_index',
    'anbima_ima_b',
    'anbima_ima_b5',
    'anbima_ima_b5+',
    'br_ibovespa',
    'brl_usd',
    'us_sp500',
    # 'gold',
    # 'crude_oil_wti',
    'br_pre_2y',
    'us_generic_10y'
]

factor_data = get_series(code=factor_cols)

# Prepare returns data
returns = helpers.prepare_data(fund_data, factor_data)

# Handle missing values in the returns data
returns = returns.dropna(how='any')
print(f"Cleaned returns data: {returns.shape[0]} rows after removing NaN values")

# Initialize analyzer
analyzer = SharpeStyleAnalysis(returns_data=returns, fund_cols=peers.values(), factor_cols=factor_cols[1:])

print("\nRunning analysis for the most recent date only...")
# Run analysis for the most recent date only
try:
    recent_results = analyzer.run_analysis(
        min_window=10,
        max_window=50,
        vif_threshold=10.0,
        most_recent_only=True  # Analyze only the most recent date
    )
except Exception as e:
    print(f"Error in recent date analysis: {e}")
    recent_results = {}

# Compare betas across all funds, organized by factor
print("\nCreating cross-fund beta comparison by factor...")

# Extract factor list (excluding non-factor columns)
factors = factor_cols[1:]  # Skip the first factor (br_cdi_index) as it's usually a benchmark or intercept

# Create a DataFrame to store betas for all funds
all_betas = pd.DataFrame()
all_pvalues = pd.DataFrame()
all_rsquared = pd.Series()
all_rsquared_adj = pd.Series()
all_windows = pd.Series()  # For storing window sizes

# Extract betas and p-values for each fund
for fund_id, fund_name in peers.items():
    if fund_name in recent_results:
        try:
            # Get beta values and drop non-factor columns
            beta_values = recent_results[fund_name].loc['beta'].drop(['date', 'fund', 'rsquared', 'rsquared_adj', 'window'], errors='ignore')
            pvalue_values = recent_results[fund_name].loc['pvalue'].drop(['date', 'fund', 'rsquared', 'rsquared_adj', 'window'], errors='ignore')
            
            # Add to DataFrames with short name as column label
            all_betas[fund_name] = beta_values
            all_pvalues[fund_name] = pvalue_values
            
            # Get R-squared values
            all_rsquared[fund_name] = recent_results[fund_name].loc['model', 'rsquared']
            all_rsquared_adj[fund_name] = recent_results[fund_name].loc['model', 'rsquared_adj']
            
            # Get window size
            all_windows[fund_name] = recent_results[fund_name].loc['model', 'window']
        except Exception as e:
            print(f"Could not extract data for fund {fund_name}: {e}")

# Create a chart for each factor if we have data
if not all_betas.empty and len(factors) > 0:
    plt.figure(figsize=(15, 20))
    n_factors = len(factors)
    rows = (n_factors + 1) // 2  # Calculate number of rows (2 charts per row)
    cols = min(2, n_factors)     # Maximum 2 columns
    
    for i, factor in enumerate(factors):
        # Create subplot
        ax = plt.subplot(rows, cols, i+1)
        
        # Get data for this factor
        if factor in all_betas.index:
            factor_betas = all_betas.loc[factor].sort_values(ascending=False)
            factor_pvalues = all_pvalues.loc[factor].loc[factor_betas.index]
            
            # Define bar positions
            x = np.arange(len(factor_betas))
            
            # Create bar colors based on p-values
            bar_colors = ['darkblue' if p < 0.10 else 'skyblue' for p in factor_pvalues]
            
            # Create bars
            bars = ax.bar(x, factor_betas, color=bar_colors)
            
            # Add a zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels above bars
            for j, v in enumerate(factor_betas):
                value_text = f"{v:.2f}"
                if v >= 0:
                    ax.text(j, v + 0.01, value_text, ha='center', va='bottom', fontsize=8, rotation=90)
                else:
                    ax.text(j, v - 0.03, value_text, ha='center', va='top', fontsize=8, rotation=90)
            
            # Customize plot
            ax.set_title(f'{factor}')
            ax.set_xticks(x)
            ax.set_xticklabels(factor_betas.index, rotation=45, ha='right')
            ax.set_ylim(min(factor_betas.min() * 1.2, -0.1), max(factor_betas.max() * 1.2, 0.1))
    
    # Add a common title and legend explanation
    plt.suptitle('Factor Exposures Comparison Across Funds\n(Dark blue = significant at 10% level)', fontsize=16)
    plt.figtext(0.5, 0.01, 'Last position of each fund', ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
    plt.savefig('factor_exposures_comparison.png')
    print("Created comparison chart: factor_exposures_comparison.png")

# Create R-squared comparison chart if we have data
if not all_rsquared.empty:
    try:
        # Sort R-squared values in descending order
        sorted_rsquared = all_rsquared.sort_values(ascending=False)
        sorted_rsquared_adj = all_rsquared_adj.reindex(sorted_rsquared.index)
        
        # Define positions
        x = np.arange(len(sorted_rsquared))
        width = 0.35  # width of the bars
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width/2, sorted_rsquared, width, label='R-squared', color='steelblue')
        bars2 = ax.bar(x + width/2, sorted_rsquared_adj, width, label='Adjusted R-squared', color='lightsteelblue')
        
        # Add a horizontal line at important thresholds
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='0.7 threshold')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='0.5 threshold')
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='0.3 threshold')
        
        # Add value labels above bars
        for i, v in enumerate(sorted_rsquared):
            ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8, rotation=0)
            
        for i, v in enumerate(sorted_rsquared_adj):
            ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Customize plot
        ax.set_title('Model R-squared Comparison Across Funds', fontsize=14)
        ax.set_xlabel('Fund', fontsize=12)
        ax.set_ylabel('R-squared Value', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_rsquared.index, rotation=45, ha='right')
        ax.set_ylim(0, min(1.0, sorted_rsquared.max() * 1.2))
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('rsquared_comparison.png')
        print("Created R-squared comparison chart: rsquared_comparison.png")
    except Exception as e:
        print(f"Error creating R-squared chart: {e}")

# Create window size comparison chart if we have data
if not all_windows.empty:
    try:
        # Sort window sizes in descending order
        sorted_windows = all_windows.sort_values(ascending=False)
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define positions
        x = np.arange(len(sorted_windows))
        
        # Create bars
        bars = ax.bar(x, sorted_windows, color='darkgreen')
        
        # Add value labels above bars
        for i, v in enumerate(sorted_windows):
            ax.text(i, v + 0.5, f"{int(v)}", ha='center', va='bottom', fontsize=10)
        
        # Customize plot
        ax.set_title('Window Size Comparison Across Funds', fontsize=14)
        ax.set_xlabel('Fund', fontsize=12)
        ax.set_ylabel('Window Size (days)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_windows.index, rotation=45, ha='right')
        
        # Add reference lines
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Min window (10)')
        ax.axhline(y=50, color='blue', linestyle='--', alpha=0.7, label='Max window (50)')
        ax.axhline(y=sum(sorted_windows) / len(sorted_windows), color='orange', 
                   linestyle='--', alpha=0.7, label=f'Average ({sum(sorted_windows) / len(sorted_windows):.1f})')
        
        ax.set_ylim(0, max(sorted_windows) * 1.1)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('window_size_comparison.png')
        print("Created window size comparison chart: window_size_comparison.png")
    except Exception as e:
        print(f"Error creating window size chart: {e}")
