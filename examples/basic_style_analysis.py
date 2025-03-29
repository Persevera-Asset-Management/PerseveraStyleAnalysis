from persevera_style_analysis.core.sharpe_style_analysis import SharpeStyleAnalysis
from persevera_style_analysis.utils.helpers import extract_betas, compute_significance_mask
from persevera_style_analysis.utils import helpers
from persevera_tools.data import get_series, get_funds_data, get_persevera_peers
import matplotlib.pyplot as plt

# Get fund data
fund_data = get_funds_data(cnpjs=['44.417.598/0001-94'], fields=['fund_nav'])

# Get factor data
factor_cols = [
    'br_cdi_index',
    'br_ibovespa',
    'brl_usd',
    'us_sp500',
    'china_kweb',
    'gold',
    'crude_oil_wti',
    'br_pre_2y',
    'us_generic_10y'
]
factor_data = get_series(code=factor_cols)

# Prepare returns data
returns = helpers.prepare_data(fund_data, factor_data)

# Initialize analyzer
analyzer = SharpeStyleAnalysis(returns_data=returns, fund_cols=['44.417.598/0001-94'], factor_cols=factor_cols[1:])

print("Running analysis for the entire period...")
# Run analysis for the entire period
full_results = analyzer.run_analysis(
    min_window=10,
    max_window=50,
    vif_threshold=10.0,
    most_recent_only=False  # Analyze the entire period
)

print("\nRunning analysis for the most recent date only...")
# Run analysis for the most recent date only
recent_results = analyzer.run_analysis(
    min_window=10,
    max_window=50,
    vif_threshold=10.0,
    most_recent_only=True  # Analyze only the most recent date
)

# Extract betas and significance data
fund_cnpj = '44.417.598/0001-94'
full_betas = extract_betas(full_results, fund_cnpj)
recent_betas = extract_betas(recent_results, fund_cnpj)
recent_results[fund_cnpj]

print("\nFull period analysis results:")
print(f"Number of dates analyzed: {len(full_betas)}")
print("Latest factor exposures:")
print(full_betas.iloc[-1])

print("\nMost recent date analysis results:")
print(f"Number of dates analyzed: {len(recent_betas)}")
print("Factor exposures:")
print(recent_betas.iloc[0])

print(recent_results[fund_cnpj].T)

# Save results to database (uncomment to use)
# analyzer.save_results(recent_results, table_name='fundos_style_analysis')

# Visualize the factor exposures
plt.figure(figsize=(12, 6))
for col in full_betas.loc['2024':].columns:
    if col not in ['rsquared', 'rsquared_adj', 'window']:
        plt.plot(full_betas.loc['2024':].index, full_betas.loc['2024':][col], label=col)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title(f'Factor Exposures for Fund {fund_cnpj}')
plt.legend()
plt.savefig('factor_exposures.png')