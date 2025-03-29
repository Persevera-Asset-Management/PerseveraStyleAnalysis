# Persevera Style Analysis

A Python package for analyzing investment fund styles using Sharpe Style Analysis and rolling regressions.

## Installation

```bash
pip install git+https://github.com/Persevera-Asset-Management/PerseveraStyleAnalysis.git
```

## Features

- **Sharpe Style Analysis**: Determine fund exposure to various asset classes
- **Rolling Regression**: Capture dynamic style exposure over time
- **Adaptive Window Size**: Optimize window size based on VIF and R-squared
- **Robust Statistical Analysis**: Uses statsmodels for comprehensive regression statistics
- **Incremental Analysis**: Option to analyze only the most recent date for faster updates
- **Database Integration**: Seamless connection with Persevera database

## Usage

```python
from persevera_style_analysis.core.sharpe_style_analysis import SharpeStyleAnalysis
from persevera_style_analysis.utils.helpers import extract_betas, compute_significance_mask
from persevera_tools.data import get_series, get_funds_data
from persevera_style_analysis.utils import helpers
import matplotlib.pyplot as plt

# Get fund data
fund_data = get_funds_data(cnpjs=['44.417.598/0001-94'], fields=['fund_nav'])

# Get factor data
factor_cols = [
    'br_cdi_index',
    'br_ibovespa',
    'brl_usd',
    'us_sp500',
    'china_msci',
    'gold',
    'crude_oil_wti',
    'br_pre_2y',
    'us_generic_10y'
]
factor_data = get_series(code=factor_cols)

# Prepare returns data
returns = helpers.prepare_data(fund_data, factor_data)

# Initialize analyzer
analyzer = SharpeStyleAnalysis(returns_data=returns)

# Run analysis for the entire period
full_results = analyzer.run_analysis(
    min_window=12,
    max_window=50,
    vif_threshold=10.0,
    most_recent_only=False  # Analyze the entire period
)

# Or run analysis for just the most recent date (faster updates)
recent_results = analyzer.run_analysis(
    min_window=12,
    max_window=50,
    vif_threshold=10.0,
    most_recent_only=True  # Analyze only the most recent date
)

# Extract betas and significance data
fund_cnpj = '44.417.598/0001-94'
betas = extract_betas(full_results, fund_cnpj)
significant = compute_significance_mask(full_results, fund_cnpj)

# Visualize the factor exposures
plt.figure(figsize=(12, 6))
for col in betas.columns:
    if col not in ['rsquared', 'rsquared_adj', 'window']:
        plt.plot(betas.index, betas[col], label=col)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title(f'Factor Exposures for Fund {fund_cnpj}')
plt.legend()
plt.savefig('factor_exposures.png')

# Save results to database
analyzer.save_results(recent_results, table_name='fundos_style_analysis')
```

## Project Structure

```
persevera_style_analysis/
├── core/            # Core analysis functionality
│   └── sharpe_style_analysis.py  # Implementation using statsmodels
├── data/            # Data loading and processing
└── utils/           # Utility functions
    └── helpers.py   # Helper functions for data preparation and analysis
```

## Requirements

- Python 3.8+
- pandas
- numpy
- statsmodels
- matplotlib (for visualization)
- persevera_tools (for data access)