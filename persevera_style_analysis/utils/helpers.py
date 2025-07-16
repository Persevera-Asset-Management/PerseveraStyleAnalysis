import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import datetime

def prepare_data(fund_data: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for style analysis by merging fund data with factors and calculating returns.
    
    Args:
        fund_data: DataFrame with fund NAV data
        factor_data: DataFrame with factor data
        
    Returns:
        DataFrame with fund and factor returns
    """
    # Merge fund and factor data
    merged_data = pd.merge(fund_data, factor_data, left_index=True, right_index=True, how='left')
    
    # Forward fill to handle missing values
    merged_data = merged_data.ffill()
    
    # Calculate returns
    yield_cols = [
        # Brazil
        'br_pre_2y', 'br_pre_3y', 'br_pre_5y', 'br_pre_10y',
        'br_bmf_di_jan27_futures', 'br_bmf_di_jan28_futures', 'br_bmf_di_jan29_futures', 'br_bmf_di_jan30_futures', 'br_bmf_di_jan31_futures', 'br_bmf_di_jan32_futures', 'br_bmf_di_jan33_futures',

        # US
        'us_generic_10y', 'br_bmf_us_treasury_10y_futures'
    ]

    returns = merged_data.apply(
        lambda col: col.pct_change() if col.name not in yield_cols else col.diff()
    ).dropna(how='all')
    
    return returns 

def calculate_contribution(factor_exposures: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate factor contribution to fund returns.
    
    Args:
        factor_exposures: DataFrame with factor exposures (betas)
        factor_returns: DataFrame with factor returns
        
    Returns:
        DataFrame with factor contributions
    """
    # Ensure index alignment
    common_index = factor_exposures.index.intersection(factor_returns.index)
    factor_exposures = factor_exposures.loc[common_index]
    factor_returns = factor_returns.loc[common_index]
    
    # Calculate contribution
    contributions = pd.DataFrame(index=common_index)
    
    for factor in factor_exposures.columns:
        if factor in factor_returns.columns:
            contributions[factor] = factor_exposures[factor] * factor_returns[factor]
    
    # Add total contribution
    contributions['total'] = contributions.sum(axis=1)
    
    return contributions

def calculate_relative_contribution(contributions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relative contribution (percentage of total).
    
    Args:
        contributions: DataFrame with factor contributions
        
    Returns:
        DataFrame with relative contributions
    """
    # Exclude 'total' column
    factors = [col for col in contributions.columns if col != 'total']
    
    # Calculate absolute contributions
    abs_contributions = contributions[factors].abs()
    total_abs = abs_contributions.sum(axis=1)
    
    # Calculate relative contributions
    relative = pd.DataFrame(index=contributions.index)
    for factor in factors:
        relative[factor] = abs_contributions[factor] / total_abs
    
    return relative

def calculate_tracking_error(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate tracking error between fund and benchmark.
    
    Args:
        fund_returns: Series with fund returns
        benchmark_returns: Series with benchmark returns
        
    Returns:
        Tracking error (annualized)
    """
    # Ensure index alignment
    common_index = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    # Calculate excess returns
    excess_returns = fund_returns - benchmark_returns
    
    # Calculate tracking error
    tracking_error = excess_returns.std() * np.sqrt(252)  # Annualized
    
    return tracking_error

def calculate_information_ratio(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate information ratio.
    
    Args:
        fund_returns: Series with fund returns
        benchmark_returns: Series with benchmark returns
        
    Returns:
        Information ratio
    """
    # Ensure index alignment
    common_index = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    # Calculate excess returns
    excess_returns = fund_returns - benchmark_returns
    
    # Calculate annualized excess return
    annual_excess = excess_returns.mean() * 252
    
    # Calculate tracking error
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    # Calculate information ratio
    information_ratio = annual_excess / tracking_error if tracking_error != 0 else 0
    
    return information_ratio

def date_windows(
    start_date: Union[str, datetime.date],
    end_date: Union[str, datetime.date],
    window_size: int = 36,
    step_size: int = 1,
    freq: str = 'M'
) -> List[Tuple[datetime.date, datetime.date]]:
    """
    Generate a list of date windows for analysis.
    
    Args:
        start_date: Start date
        end_date: End date
        window_size: Window size in periods
        step_size: Step size in periods
        freq: Frequency ('D' for daily, 'M' for monthly, 'Y' for yearly)
        
    Returns:
        List of (start_date, end_date) tuples
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate windows
    windows = []
    for i in range(window_size, len(date_range), step_size):
        window_end = date_range[i].date()
        window_start = date_range[i - window_size].date()
        windows.append((window_start, window_end))
    
    return windows

def calculate_style_drift(factor_exposures: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Calculate style drift based on changes in factor exposures.
    
    Args:
        factor_exposures: DataFrame with factor exposures
        window: Window for calculating drift
        
    Returns:
        DataFrame with style drift metrics
    """
    # Calculate the standard deviation of factor exposures over rolling windows
    drift = factor_exposures.rolling(window=window).std()
    
    # Calculate total drift
    drift['total'] = np.sqrt((drift ** 2).sum(axis=1))
    
    return drift

def extract_betas(results: Dict[str, pd.DataFrame], fund: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Extract beta coefficients from style analysis results.
    
    Args:
        results: Dictionary of results from StyleAnalysis.run_analysis()
        fund: Fund name to extract betas for
        start_date: Optional start date to filter results
        end_date: Optional end date to filter results
        
    Returns:
        DataFrame with beta coefficients
    """
    if fund not in results:
        raise ValueError(f"Fund '{fund}' not found in results")
    
    # Get the data for this fund
    df = results[fund].copy()
    
    # Filter by date if specified
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    
    # Extract betas
    betas = df[df.index == 'beta'].drop(columns=['fund']).set_index('date')
    
    return betas

def compute_significance_mask(results: Dict[str, pd.DataFrame], fund: str, p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Create a mask for statistically significant betas.
    
    Args:
        results: Dictionary of results from StyleAnalysis.run_analysis()
        fund: Fund name
        p_threshold: p-value threshold for statistical significance
        
    Returns:
        Boolean DataFrame with True for significant betas
    """
    if fund not in results:
        raise ValueError(f"Fund '{fund}' not found in results")
    
    # Get p-values for this fund
    df = results[fund].copy()
    pvalues = df[df.index == 'pvalue'].drop(columns=['fund']).set_index('date')
    
    # Create significance mask
    significant = pvalues < p_threshold
    
    return significant 