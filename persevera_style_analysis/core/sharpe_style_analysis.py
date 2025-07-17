import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Union, Optional, Tuple
from persevera_tools.db import to_sql
from persevera_style_analysis.utils import helpers

logger = logging.getLogger(__name__)

class SharpeStyleAnalysis:
    """
    Style analysis implementation using Sharpe's approach with rolling regressions.
    """
    
    def __init__(
        self, 
        returns_data: pd.DataFrame,
        fund_cols: Optional[List[str]] = None,
        factor_cols: Optional[List[str]] = None,
        risk_free: str = 'br_cdi_index'
    ):
        """
        Initialize the style analysis with return data.
        
        Args:
            returns_data: DataFrame with returns for funds and factors
            fund_cols: List of column names for funds
            factor_cols: List of column names for factors
            risk_free: Column name of risk-free rate
        """
        self.returns = returns_data
        
        # If fund_cols and factor_cols are not provided, try to infer them
        if fund_cols is None and factor_cols is None:
            # Default factors if not provided
            default_factors = helpers.COLS_FACTOR_DEFAULT
            self.factor_cols = [col for col in self.returns.columns if col in default_factors]
            self.fund_cols = [col for col in self.returns.columns if col not in self.factor_cols]
        else:
            self.fund_cols = fund_cols or []
            self.factor_cols = factor_cols or []
            
        self.risk_free = risk_free
        
        # Validate that columns exist in the data
        missing_funds = [col for col in self.fund_cols if col not in self.returns.columns]
        missing_factors = [col for col in self.factor_cols if col not in self.returns.columns]
        
        if missing_funds:
            raise ValueError(f"Fund columns not found in data: {missing_funds}")
        if missing_factors:
            raise ValueError(f"Factor columns not found in data: {missing_factors}")
        if self.risk_free not in self.returns.columns:
            raise ValueError(f"Risk-free rate column '{self.risk_free}' not found in data")
    
    def _find_optimal_window(
        self, 
        fund: str, 
        date_idx: int, 
        min_window: int = 10, 
        max_window: int = 50,
        vif_threshold: float = 10.0
    ) -> int:
        """
        Find the optimal regression window size based on adjusted R-squared and VIF.
        
        Args:
            fund: Fund column name
            date_idx: Index position for the current date
            min_window: Minimum window size to consider
            max_window: Maximum window size to consider
            vif_threshold: Threshold for Variance Inflation Factor
            
        Returns:
            Optimal window size
        """
        df_window_opt = pd.DataFrame()
        
        for window in range(min_window, max_window + 1):
            # print(f"Analyzing window size: {window}")
            # Check if we have enough data
            if date_idx - window < 0:
                continue
                
            # Get the excess returns
            y = self.returns[fund].iloc[date_idx - window:date_idx] - self.returns[self.risk_free].iloc[date_idx - window:date_idx]
            
            # Get the factor returns (excluding risk-free)
            X_cols = [col for col in self.factor_cols if col != self.risk_free]
            X = self.returns[X_cols].iloc[date_idx - window:date_idx]
            X = sm.add_constant(X)
            
            # Check for multicollinearity using VIF
            vif = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
            
            # Run regression
            model = sm.OLS(y, X).fit()
            
            # Store results
            df_window_opt = pd.concat(
                [df_window_opt, pd.DataFrame({
                    'window': window,
                    'rsquared_adj': model.rsquared_adj,
                    'vif_above_threshold': sum(v > vif_threshold for v in vif)
                }, index=[0])],
                ignore_index=True
            )
        
        if df_window_opt.empty:
            return min_window
            
        # Sort by VIF (ascending) and R-squared (descending)
        window_opt = int(
            df_window_opt.sort_values(['vif_above_threshold', 'rsquared_adj'], 
                                      ascending=[True, False]).iloc[0]['window']
        )
        
        return window_opt
    
    def run_analysis(
        self, 
        min_window: int = 10, 
        max_window: int = 50,
        vif_threshold: float = 10.0,
        funds: Optional[List[str]] = None,
        most_recent_only: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run style analysis for the specified funds.
        
        Args:
            min_window: Minimum window size for analysis
            max_window: Maximum window size for analysis
            vif_threshold: Threshold for Variance Inflation Factor
            funds: List of funds to analyze (if None, analyzes all funds)
            most_recent_only: If True, only analyze the most recent date
            
        Returns:
            Dictionary of results DataFrames for each fund
        """
        funds = funds or self.fund_cols
        results = {}
        
        for fund in funds:
            logger.info(f"Analyzing {fund}...")
            
            output = pd.DataFrame()
            
            # Determine the date range to analyze
            if most_recent_only:
                # Only analyze the most recent date
                date_indices = [len(self.returns) - 1]
            else:
                # Analyze the entire period
                date_indices = range(min_window, len(self.returns))
            
            for i in date_indices:
                current_date = self.returns.index[i]
                if not most_recent_only:
                    print(current_date)
                
                # Find optimal window size
                window_size = self._find_optimal_window(
                    fund, i, min_window, max_window, vif_threshold
                )
                
                # Get excess returns
                y = (self.returns[fund].iloc[i - window_size:i] - 
                     self.returns[self.risk_free].iloc[i - window_size:i])
                
                # Get factor returns (excluding risk-free)
                X_cols = [col for col in self.factor_cols if col != self.risk_free]
                X = self.returns[X_cols].iloc[i - window_size:i]
                X = sm.add_constant(X)
                
                # Run regression
                model = sm.OLS(y, X).fit()
                
                # Store results
                temp_output = pd.concat([
                    model.params.to_frame().T.rename(index={0: 'beta'}),
                    model.pvalues.to_frame().T.rename(index={0: 'pvalue'}),
                    pd.DataFrame({'rsquared': [model.rsquared], 
                                 'rsquared_adj': [model.rsquared_adj],
                                 'window': [window_size]}, index=['model'])
                ])
                
                temp_output['date'] = current_date
                temp_output['fund'] = fund
                
                output = pd.concat([output, temp_output], ignore_index=False)
            
            # Reshape the output for better usability
            output = output.round(5)
            
            # Store in results dictionary
            results[fund] = output
            
        return results
    
    def save_results(self, results: Dict[str, pd.DataFrame], table_name: str = 'fundos_style_analysis'):
        """
        Save results to SQL database.
        
        Args:
            results: Dictionary of results DataFrames
            table_name: Name of the database table
        """
            
        # Format the data for SQL
        for fund, df in results.items():
            # Prepare the data for SQL
            for metric in ['beta', 'pvalue']:
                subset = df[df.index == metric].drop(columns=['rsquared', 'rsquared_adj', 'window'])
                subset['field'] = metric
                
                # Upload to database
                to_sql(
                    subset, table_name=table_name,
                    primary_keys=['fund', 'date', 'field'],
                    update=True, batch_size=5000
                )
                
        print(f"Results saved to {table_name} table")