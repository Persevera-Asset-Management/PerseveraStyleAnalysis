import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Union, Optional, Tuple
from persevera_tools.db import to_sql
from persevera_tools.utils.logging import get_logger

logger = get_logger(__name__)


class StepwiseStyleAnalysis:
    """
    Style analysis implementation using Backward Stepwise Regression with rolling windows.
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
            default_factors = ['br_cdi_index', 'br_ibovespa', 'brl_usd', 'us_sp500', 
                               'crb_index', 'br_pre_2y', 'us_generic_10y']
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
    
    def _backward_stepwise(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold_out: float = 0.05
    ) -> tuple:
        """
        Perform backward stepwise regression.
        
        Args:
            X: Predictor variables DataFrame (with constant)
            y: Target variable Series
            threshold_out: Significance threshold for removing variables
            
        Returns:
            Tuple of (final model, list of included column indices)
        """
        # Start with all columns (excluding constant)
        included = list(range(1, X.shape[1]))  # Skip constant
        X_column_names = list(X.columns)
        
        while True:
            # Check if we have any variables left
            if not included:
                break
                
            # Create model with included columns
            X_subset = X.iloc[:, [0] + included]  # Always include constant (first column)
            model = sm.OLS(y, X_subset).fit()
            
            # Look at p-values (excluding constant)
            pvalues = model.pvalues[1:]
            max_pvalue = pvalues.max()
            
            # If maximum p-value exceeds threshold, remove that variable
            if max_pvalue > threshold_out:
                # Get the name of the column with the highest p-value
                worst_col_name = pvalues.idxmax()
                
                # Find the index of this column in the original X dataframe
                worst_col_idx = X_column_names.index(worst_col_name)
                
                # Find and remove this index from the included list
                if worst_col_idx in included:
                    included.remove(worst_col_idx)
            else:
                break
        
        # Return final model and included variables
        X_final = X.iloc[:, [0] + included]
        return sm.OLS(y, X_final).fit(), included
    
    def _find_optimal_window(
        self, 
        fund: str, 
        date_idx: int, 
        min_window: int = 10, 
        max_window: int = 50,
        vif_threshold: float = 10.0,
        threshold_out: float = 0.05
    ) -> int:
        """
        Find the optimal regression window size based on backward stepwise regression.
        
        Args:
            fund: Fund column name
            date_idx: Index position for the current date
            min_window: Minimum window size to consider
            max_window: Maximum window size to consider
            vif_threshold: Threshold for Variance Inflation Factor
            threshold_out: Significance threshold for removing variables
            
        Returns:
            Optimal window size
        """
        logger.info(f"\n=== DEBUGGING STEPWISE - OPTIMAL WINDOW SELECTION FOR {fund} ===")
        logger.info(f"Significance threshold for variable removal: {threshold_out}")
        
        df_window_opt = pd.DataFrame()
        
        for window in range(min_window, max_window + 1):
            logger.info(f"\n--- Window size: {window} ---")
            # Check if we have enough data
            if date_idx - window < 0:
                logger.info(f"Window size {window} is too large for available data.")
                continue
                
            # Get the excess returns
            y = self.returns[fund].iloc[date_idx - window:date_idx] - self.returns[self.risk_free].iloc[date_idx - window:date_idx]
            
            # Get the factor returns (excluding risk-free)
            X_cols = [col for col in self.factor_cols if col != self.risk_free]
            X = self.returns[X_cols].iloc[date_idx - window:date_idx]
            X = sm.add_constant(X)
            
            # Run backward stepwise regression instead of full regression
            best_model, included_indices = self._backward_stepwise(X, y, threshold_out)
            
            # Calculate VIFs only for the included features
            if included_indices:
                X_subset = X.iloc[:, [0] + included_indices]  # Include constant and selected features
                vif_above_threshold = sum(variance_inflation_factor(X_subset.values, i) > vif_threshold 
                                         for i in range(len(X_subset.columns)))
                
                logger.info("Included factors:")
                for idx in included_indices:
                    factor_name = X.columns[idx]
                    logger.info(f"  - {factor_name}")
            else:
                # If no features were selected, set VIF count to 0
                vif_above_threshold = 0
                logger.info("No factors were selected")
            
            logger.info(f"R²: {best_model.rsquared:.4f}")
            logger.info(f"Adj R²: {best_model.rsquared_adj:.4f}")
            logger.info(f"VIF above threshold: {vif_above_threshold}")
            logger.info(f"Number of included factors: {len(included_indices)}")
            
            result_dict = {
                'window': window,
                'rsquared_adj': best_model.rsquared_adj,
                'vif_above_threshold': vif_above_threshold,
                'included_factors': len(included_indices)
            }
            
            # Store results
            df_window_opt = pd.concat(
                [df_window_opt, pd.DataFrame(result_dict, index=[0])],
                ignore_index=True
            )
        
        logger.info("\n=== WINDOW SELECTION SUMMARY ===")
        logger.info(df_window_opt.sort_values('window').to_string())
        
        if df_window_opt.empty:
            logger.info(f"No suitable windows found. Using minimum window: {min_window}")
            return min_window
            
        # Sort by VIF (ascending) and R-squared (descending)
        window_opt = int(
            df_window_opt.sort_values(['vif_above_threshold', 'rsquared_adj'], 
                                      ascending=[True, False]).iloc[0]['window']
        )
        
        logger.info("\nSorted by VIF (ascending) and Adj R² (descending):")
        logger.info(df_window_opt.sort_values(['vif_above_threshold', 'rsquared_adj'], 
                                     ascending=[True, False]).head(10).to_string())
        
        logger.info(f"\nSelected optimal window size: {window_opt}")
        logger.info("========================================\n")
        
        return window_opt
    
    def run_analysis(
        self, 
        min_window: int = 10, 
        max_window: int = 50,
        vif_threshold: float = 10.0,
        threshold_out: float = 0.05,
        funds: Optional[List[str]] = None,
        most_recent_only: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run style analysis for the specified funds using backward stepwise regression.
        
        Args:
            min_window: Minimum window size for analysis
            max_window: Maximum window size for analysis
            vif_threshold: Threshold for Variance Inflation Factor
            threshold_out: Significance threshold for removing variables
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
                    logger.info(current_date)
                
                # Find optimal window size using backward stepwise regression
                window_size = self._find_optimal_window(
                    fund, i, min_window, max_window, vif_threshold, threshold_out
                )
                
                # Get excess returns
                y = (self.returns[fund].iloc[i - window_size:i] - 
                     self.returns[self.risk_free].iloc[i - window_size:i])
                
                # Get factor returns (excluding risk-free)
                X_cols = [col for col in self.factor_cols if col != self.risk_free]
                X = self.returns[X_cols].iloc[i - window_size:i]
                X = sm.add_constant(X)
                
                # Run backward stepwise regression with the optimal window
                model, included_indices = self._backward_stepwise(X, y, threshold_out)
                
                # Create parameter dictionary including all original factors
                # Initialize with zeros
                param_dict = {'const': model.params.iloc[0]}
                pvalue_dict = {'const': model.pvalues.iloc[0]}
                
                # Fill in parameters for included factors
                for idx, factor_idx in enumerate(included_indices):
                    col_name = X.columns[factor_idx]  # Direct lookup by index
                    param_idx = idx + 1  # +1 to skip constant
                    param_dict[col_name] = model.params.iloc[param_idx]
                    pvalue_dict[col_name] = model.pvalues.iloc[param_idx]
                
                # Fill missing factors with zero
                for col in X_cols:
                    if col not in param_dict:
                        param_dict[col] = 0
                        pvalue_dict[col] = 1.0  # Max p-value for excluded variables
                
                # Convert to Series and add metadata
                beta_series = pd.Series(param_dict, name='beta')
                pvalue_series = pd.Series(pvalue_dict, name='pvalue')
                
                # Store results
                temp_output = pd.concat([
                    beta_series.to_frame().T,
                    pvalue_series.to_frame().T,
                    pd.DataFrame({
                        'rsquared': [model.rsquared], 
                        'rsquared_adj': [model.rsquared_adj],
                        'window': [window_size],
                        'included_factors': [len(included_indices)]
                    }, index=['model'])
                ])
                
                temp_output['date'] = current_date
                temp_output['fund'] = fund
                
                output = pd.concat([output, temp_output], ignore_index=False)
            
            # Reshape the output for better usability
            output = output.round(5)
            
            # Store in results dictionary
            results[fund] = output
            
        return results
    
    def save_results(self, results: Dict[str, pd.DataFrame], table_name: str = 'fundos_stepwise_style_analysis'):
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
                subset = df[df.index == metric].drop(columns=['rsquared', 'rsquared_adj', 'window', 'included_factors'])
                subset['field'] = metric
                
                # Upload to database
                to_sql(
                    subset, table_name=table_name,
                    primary_keys=['fund', 'date', 'field'],
                    update=True, batch_size=5000
                )
                
        logger.info(f"Results saved to {table_name} table") 