import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Union, Optional, Tuple
from persevera_tools.db import to_sql
from persevera_style_analysis.utils import helpers

logger = logging.getLogger(__name__)

class BestSubsetStyleAnalysis:
    """
    Style analysis implementation using Best Subset Selection with rolling regressions.
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
    
    def _best_subset_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'adjr2'
    ) -> tuple:
        """
        Run all subsets regression and return the best model.
        
        Args:
            X: Predictor variables DataFrame (with constant)
            y: Target variable Series
            metric: Selection criterion ('aic', 'bic', 'adjr2')
            
        Returns:
            Tuple of (best model, list of included column indices)
        """
        # Define scoring function based on metric
        if metric == 'aic':
            score_func = lambda m: m.aic
            is_better = lambda new, old: new < old
            best_score = np.inf
        elif metric == 'bic':
            score_func = lambda m: m.bic
            is_better = lambda new, old: new < old
            best_score = np.inf
        elif metric == 'adjr2':
            score_func = lambda m: m.rsquared_adj
            is_better = lambda new, old: new > old
            best_score = -np.inf
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        best_model = None
        best_subset = None
        
        # First column is always constant, so skip it in combinations
        n_features = X.shape[1] - 1
        
        # Loop through all possible combinations of features (excluding constant)
        for k in range(1, n_features + 1):
            # Limit to manageable number for large feature sets
            # if n_features > 10 and k > 5:
            #     break
                
            for subset_idx in itertools.combinations(range(n_features), k):
                # Always include constant (index 0) and add selected features
                columns = [0] + [idx + 1 for idx in subset_idx]  # +1 to account for constant
                X_subset = X.iloc[:, columns]
                
                # Fit model
                model = sm.OLS(y, X_subset).fit()
                
                # Calculate score
                score = score_func(model)
                
                # Update best model if better
                if best_model is None or is_better(score, best_score):
                    best_score = score
                    best_model = model
                    best_subset = subset_idx
        
        # Return best model and the subset
        # Get column indices from original X (excluding constant)
        included_cols = list(best_subset) if best_subset is not None else []
        
        return best_model, included_cols
    
    def _find_optimal_window(
        self, 
        fund: str, 
        date_idx: int, 
        min_window: int = 10, 
        max_window: int = 50,
        vif_threshold: float = 10.0,
        selection_metric: str = 'adjr2'
    ) -> int:
        """
        Find the optimal regression window size based on best subset selection.
        
        Args:
            fund: Fund column name
            date_idx: Index position for the current date
            min_window: Minimum window size to consider
            max_window: Maximum window size to consider
            vif_threshold: Threshold for Variance Inflation Factor
            selection_metric: Criterion for model selection ('aic', 'bic', or 'adjr2')
            
        Returns:
            Optimal window size
        """
        logger.info(f"\n=== DEBUGGING BEST SUBSET - OPTIMAL WINDOW SELECTION FOR {fund} ===")
        logger.info(f"Selection metric: {selection_metric}")
        
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
            
            # Run best subset regression instead of full regression
            best_model, included_indices = self._best_subset_regression(X, y, selection_metric)
            
            # Calculate VIFs only for the included features (if any)
            if included_indices:
                X_subset_cols = [0] + [idx + 1 for idx in included_indices]  # +1 to account for constant
                X_subset = X.iloc[:, X_subset_cols]
                vif_above_threshold = sum(variance_inflation_factor(X_subset.values, i) > vif_threshold 
                                         for i in range(len(X_subset.columns)))
                
                logger.info("Included factors:")
                for idx in included_indices:
                    factor_name = X.columns[idx + 1]
                    logger.info(f"  - {factor_name}")
            else:
                # If no features were selected, set VIF count to 0
                vif_above_threshold = 0
                logger.info("No factors were selected")
            
            # Define the scoring metric based on the selection criterion
            if selection_metric == 'aic':
                score = best_model.aic
                better_score = score < df_window_opt['score'].min() if not df_window_opt.empty else True
                logger.info(f"AIC: {score:.4f}")
            elif selection_metric == 'bic':
                score = best_model.bic
                better_score = score < df_window_opt['score'].min() if not df_window_opt.empty else True
                logger.info(f"BIC: {score:.4f}")
            else:  # 'adjr2'
                score = best_model.rsquared_adj
                better_score = score > df_window_opt['score'].max() if not df_window_opt.empty else True
                logger.info(f"Adj R²: {score:.4f}")
            
            logger.info(f"R²: {best_model.rsquared:.4f}")
            logger.info(f"Adj R²: {best_model.rsquared_adj:.4f}")
            logger.info(f"VIF above threshold: {vif_above_threshold}")
            logger.info(f"Number of included factors: {len(included_indices)}")

            # Store results
            df_window_opt = pd.concat(
                [df_window_opt, pd.DataFrame({
                    'window': window,
                    'score': score,
                    'rsquared_adj': best_model.rsquared_adj,
                    'vif_above_threshold': vif_above_threshold,
                    'included_factors': len(included_indices)
                }, index=[0])],
                ignore_index=True
            )
        
        logger.info("\n=== WINDOW SELECTION SUMMARY ===")
        logger.info(df_window_opt.sort_values('window').to_string())
        
        if df_window_opt.empty:
            logger.info(f"No suitable windows found. Using minimum window: {min_window}")
            return min_window
            
        # Sort by VIF (ascending) and selection criterion
        if selection_metric in ['aic', 'bic']:
            # Lower is better for AIC/BIC
            window_opt = int(
                df_window_opt.sort_values(['vif_above_threshold', 'score'], 
                                        ascending=[True, True]).iloc[0]['window']
            )
            logger.info("\nSorted by VIF (ascending) and score (ascending):")
        else:
            # Higher is better for adjusted R-squared
            window_opt = int(
                df_window_opt.sort_values(['vif_above_threshold', 'score'], 
                                        ascending=[True, False]).iloc[0]['window']
            )
            logger.info("\nSorted by VIF (ascending) and score (descending):")
            
        logger.info(df_window_opt.sort_values(['vif_above_threshold', 'score'], 
                                    ascending=[True, False if selection_metric == 'adjr2' else True]).head(10).to_string())
        
        logger.info(f"\nSelected optimal window size: {window_opt}")
        logger.info("========================================\n")
        
        return window_opt
    
    def run_analysis(
        self, 
        min_window: int = 10, 
        max_window: int = 50,
        vif_threshold: float = 10.0,
        selection_metric: str = 'bic',
        funds: Optional[List[str]] = None,
        most_recent_only: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run style analysis for the specified funds using best subset selection.
        
        Args:
            min_window: Minimum window size for analysis
            max_window: Maximum window size for analysis
            vif_threshold: Threshold for Variance Inflation Factor
            selection_metric: Criterion for model selection ('aic', 'bic', or 'adjr2')
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
                
                # Find optimal window size using best subset selection
                window_size = self._find_optimal_window(
                    fund, i, min_window, max_window, vif_threshold, selection_metric
                )
                
                # Get excess returns
                y = (self.returns[fund].iloc[i - window_size:i] - 
                     self.returns[self.risk_free].iloc[i - window_size:i])
                
                # Get factor returns (excluding risk-free)
                X_cols = [col for col in self.factor_cols if col != self.risk_free]
                X = self.returns[X_cols].iloc[i - window_size:i]
                X = sm.add_constant(X)
                
                # Run best subset regression with the optimal window
                model, included_indices = self._best_subset_regression(X, y, selection_metric)
                
                # Create parameter dictionary including all original factors
                # Initialize with zeros
                param_dict = {'const': model.params.iloc[0]}
                pvalue_dict = {'const': model.pvalues.iloc[0]}
                
                # Map from column indices to factor names
                included_factors = []
                for idx, param_idx in enumerate(included_indices):
                    col_name = X.columns[param_idx + 1]  # +1 to skip constant
                    param_dict[col_name] = model.params.iloc[idx + 1]  # +1 to skip constant
                    pvalue_dict[col_name] = model.pvalues.iloc[idx + 1]
                    included_factors.append(col_name)
                
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
                        'included_factors': [len(included_indices)],
                        'selection_metric': [selection_metric]
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
    
    def save_results(self, results: Dict[str, pd.DataFrame], table_name: str = 'fundos_best_subset_style_analysis'):
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
                subset = df[df.index == metric].drop(columns=['rsquared', 'rsquared_adj', 'window', 
                                                             'included_factors', 'selection_metric'])
                subset['field'] = metric
                
                # Upload to database
                to_sql(
                    subset, table_name=table_name,
                    primary_keys=['fund', 'date', 'field'],
                    update=True, batch_size=5000
                )
                
        logger.info(f"Results saved to {table_name} table") 