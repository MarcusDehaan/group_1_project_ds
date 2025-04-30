#!/usr/bin/env python3
"""
Script to run a RAM-efficient GLM model on yearly Parquet files (1957–2016),
replicating Kelly, Gu, and Xiu (2020) using all predictors (~600).
Reduces computational load with low maxiter and single train-test split.
Avoids recursion issues with matrix-based GLM. Optimized for 8 cores, 64GB RAM.
"""
import os
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import gc
import logging

# Config
DATA_DIR = "INSERT DATA PATH HERE"
RESULTS_DIR = "INSERT OUTPUT PATH HERE"
RESULTS_FILE = os.path.join(RESULTS_DIR, "year_results.csv")
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.csv")
START_YEAR = 1957
END_YEAR = 2016
TARGET = 'ret_excess'
TEST_SIZE = 0.2
TOP_N_PREDICTORS = 5
RISK_FREE_RATE = 0.0
PERIODS_PER_YEAR = 12  # Monthly data
MAX_ITER = 20  # Reduced for efficiency
VARIANCE_THRESHOLD = 1e-5  # Remove very low-variance predictors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_predictors(parquet_file):
    """Dynamically identify predictor columns, excluding non-predictors."""
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    exclude_cols = [
        'permno', 'month', 'mktcap_lag', 'ret_excess',
        '__index_level_0__', '__fragment_index', '__batch_index',
        '__last_in_fragment', '__filename'
    ]
    predictors = [col for col in df.columns if col not in exclude_cols]
    del df
    gc.collect()
    return predictors

def setup_results_files():
    """Create or clear results and metrics files with headers."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    if not os.path.exists(RESULTS_FILE):
        pd.DataFrame(columns=['year', 'r2', 'sharpe_ratio']).to_csv(RESULTS_FILE, index=False)
    
    if not os.path.exists(METRICS_FILE):
        pd.DataFrame(columns=[
            'year', 'mse', 'r2', 'tss', 'ess', 'sharpe_ratio', 'portfolio_return',
            'hit_ratio', 'n_observations', 'n_predictors'
        ]).to_csv(METRICS_FILE, index=False)
    
    logger.info(f"Results will be saved to {RESULTS_FILE}")
    logger.info(f"Metrics will be saved to {METRICS_FILE}")

def compute_portfolio_metrics(y_true, y_pred, year):
    """Compute long-short portfolio returns, Sharpe ratio, and hit ratio."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values('y_pred', ascending=False)
    top_quintile = df.iloc[:int(len(df) * 0.2)]
    bottom_quintile = df.iloc[-int(len(df) * 0.2):]
    
    portfolio_return = top_quintile['y_true'].mean() - bottom_quintile['y_true'].mean()
    annualized_return = portfolio_return * PERIODS_PER_YEAR
    portfolio_std = (top_quintile['y_true'] - bottom_quintile['y_true']).std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / portfolio_std if portfolio_std > 0 else np.nan
    hit_ratio = (top_quintile['y_true'] > bottom_quintile['y_true']).mean()
    
    return portfolio_return, sharpe_ratio, hit_ratio

def fit_glm(train_df, predictors):
    """Fit matrix-based GLM with reduced maxiter."""
    try:
        X_train = train_df[predictors]
        # Remove low-variance predictors
        selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        X_train = selector.fit_transform(X_train)
        selected_predictors = [predictors[i] for i in selector.get_support(indices=True)]
        
        X_train = sm.add_constant(X_train)  # Add intercept
        y_train = train_df[TARGET]
        model = sm.GLM(y_train, X_train, family=sm.families.Gaussian()).fit(maxiter=MAX_ITER, cov_type='nonrobust')
        return model, selected_predictors
    except Exception as e:
        logger.error(f"GLM fitting failed: {str(e)}")
        return None, None

def process_year(year, predictors):
    """Process a single year's Parquet file and compute metrics."""
    parquet_file = os.path.join(DATA_DIR, f"year_{year}.parquet")
    
    if not os.path.exists(parquet_file):
        logger.warning(f"Parquet file for year {year} not found")
        return None
    
    df = None
    train_df = None
    test_df = None
    
    try:
        columns = [TARGET] + predictors
        df = pd.read_parquet(parquet_file, columns=columns, engine='pyarrow')
        logger.info(f"Loaded {len(df)} rows for year {year} with {len(predictors)} predictors")
        
        if df[columns].isnull().any().any():
            logger.warning(f"Missing values in year {year}, imputing with mean")
            df.fillna(df.mean(), inplace=True)
        
        train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
        
        # Fit GLM
        model, selected_predictors = fit_glm(train_df, predictors)
        if model is None or selected_predictors is None:
            return None
        
        # Predict
        X_test = test_df[selected_predictors]
        selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        X_test = selector.fit_transform(X_test)  # Apply same thresholding
        X_test = sm.add_constant(X_test)
        y_pred = model.predict(X_test)
        y_true = test_df[TARGET]
        
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        y_mean = y_true.mean()
        tss = ((y_true - y_mean) ** 2).sum()
        ess = ((y_pred - y_mean) ** 2).sum()
        
        portfolio_return, sharpe_ratio, hit_ratio = compute_portfolio_metrics(y_true, y_pred, year)
        
        p_values = model.pvalues[1:]  # Exclude intercept
        top_indices = np.argsort(p_values)[:TOP_N_PREDICTORS]
        top_predictors = [selected_predictors[i] for i in top_indices]
        top_p_values = p_values[top_indices]
        
        predictions_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        })
        predictions_file = os.path.join(RESULTS_DIR, f"predictions_{year}.csv")
        predictions_df.to_csv(predictions_file, index=False)
        
        predictors_df = pd.DataFrame({
            'predictor': top_predictors,
            'p_value': top_p_values
        })
        predictors_file = os.path.join(RESULTS_DIR, f"top_predictors_{year}.csv")
        predictors_df.to_csv(predictors_file, index=False)
        
        portfolio_df = pd.DataFrame({
            'year': [year],
            'portfolio_return': [portfolio_return]
        })
        portfolio_file = os.path.join(RESULTS_DIR, f"portfolio_{year}.csv")
        portfolio_df.to_csv(portfolio_file, index=False)
        
        logger.info(f"Year {year}: Out-of-sample R² = {r2:.6f}, Sharpe = {sharpe_ratio:.6f}")
        
        return {
            'year': year,
            'r2': r2,
            'sharpe_ratio': sharpe_ratio,
            'mse': mse,
            'tss': tss,
            'ess': ess,
            'portfolio_return': portfolio_return,
            'hit_ratio': hit_ratio,
            'n_observations': len(df),
            'n_predictors': len(selected_predictors),
            'top_predictors': top_predictors
        }
    
    except Exception as e:
        logger.error(f"Error processing year {year}: {str(e)}")
        return None
    finally:
        if df is not None:
            del df
        if train_df is not None:
            del train_df
        if test_df is not None:
            del test_df
        gc.collect()

def save_results(result):
    """Append results to CSV files."""
    if result is None:
        return
    
    result_df = pd.DataFrame({
        'year': [result['year']],
        'r2': [result['r2']],
        'sharpe_ratio': [result['sharpe_ratio']]
    })
    result_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    
    metrics_df = pd.DataFrame({
        'year': [result['year']],
        'mse': [result['mse']],
        'r2': [result['r2']],
        'tss': [result['tss']],
        'ess': [result['ess']],
        'sharpe_ratio': [result['sharpe_ratio']],
        'portfolio_return': [result['portfolio_return']],
        'hit_ratio': [result['hit_ratio']],
        'n_observations': [result['n_observations']],
        'n_predictors': [result['n_predictors']]
    })
    metrics_df.to_csv(METRICS_FILE, mode='a', header=False, index=False)

def main():
    logger.info("Starting GLM processing for yearly Parquet files (1957–2016)")
    
    setup_results_files()
    
    # Get predictors from the first available Parquet file
    first_parquet = os.path.join(DATA_DIR, f"year_{START_YEAR}.parquet")
    if not os.path.exists(first_parquet):
        logger.error(f"First Parquet file {first_parquet} not found")
        return
    predictors = get_predictors(first_parquet)
    logger.info(f"Using {len(predictors)} predictors")
    
    # Process years sequentially
    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"Processing year {year}")
        result = process_year(year, predictors)
        save_results(result)
    
    logger.info("Processing complete")
    
    try:
        results_df = pd.read_csv(RESULTS_FILE)
        metrics_df = pd.read_csv(METRICS_FILE)
        
        logger.info("\nFinal Out-of-Sample R² Statistics:")
        logger.info(f"Count: {len(results_df)}")
        logger.info(f"Min R²: {results_df['r2'].min():.6f}")
        logger.info(f"Max R²: {results_df['r2'].max():.6f}")
        logger.info(f"Mean R²: {results_df['r2'].mean():.6f}")
        
        logger.info("\nFinal Sharpe Ratio Statistics:")
        logger.info(f"Min Sharpe: {results_df['sharpe_ratio'].min():.6f}")
        logger.info(f"Max Sharpe: {results_df['sharpe_ratio'].max():.6f}")
        logger.info(f"Mean Sharpe: {results_df['sharpe_ratio'].mean():.6f}")
        
        logger.info("\nFinal Metrics Summary:")
        logger.info(metrics_df.describe().to_string())
    except Exception as e:
        logger.error(f"Error reading results: {e}")

if __name__ == "__main__":
    main()