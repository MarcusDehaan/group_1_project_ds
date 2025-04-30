import os
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Paths
data_folder = os.path.expanduser('INSERT DATA PATH HERE')
results_folder = os.path.expanduser('INSERT OUTPUT PATH HERE')
os.makedirs(results_folder, exist_ok=True)

# Settings
oos_years = list(range(1987, 2017))
validation_window = 12  # 12 years back

def load_data(years):
    dfs = []
    for year in years:
        path = os.path.join(data_folder, f'year_{year}.parquet')
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True)

# GBRT Hyperparameters (Gu, Kelly, Xiu style)
gbrt_params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.002,
    'n_estimators': 5000,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_samples': 10,
    'max_depth': 4,
    'random_state': 42,
    'n_jobs': -1
}

for oos_year in oos_years:
    print(f"\n=== OOS Year: {oos_year} ===")
    train_years = list(range(oos_year - validation_window, oos_year))
    
    # Load data
    df_train = load_data(train_years)
    df_test = load_data([oos_year])

    # Prepare features and target
    feature_cols = [col for col in df_train.columns if col not in ['permno', 'month', 'ret_excess']]
    target_col = 'ret_excess'

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    # Optional: Split a small validation set
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Train GBRT model
    model = lgb.LGBMRegressor(**gbrt_params)
    model.fit(
        X_train_main, y_train_main,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100)]
    )

    # Predictions
    oos_preds = model.predict(X_test)

    # Save OOS predictions
    oos_df = df_test[['permno', 'month']].copy()
    oos_df['y_true'] = y_test.values
    oos_df['y_pred'] = oos_preds
    oos_df.to_csv(os.path.join(results_folder, f'{oos_year}_oos_predictions.csv'), index=False)

    # Portfolio sorting: Top 20% minus Bottom 20%
    df_test['y_pred'] = oos_preds
    df_test['pred_qcut'] = pd.qcut(df_test['y_pred'], 5, labels=False)
    long = df_test[df_test['pred_qcut'] == 4]['ret_excess'].mean()
    short = df_test[df_test['pred_qcut'] == 0]['ret_excess'].mean()
    portfolio_return = long - short

    # Save portfolio returns
    portfolio_df = pd.DataFrame({'portfolio_return': [portfolio_return]})
    portfolio_df.to_csv(os.path.join(results_folder, f'{oos_year}_portfolio.csv'), index=False)

    # Variable importance (top 20)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    importance_df.head(20).to_csv(os.path.join(results_folder, f'{oos_year}_top_20_predictors.csv'), index=False)

    # Sharpe ratio
    sharpe_ratio = df_test['ret_excess'].mean() / df_test['ret_excess'].std()
    sharpe_df = pd.DataFrame({'sharpe_ratio': [sharpe_ratio]})
    sharpe_df.to_csv(os.path.join(results_folder, f'{oos_year}_sharpe_ratio.csv'), index=False)

    # OOS R2
    r2 = r2_score(y_test, oos_preds)
    r2_df = pd.DataFrame({'oos_r2': [r2]})
    r2_df.to_csv(os.path.join(results_folder, f'{oos_year}_oos_r2.csv'), index=False)

    del df_train, df_test, X_train, y_train, X_test, y_test, X_train_main, X_val, y_train_main, y_val, model, oos_df
    gc.collect()

print("\nGBRT Training Complete!")