import os
import glob
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# PARAMETERS
DATA_DIR = 'INSERT DATA PATH HERE'
RESULTS_DIR = 'INSERT OUTPUT PATH HERE'
VALIDATION_LENGTH = 12
START_YEAR = 1957
END_YEAR = 2016

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Data
full_data = pd.concat([pd.read_parquet(f).assign(year=int(os.path.basename(f).split('_')[1].split('.')[0]))
                       for f in glob.glob(os.path.join(DATA_DIR, 'year_*.parquet'))], ignore_index=True)
print(f"Loaded data shape: {full_data.shape}")

# Features and Target
TARGET = 'ret_excess'
DROP_COLS = ['ret_excess', 'year', 'month', 'permno']
FEATURES = [col for col in full_data.columns if col not in DROP_COLS]
full_data = full_data.dropna(subset=[TARGET])

# Remove zero-variance columns
zero_var_cols = full_data[FEATURES].nunique()[full_data[FEATURES].nunique() <= 1].index.tolist()
FEATURES = [col for col in FEATURES if col not in zero_var_cols]

# Estimation Periods
def create_estimation_periods(start_year, end_year, validation_length):
    return pd.DataFrame([{
        'oos_year': y,
        'training_start': start_year,
        'training_end': y - validation_length - 1,
        'validation_start': y - validation_length,
        'validation_end': y - 1
    } for y in range(1987, end_year + 1)])

estimation_periods = create_estimation_periods(START_YEAR, END_YEAR, VALIDATION_LENGTH)

for _, period in estimation_periods.iterrows():
    print(f"Processing OOS Year: {period.oos_year}")
    year_folder = os.path.join(RESULTS_DIR, str(period.oos_year))
    os.makedirs(year_folder, exist_ok=True)

    train = full_data[(full_data['year'] >= period.training_start) & (full_data['year'] <= period.training_end)]
    oos = full_data[full_data['year'] == period.oos_year]

    X_train, y_train = train[FEATURES].fillna(0), train[TARGET]
    X_oos, y_oos = oos[FEATURES].fillna(0), oos[TARGET]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_oos = scaler.transform(X_oos)

    model = SGDRegressor(loss='huber', penalty=None, max_iter=50)
    model.fit(X_train, y_train)

    y_pred_oos = model.predict(X_oos)
    r2_oos = r2_score(y_oos, y_pred_oos)

    pd.DataFrame({'permno': oos['permno'], 'month': oos['month'], 'y_true': y_oos, 'y_pred': y_pred_oos})\
      .to_csv(os.path.join(year_folder, 'oos_predictions.csv'), index=False)

    pos = np.where(y_pred_oos >= np.quantile(y_pred_oos, 0.7), 1, np.where(y_pred_oos <= np.quantile(y_pred_oos, 0.3), -1, 0))
    port_ret = pos * y_oos.values

    pd.DataFrame({'month': oos['month'], 'portfolio_return': port_ret})\
      .to_csv(os.path.join(year_folder, 'portfolio.csv'), index=False)

    top20_idx = np.argsort(np.abs(model.coef_))[-20:][::-1]
    pd.DataFrame({'feature': np.array(FEATURES)[top20_idx], 'coefficient': model.coef_[top20_idx]})\
      .to_csv(os.path.join(year_folder, 'top_20_predictors.csv'), index=False)

    sharpe = port_ret.mean() / port_ret.std() if port_ret.std() != 0 else np.nan
    pd.DataFrame({'sharpe_ratio': [sharpe]}).to_csv(os.path.join(year_folder, 'sharpe_ratio.csv'), index=False)
    pd.DataFrame({'oos_r2': [r2_oos]}).to_csv(os.path.join(year_folder, 'oos_r2.csv'), index=False)

    del train, oos, X_train, X_oos, y_train, y_oos, scaler, model, pos, port_ret, y_pred_oos
    gc.collect()
    gc.collect()

print("Finished all OOS years.")
