import os
import glob
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# PARAMETERS
DATA_DIR = 'INSERT DATA PATH HERE'
RESULTS_DIR = 'INSERT OUTPUT PATH HERE'
VALIDATION_LENGTH = 12
START_YEAR = 1957
END_YEAR = 2021
PENALTY_GRID = [1e-4, 1e-2, 1e-1]
BATCH_SIZE = 10000  # mini-batch size for fitting

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Data
def load_data(data_dir):
    all_files = glob.glob(os.path.join(data_dir, 'year_*.parquet'))
    dfs = []
    for filename in all_files:
        year = int(os.path.basename(filename).split('_')[1].split('.')[0])
        df = pd.read_parquet(filename)
        df['year'] = year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

full_data = load_data(DATA_DIR)
print(f"Loaded data shape: {full_data.shape}")

# Features and Target
TARGET = 'ret_excess'
FEATURES = [col for col in full_data.columns if col not in ['ret_excess', 'year', 'month', 'permno']]
full_data = full_data.dropna(subset=[TARGET])

# Estimation Periods
def create_estimation_periods(start_year, end_year, validation_length):
    periods = []
    for oos_year in range(1987, end_year + 1):
        validation_end = oos_year - 1
        validation_start = validation_end - validation_length + 1
        training_start = start_year
        training_end = validation_start - 1

        periods.append({
            'oos_year': oos_year,
            'training_start': training_start,
            'training_end': training_end,
            'validation_start': validation_start,
            'validation_end': validation_end
        })
    return pd.DataFrame(periods)

estimation_periods = create_estimation_periods(START_YEAR, END_YEAR, VALIDATION_LENGTH)

# Prepare storage
oos_r2_scores = []

for idx, period in estimation_periods.iterrows():
    print(f"Processing OOS Year: {period.oos_year}")

    train = full_data[(full_data['year'] >= period.training_start) & (full_data['year'] <= period.training_end)]
    val = full_data[(full_data['year'] >= period.validation_start) & (full_data['year'] <= period.validation_end)]
    oos = full_data[full_data['year'] == period.oos_year]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_oos, y_oos = oos[FEATURES], oos[TARGET]

    # Impute missing
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_train.mean())
    X_oos = X_oos.fillna(X_train.mean())

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_oos_scaled = scaler.transform(X_oos)

    best_val_mse = np.inf
    best_model = None
    best_penalty = None

    for penalty in PENALTY_GRID:
        model = SGDRegressor(loss='huber', penalty='l1', alpha=penalty, learning_rate='invscaling', eta0=0.01, max_iter=1, warm_start=True)

        # Mini-batch training
        n_samples = X_train_scaled.shape[0]
        for epoch in range(3):  # 3 epochs through data
            idxs = np.random.permutation(n_samples)
            for batch_start in range(0, n_samples, BATCH_SIZE):
                batch_idx = idxs[batch_start:batch_start+BATCH_SIZE]
                model.partial_fit(X_train_scaled[batch_idx], y_train.values[batch_idx])

        val_preds = model.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, val_preds)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
            best_penalty = penalty

    # Predict OOS
    y_pred_oos = best_model.predict(X_oos_scaled)
    r2_oos = r2_score(y_oos, y_pred_oos)
    oos_r2_scores.append({'oos_year': period.oos_year, 'oos_r2': r2_oos})

    # Save year results immediately
    year_results = pd.DataFrame({
        'permno': oos['permno'],
        'month': oos['month'],
        'y_true': y_oos,
        'y_pred': y_pred_oos
    })
    year_results.to_csv(os.path.join(RESULTS_DIR, f'oos_predictions_{period.oos_year}.csv'), index=False)

    # Save portfolio
    oos_copy = oos.copy()
    oos_copy['y_pred'] = y_pred_oos
    threshold_long = oos_copy['y_pred'].quantile(0.7)
    threshold_short = oos_copy['y_pred'].quantile(0.3)
    oos_copy['position'] = np.where(oos_copy['y_pred'] >= threshold_long, 1,
                            np.where(oos_copy['y_pred'] <= threshold_short, -1, 0))
    oos_copy['portfolio_return'] = oos_copy['position'] * oos_copy['ret_excess']
    oos_copy[['month', 'portfolio_return']].to_csv(os.path.join(RESULTS_DIR, f'portfolio_{period.oos_year}.csv'), index=False)

    # Garbage collection
    del train, val, oos, X_train, X_val, X_oos, y_train, y_val, y_oos, scaler, model, best_model
    gc.collect()

# Save R2 scores
r2_df = pd.DataFrame(oos_r2_scores)
r2_df.to_csv(os.path.join(RESULTS_DIR, 'oos_r2_scores.csv'), index=False)
print("Finished all OOS years.")
