import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
DATA_DIR = Path("INSERT DATA PATH HERE")
N_COMPONENTS = 20
START_YEAR = 1957
END_YEAR = 2016

results = []

for year in range(START_YEAR, END_YEAR + 1):
    file_path = DATA_DIR / f"mini_{year}.parquet"
    print(f"Processing {file_path.name}...")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Could not read file: {e}")
        continue

    df = df.sort_values(['permno', 'month'])

    # Target = already-aligned next month's excess return
    df['target'] = df['ret_excess']

    # Use only firm-level predictors
    feature_cols = [col for col in df.columns if col.startswith('characteristic_')]
    if not feature_cols:
        print(f"No firm-level predictors found in {year}")
        continue

    X = df[feature_cols]
    y = df['target'].values
    months = df['month'].values

    if len(X) < 100:
        print(f"Skipping year {year} due to small sample size ({len(X)})")
        continue

    # === Train/Test split (80/20 by row count) ===
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    months_test = months[split_idx:]

    # === Standardize predictors ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === PCA + Linear Regression ===
    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    # === Monthly cross-sectional R² (naive forecast = 0) ===
    test_df = pd.DataFrame({
        'month': months_test,
        'y_true': y_test,
        'y_pred': y_pred
    })

    monthly_r2s = []
    for month, group in test_df.groupby('month'):
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot_naive = np.sum(y_true ** 2)
        if ss_tot_naive > 0:
            r2_m = 1 - ss_res / ss_tot_naive
            monthly_r2s.append(r2_m)

    r2_final = np.mean(monthly_r2s) if monthly_r2s else np.nan
    results.append({'year': year, 'r2': r2_final})
    print(f"Year {year}: Avg. R² = {r2_final:.6f}")

# === Save and summarize results ===
df_results = pd.DataFrame(results)

avg_r2 = df_results['r2'].mean()
print(f"\nFinal Average R² (1957–2016): {avg_r2:.6f}")

avg_row = pd.DataFrame([{'year': 'Average', 'r2': avg_r2}])
df_results = pd.concat([df_results, avg_row], ignore_index=True)

df_results.to_csv("pcr_results.csv", index=False)
print("Saved results to pcr_results.csv")
