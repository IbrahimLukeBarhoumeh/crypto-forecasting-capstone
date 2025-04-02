"""
Finalize and persist best models for each coin+horizon based on advanced tuning results.

Usage:
  (base) $ cd /path/to/crypto-forecasting-capstone
  (base) $ python scripts/finalize_and_persist_models.py

Prerequisites:
- data/advanced_tuning_results.csv (or similar) with columns like:
    [Coin, Horizon, RF_RMSE, RF_MAE, XGB_RMSE, XGB_MAE, Best_Model, Best_RMSE, ...]
- models/final_feature_columns.json which maps e.g. "ADA__Close_t+7" -> ["Close","MA_20",...]
- data/{Coin}_features.csv for each coin, e.g. "ADA_features.csv"
- The 'models/' folder (created if missing) to store final .pkl files.

This version:
- Uses a manual RMSE calculation (manual_rmse) to avoid scikit-learn version issues.
- Expects your coin CSVs to be named like "ADA_features.csv", "BTC_features.csv", etc.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error  # old versions usually support this

# -------------------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------------------
ADVANCED_RESULTS_CSV = "data/advanced_tuning_results.csv"  # Tuning results
FEATURE_MAP_JSON      = "models/final_feature_columns.json"
FEATURES_FOLDER       = "data/"     # Where "ADA_features.csv" etc. are located
MODELS_FOLDER         = "models/"   # Where final .pkl will be saved
TEST_RATIO            = 0.20        # e.g. 80% train, 20% test

# Example final hyperparams for your models:
FINAL_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "random_state": 42,
    # etc. 
}
FINAL_XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    # etc.
}

# -------------------------------------------------------------------------
# HELPER: Manual RMSE (avoid squared=False param)
# -------------------------------------------------------------------------
def manual_rmse(y_true, y_pred):
    """
    Compute RMSE manually, to avoid scikit-learn version issues.
    """
    diff = y_true - y_pred
    return np.sqrt(np.mean(diff ** 2))

# -------------------------------------------------------------------------
# HELPER: Parse horizon from "Close_t+7" -> integer 7
# -------------------------------------------------------------------------
def parse_horizon_days(horizon_str):
    # e.g. "Close_t+30" -> 30
    return int(horizon_str.replace("Close_t+", "").strip())

# -------------------------------------------------------------------------
# HELPER: Prepare data, applying the correct feature columns
# -------------------------------------------------------------------------
def prepare_train_test(coin, horizon_col, feature_cols, test_ratio=TEST_RATIO):
    """
    1) Load e.g. data/ADA_features.csv if coin == 'ADA'
    2) Shift the 'Close' by parse_horizon_days(horizon_col) => 'Target'
    3) Drop rows with NaN
    4) Use ONLY feature_cols in the correct order
    5) Time-based train/test split => X_train, X_test, y_train, y_test
    """
    # 1. Load e.g. "ADA_features.csv"
    feat_csv = os.path.join(FEATURES_FOLDER, f"{coin}_features.csv")
    if not os.path.exists(feat_csv):
        raise FileNotFoundError(f"[Error] Missing CSV: {feat_csv}")

    df = pd.read_csv(feat_csv, parse_dates=["Date"])
    df.sort_values("Date", inplace=True, ignore_index=True)

    # 2. Create target
    horizon_days = parse_horizon_days(horizon_col)
    df["Target"] = df["Close"].shift(-horizon_days)

    # 3. Drop the final rows that have become NaN
    df.dropna(subset=["Target"], inplace=True)

    # 4. Only use the feature_cols + 'Target'
    #    Make sure all feature_cols exist:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[Error] Missing feature(s) for {coin}/{horizon_col}: {missing}")

    # Extract X,y
    X_all = df[feature_cols].values
    y_all = df["Target"].values

    # 5. Time-based split
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    X_train, X_test = X_all[:split_idx],  X_all[split_idx:]
    y_train, y_test = y_all[:split_idx],  y_all[split_idx:]
    return X_train, X_test, y_train, y_test

# -------------------------------------------------------------------------
# HELPER: Train final model (RF or XGB)
# -------------------------------------------------------------------------
def train_final_model(model_type, X_train, y_train):
    """
    model_type is either "RF" or "XGB".
    Returns the fitted model.
    """
    if model_type == "RF":
        model = RandomForestRegressor(**FINAL_RF_PARAMS)
    elif model_type == "XGB":
        model = XGBRegressor(**FINAL_XGB_PARAMS)
    else:
        raise ValueError(f"[Error] Unknown model_type={model_type}")

    model.fit(X_train, y_train)
    return model

# -------------------------------------------------------------------------
# MAIN: loop over advanced tuning results, finalize each coin/horizon
# -------------------------------------------------------------------------
def main():
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    # 1) Load advanced tuning results
    if not os.path.exists(ADVANCED_RESULTS_CSV):
        raise FileNotFoundError(f"[Error] Tuning results not found: {ADVANCED_RESULTS_CSV}")
    results_df = pd.read_csv(ADVANCED_RESULTS_CSV)
    print(f"[Info] Loaded tuning results: {results_df.shape} rows")

    # Expect columns like "Coin", "Horizon", "Best_Model", ...
    if "Best_Model" not in results_df.columns:
        # maybe it's "best_model"
        if "best_model" in results_df.columns:
            results_df["Best_Model"] = results_df["best_model"]
        else:
            raise ValueError("[Error] No 'Best_Model' column in CSV. Check naming.")

    # 2) Load final feature map JSON
    if not os.path.exists(FEATURE_MAP_JSON):
        raise FileNotFoundError(f"[Error] Missing {FEATURE_MAP_JSON}")
    with open(FEATURE_MAP_JSON, "r") as ff:
        feature_map = json.load(ff)

    # 3) Group by coin, horizon
    group_cols = ["Coin", "Horizon"]
    for col in group_cols:
        if col not in results_df.columns:
            raise ValueError(f"[Error] '{col}' column missing in {ADVANCED_RESULTS_CSV}")

    grouped = results_df.groupby(group_cols)
    for (coin, horizon_col), grp in grouped:
        row = grp.iloc[0]
        best_model_type = row["Best_Model"]  # e.g. 'RF' or 'XGB'

        # Feature map key: e.g. "ADA__Close_t+7" if your JSON uses "Coin__Horizon"
        # Adjust if your JSON keys differ. Let's assume it's <coin>__<horizon_col>:
        map_key = f"{coin}__{horizon_col}"
        if map_key not in feature_map:
            print(f"[Warning] No feature map for {map_key}, skipping.")
            continue
        feature_cols = feature_map[map_key]

        print(f"\n=== Finalizing {best_model_type} for {coin}/{horizon_col} ===")

        # Prepare train/test
        X_train, X_test, y_train, y_test = prepare_train_test(coin, horizon_col, feature_cols, TEST_RATIO)
        if len(X_train) < 10:
            print(f"    [Warning] Not enough training rows => skipping.")
            continue

        # Train final model
        model = train_final_model(best_model_type, X_train, y_train)

        # Evaluate quickly
        y_pred_test = model.predict(X_test)
        test_rmse = manual_rmse(y_test, y_pred_test)
        test_mae  = mean_absolute_error(y_test, y_pred_test)

        print(f"    [Info] #Test={len(X_test)}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

        # Save
        out_name = f"{coin}_{horizon_col}_{best_model_type}.pkl"
        out_name = out_name.replace("/", "_").replace("+", "plus_").replace(" ", "_")
        out_path = os.path.join(MODELS_FOLDER, out_name)
        joblib.dump(model, out_path)
        print(f"    [Saved] {out_path}")

    print("\n[Info] All final models have been trained & saved!\n")

if __name__ == "__main__":
    main()

