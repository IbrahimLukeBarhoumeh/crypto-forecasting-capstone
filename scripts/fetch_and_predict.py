#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fetch_and_predict.py
"""

import os
import sys
import json
import joblib
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--api_key", required=True)
    return parser.parse_args()

###############################################################################
# 1) MATCHING SYNONYMS
###############################################################################
COIN_SYNONYMS = {
    "ADA": "ADA", "Cardano": "ADA",
    "AVAX": "AVAX", "Avalanche": "AVAX",
    "BCH": "BCH", "BitcoinCash": "BCH",
    "BNB": "BNB", "BinanceCoin": "BNB",
    "BTC": "BTC", "Bitcoin": "BTC",
    "DOGE": "DOGE","Dogecoin": "DOGE",
    "DOT": "DOT","Polkadot": "DOT",
    "ETH": "ETH","Ethereum": "ETH",
    "LEO": "LEO","UnusSedLeo": "LEO",
    "LINK":"LINK","Chainlink":"LINK",
    "LTC":"LTC","Litecoin":"LTC",
    "MATIC":"MATIC","Polygon":"MATIC",
    "NEAR":"NEAR",
    "SHIB":"SHIB","Shiba":"SHIB","ShibaInu":"SHIB",
    "SOL":"SOL","Solana":"SOL",
    "TON":"TON","Toncoin":"TON",
    "TRX":"TRX","Tron":"TRX",
    "UNI":"UNI","Uniswap":"UNI",
    "XRP":"XRP","Ripple":"XRP",
}

COIN_TO_SYMBOL_FOR_CMC = {
    "ADA": "ADA","AVAX": "AVAX","BCH":"BCH","BNB":"BNB",
    "BTC":"BTC","DOGE":"DOGE","DOT":"DOT","ETH":"ETH",
    "LEO":"LEO","LINK":"LINK","LTC":"LTC","MATIC":"MATIC",
    "NEAR":"NEAR","SHIB":"SHIB","SOL":"SOL","TON":"TON",
    "TRX":"TRX","UNI":"UNI","XRP":"XRP"
}

def fetch_latest_data_cmc(cmc_symbol, api_key):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    params = {"symbol": cmc_symbol, "convert": "USD"}
    headers = {"Accepts":"application/json","X-CMC_PRO_API_KEY":api_key}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if cmc_symbol not in data["data"]:
        raise ValueError(f"[CMC Error] Symbol '{cmc_symbol}' not found.")
    coin_info = data["data"][cmc_symbol]
    quote_usd = coin_info["quote"]["USD"]
    price = float(quote_usd["price"])
    vol_24h = float(quote_usd["volume_24h"])
    row_df = pd.DataFrame({
        "Date": [datetime.utcnow().strftime("%Y-%m-%d")],
        "Open": [price],
        "High": [price],
        "Low": [price],
        "Close": [price],
        "Volume": [vol_24h],
    })
    return row_df

def compute_indicators(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values("Date", inplace=True, ignore_index=True)
    df["Daily_Returns_%"] = df["Close"].pct_change(fill_method="ffill") * 100
    df["MA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    std_20 = df["Close"].rolling(window=20, min_periods=1).std()
    df["BB_upper"] = df["MA_20"] + 2.0*std_20
    df["BB_lower"] = df["MA_20"] - 2.0*std_20

    delta = df["Close"].diff()
    gain = np.where(delta>0, delta,0.0)
    loss = np.where(delta<0, -delta,0.0)
    roll_gain = pd.Series(gain).rolling(14,min_periods=1).mean()
    roll_loss = pd.Series(loss).rolling(14,min_periods=1).mean()
    rs = roll_gain/(roll_loss+1e-9)
    df["RSI_14"] = 100.0 - (100.0/(1.0+rs))

    ema12 = df["Close"].ewm(span=12,adjust=False).mean()
    ema26 = df["Close"].ewm(span=26,adjust=False).mean()
    df["MACD"] = ema12-ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"]-df["MACD_Signal"]

    low14 = df["Low"].rolling(14,min_periods=1).min()
    high14= df["High"].rolling(14,min_periods=1).max()
    df["Stoch_%K"] = (df["Close"]-low14)/(1e-9+(high14-low14))*100
    df["Stoch_%D"] = df["Stoch_%K"].rolling(3,min_periods=1).mean()

    return df

def find_best_model_pkl(canonical_coin, horizon_str):
    csv_path = os.path.join("..","data","advanced_tuning_results.csv")
    if not os.path.exists(csv_path):
        return f"{canonical_coin}_Close_tplus_{horizon_str}_RF.pkl"
    df = pd.read_csv(csv_path)
    needed = {"Coin","Horizon","Best_Model"}
    if not needed.issubset(df.columns):
        return f"{canonical_coin}_Close_tplus_{horizon_str}_RF.pkl"

    def parse_int(h):
        if "Close_t+" in str(h):
            return int(h.replace("Close_t+",""))
        else:
            return int(h)

    cdf = df[df["Coin"]==canonical_coin]
    if cdf.empty:
        return f"{canonical_coin}_Close_tplus_{horizon_str}_RF.pkl"

    cdf["__hval"] = cdf["Horizon"].apply(parse_int)
    horizon_int = parse_int(horizon_str)
    row = cdf[cdf["__hval"]==horizon_int]
    if row.empty:
        return f"{canonical_coin}_Close_tplus_{horizon_str}_RF.pkl"

    best_model = row.iloc[0]["Best_Model"]
    return f"{canonical_coin}_Close_tplus_{horizon_str}_{best_model}.pkl".replace("+","plus_")

def main():
    args = parse_args()
    user_coin = args.coin.strip()
    user_horizon = args.horizon.strip()
    api_key = args.api_key.strip()

    # unify synonyms
    coin_upper = user_coin.upper()
    coin_title = user_coin.title()
    if coin_upper in COIN_SYNONYMS:
        canonical_coin = COIN_SYNONYMS[coin_upper]
    elif coin_title in COIN_SYNONYMS:
        canonical_coin = COIN_SYNONYMS[coin_title]
    else:
        raise ValueError(f"Unknown coin='{user_coin}'.")

    # unify horizon => 'Close_t+X'
    if user_horizon.isdigit():
        horizon_str = f"Close_t+{user_horizon}"
    elif "Close_t+" in user_horizon:
        horizon_str = user_horizon
    else:
        horizon_str = f"Close_t+{user_horizon}"

    print(f"\n=== Fetch & Predict coin='{canonical_coin}' horizon='{horizon_str}' ===")

    # map to CMC symbol
    if canonical_coin not in COIN_TO_SYMBOL_FOR_CMC:
        raise ValueError(f"Unknown coin='{canonical_coin}' not in COIN_TO_SYMBOL_FOR_CMC.")

    cmc_symbol = COIN_TO_SYMBOL_FOR_CMC[canonical_coin]
    latest_df = fetch_latest_data_cmc(cmc_symbol, api_key)
    print("[Info] Latest row from CMC:\n", latest_df, "\n")

    local_csv = os.path.join("..","data", f"{canonical_coin}_features.csv")
    if not os.path.exists(local_csv):
        sys.exit(f"[Error] local CSV not found: {local_csv}")

    hist_df = pd.read_csv(local_csv)
    combined_df = pd.concat([hist_df, latest_df], ignore_index=True)
    combined_df = compute_indicators(combined_df)
    final_row = combined_df.iloc[[-1]].copy()

    fm_path = os.path.join("..","models","final_feature_columns.json")
    if not os.path.exists(fm_path):
        sys.exit("[Error] 'final_feature_columns.json' missing in ../models/ folder.")

    with open(fm_path,"r") as fp:
        feat_map = json.load(fp)

    fm_key = f"{canonical_coin}__{horizon_str}"
    if fm_key not in feat_map:
        alt_key = f"{canonical_coin}__{user_horizon}"
        if alt_key in feat_map:
            fm_key = alt_key
        else:
            sys.exit(f"[Error] No feature columns for {fm_key} or {alt_key}")

    feature_cols = feat_map[fm_key]
    pkl_name = find_best_model_pkl(canonical_coin, user_horizon.replace("Close_t+",""))
    pkl_path = os.path.join("..","models", pkl_name)
    if not os.path.exists(pkl_path):
        alt_name = f"{canonical_coin}_{horizon_str}_RF.pkl".replace("+","plus_")
        alt_path = os.path.join("..","models",alt_name)
        if os.path.exists(alt_path):
            pkl_path = alt_path
        else:
            sys.exit(f"[Error] No model found: tried {pkl_path} and {alt_path}")

    print(f"[Info] Loading best model => {pkl_path}")
    model = joblib.load(pkl_path)

    final_input_df = final_row.reindex(columns=feature_cols, fill_value=0.0)
    X_input = final_input_df.values
    y_pred = model.predict(X_input)
    pred_price = float(y_pred[0])

    print("-----------------------------------------------------------")
    print(f"Predicted {horizon_str} price for '{canonical_coin}' = {pred_price:.4f} USD")
    best_model_type = "?"
    if "_XGB.pkl" in pkl_name:
        best_model_type="XGB"
    elif "_RF.pkl" in pkl_name:
        best_model_type="RF"
    print(f"(auto-selected best model: {best_model_type})")
    print("-----------------------------------------------------------")

if __name__=="__main__":
    main()
