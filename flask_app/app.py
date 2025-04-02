# file: flask_app/app.py

import os
import sys
import json
import subprocess
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

CMC_API_KEY = os.getenv("CMC_API_KEY", "")
if not CMC_API_KEY:
    print("[Warning] No CMC_API_KEY found in .env.")

###############################################################################
# 1) COMPLETE COIN_SYNONYMS
###############################################################################
COIN_SYNONYMS = {
    # ADA / Cardano
    "ADA": "ADA", "Cardano": "ADA",

    # AVAX / Avalanche
    "AVAX": "AVAX", "Avalanche": "AVAX",

    # BCH / Bitcoin Cash
    "BCH": "BCH", "BitcoinCash": "BCH",

    # BNB / Binance Coin
    "BNB": "BNB", "BinanceCoin": "BNB",

    # BTC / Bitcoin
    "BTC": "BTC", "Bitcoin": "BTC",

    # DOGE / Dogecoin
    "DOGE": "DOGE", "Dogecoin": "DOGE",

    # DOT / Polkadot
    "DOT": "DOT", "Polkadot": "DOT",

    # ETH / Ethereum
    "ETH": "ETH", "Ethereum": "ETH",

    # LEO / UNUS SED LEO
    "LEO": "LEO", "UnusSedLeo": "LEO",

    # LINK / Chainlink
    "LINK": "LINK", "Chainlink": "LINK",

    # LTC / Litecoin
    "LTC": "LTC", "Litecoin": "LTC",

    # MATIC / Polygon
    "MATIC": "MATIC", "Polygon": "MATIC",

    # NEAR
    "NEAR": "NEAR",

    # SHIB / Shiba Inu
    "SHIB": "SHIB", "Shiba": "SHIB", "ShibaInu": "SHIB",

    # SOL / Solana
    "SOL": "SOL", "Solana": "SOL",

    # TON / Toncoin
    "TON": "TON", "Toncoin": "TON",

    # TRX / Tron
    "TRX": "TRX", "Tron": "TRX",

    # UNI / Uniswap
    "UNI": "UNI", "Uniswap": "UNI",

    # XRP / Ripple
    "XRP": "XRP", "Ripple": "XRP",
}

@app.route("/")
def index():
    return "Flask API is running. POST to /api/predict or GET /api/market_data."

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        coin_input = data.get("coin", "").strip()
        horizon_input = data.get("horizon", "").strip()
        if not coin_input or not horizon_input:
            return jsonify({"error": "Missing fields coin/horizon"}), 400

        # 2) unify coin => synonyms
        coin_upper = coin_input.upper()
        coin_title = coin_input.title()
        if coin_upper in COIN_SYNONYMS:
            canonical_coin = COIN_SYNONYMS[coin_upper]
        elif coin_title in COIN_SYNONYMS:
            canonical_coin = COIN_SYNONYMS[coin_title]
        else:
            return jsonify({"error": f"Unknown coin or synonym: {coin_input}"}), 400

        
        env_copy = os.environ.copy()
        env_copy["CMC_API_KEY"] = CMC_API_KEY

        script_path = os.path.join("..", "scripts", "fetch_and_predict.py")
        cmd = [
            sys.executable,
            script_path,
            "--coin", canonical_coin,
            "--horizon", horizon_input,
            "--api_key", CMC_API_KEY
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, env=env_copy
        )

        
        predicted_str = ""
        for line in result.stdout.splitlines():
            if "Predicted" in line and "price for" in line and "=" in line:
                predicted_str = line.strip()
                break

        if not predicted_str:
            return jsonify({
                "status": "error",
                "error_msg": "No 'Predicted' line found in script output",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500

        right_side = predicted_str.split("=")[1].strip()
        price_str = right_side.split()[0]
        predicted_price = float(price_str)

        return jsonify({
            "status": "ok",
            "coin": canonical_coin,
            "horizon": horizon_input,
            "predicted_price": predicted_price,
        })

    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "error_msg": f"Subprocess returned error code {e.returncode}",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
    except Exception as ex:
        return jsonify({"status": "error", "error_msg": str(ex)}), 500


@app.route("/api/market_data", methods=["GET"])
def api_market_data():
    """
    Returns top N cryptos from CoinMarketCap, default 10 or ?limit=50
    """
    try:
        limit = request.args.get("limit", 10, type=int)
        if not CMC_API_KEY:
            return jsonify({"status":"error","msg":"No server-side API key configured"}), 500

        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        params = {
            "limit": limit,
            "convert": "USD"
        }
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": CMC_API_KEY
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return jsonify({
            "status": "ok",
            "coins": data.get("data", [])
        })

    except Exception as ex:
        return jsonify({"status":"error","msg":str(ex)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
