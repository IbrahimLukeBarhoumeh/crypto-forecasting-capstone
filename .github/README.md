# Crypto-Forecasting-Capstone
---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source & Preprocessing](#data-source--preprocessing)
3. [Modeling Approaches](#modeling-approaches)
   - [Baseline Models](#baseline-models)
   - [Advanced Models](#advanced-models)
4. [Notebooks & Key Artifacts](#notebooks--key-artifacts)
   - [01_...\_data\_preprocessing.ipynb](#01_data_preprocessingipynb)
   - [02_...\_feature_engineering.ipynb](#02_feature_engineeringipynb)
   - [03_...\_baseline_modeling.ipynb](#03_baseline_modelingipynb)
   - [04_...\_advanced_training_tuning.ipynb](#04_advanced_training_tuningipynb)
   - [05_...\_evaluation.ipynb](#05_evaluationipynb)
   - [06_...\_final_comparison.ipynb](#06_final_comparisonipynb)
   - [07_...\_advanced_tuning_no_talib.ipynb](#07_advanced_tuning_no_talibipynb)
5. [Results & Highlights](#results--highlights)
6. [Repository Structure](#repository-structure)
7. [Usage Instructions](#usage-instructions)
8. [Future Extensions](#future-extensions)
9. [License](#license)
10. [Contact](#contact)

---

## Project Overview

This project integrates rigorous feature engineering, time-series modeling (ARIMA, Naive baseline), machine learning (RandomForest, XGBoost), hyperparameter tuning, and a React + Flask web interface for real-time prediction. Below you'll find an overview of the repository structure, workflow, methodologies, and key results. An end-to-end **multi-horizon forecasting** project that marries **time-series analysis** and **machine learning** to predict cryptocurrency prices at multiple future points (`t+1`, `t+7`, `t+30`, `t+90`). In addition to robust feature engineering and advanced model tuning, this project provides a **React + Flask** web app for real-time forecasting demos.

- Compare performances across various horizons.  
- Illustrate best practices in data preprocessing, feature engineering, and hyperparameter tuning.  
- Provide a **deployment-ready** experience via a React + Flask web application, enabling real-time predictions for the end user.

**Key accomplishments** in this project include:
- Handling daily crypto price data from multiple coins (e.g., BTC, ETH, ADA, DOGE, etc.).
- Incorporating manual technical indicators (MA, RSI, MACD, Bollinger Bands, etc.).
- Performing advanced model selection (RandomForest, XGBoost) with cross-validation and hyperparameter tuning.
- Finalizing & persisting best models per coin + horizon for direct consumption.
- Creating a streamlined **front-end** (React) + **back-end** (Flask) that can be deployed for real user interaction.

---

## Data Source & Preprocessing

**Data Source**:  
1. **CoinMarketCap** for real-time price fetches (`quotes/latest`).  
2. **Historical CSVs** in `data/`, one per coin (`<COIN>_features.csv`), storing daily O/H/L/C, Volume, and engineered indicators.

**Preprocessing**:
- **Merging & Cleaning**: Combined raw price data into a uniform structure, ensuring date alignment and dropping any irregular entries.  
- **Shifting Targets**: For multi-horizon forecasting, columns like `Close_t+1`, `Close_t+7`, etc. were created by shifting the `Close` price accordingly. Rows that become invalid after shifting are dropped.  
- **Data Splits**: Most notebooks follow time-based splits (train/test) with ~80/20 partition to respect chronological order.

---

## Feature Engineering

**Manual Indicators**:
- **Moving Averages**: (SMA5, SMA50)  
- **Bollinger Bands**: (20-day rolling mean ± 2 std)  
- **RSI (14)**: Captures momentum.  
- **MACD (12,26)** + Signal line + MACD histogram.  
- **Stoch Oscillator (14,3)**  
- **Daily Returns**: Percentage returns from day-to-day.

These features are computed within the notebooks or in the final scripts, ensuring each coin’s CSV contains all required columns for training the ML models.

---

## Modeling Approaches

### Baseline Models
1. **Naive**  
   - Predicts: “Next day’s price = Today’s price.”  
   - Good yardstick for short horizons, but performance quickly diminishes for `t+7`, `t+30`, etc.

2. **ARIMA**  
   - Leveraged `statsmodels` for auto ARIMA or manual `(p, d, q)` setups.
   - Often outperforms Naive for stable timescales but can struggle with high crypto volatility.

### Advanced Models
1. **RandomForestRegressor**  
   - Well-suited for tabular feature inputs, stable performance on short/medium horizons.
   - We performed hyperparameter tuning (e.g., `n_estimators`, `max_depth`, `min_samples_split`).
2. **XGBRegressor**  
   - Gradient boosting approach with flexible loss function and strong performance in time-series tasks where structured features are available.
   - Tuning included `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.

---

## Key Results & Highlights

- **Baselines**  
  - Naive RMSE can be acceptable for `Close_t+1` but typically balloon at `Close_t+30` or `Close_t+90`.
  - ARIMA frequently does well at `t+7` if hyperparameters match market patterns.

- **Advanced ML**  
  - XGB or RF almost always edges out ARIMA on longer horizons. Gains can be ~10–30% improvement in RMSE vs. baselines.  
  - Tuning typically helps even more on altcoins with frequent volatility.

**Sample RMSE** (illustrative summary):
| Coin | Horizon     | Best Model | RMSE       | MAE        |
|------|------------|-----------:|-----------:|-----------:|
| **ADA** | Close_t+1  | RF         | 0.026      | 0.017      |
| **BTC** | Close_t+7  | XGB        | 15794.76   | 10278.19   |
| **ETH** | Close_t+30 | XGB        | 648.70     | 533.37     |
| **DOGE**| Close_t+90 | XGB        | 0.105      | 0.059      |

**Visual Example**  
Below is a sample side-by-side bar plot (from the notebooks) comparing RMSE for Naive vs. RF vs. XGB for ADA across different horizons:

| Coin | Horizon    | RF_RMSE   | RF_MAE   | XGB_RMSE  | XGB_MAE  |
|------|------------|----------:|---------:|----------:|---------:|
| ADA  | Close_t+1  | 0.0254    | 0.0167   | 0.0283    | 0.0187   |
| ADA  | Close_t+7  | 0.0703    | 0.0486   | 0.0762    | 0.0524   |
| ADA  | Close_t+30 | 0.1992    | 0.1277   | 0.2040    | 0.1256   |
| ADA  | Close_t+90 | 0.3243    | 0.2178   | 0.3162    | 0.2090   |
| AVAX | Close_t+1  | 2.3379    | 1.6265   | 2.6335    | 1.8406   |
| AVAX | Close_t+7  | 6.4053    | 4.6710   | 6.5166    | 4.8704   |
| AVAX | Close_t+30 |13.7672    |10.2980   |13.1476    | 9.9601   |
| AVAX | Close_t+90 |25.2420    |22.0641   |26.5681    |21.4741   |
| BCH  | Close_t+1  |19.5183    |12.7554   |19.1331    |12.3308   |
| BCH  | Close_t+7  |49.2207    |35.0985   |49.5446    |32.6028   |
| BCH  | Close_t+30 |101.6945   |73.4903   |86.2471    |60.7858   |
| BCH  | Close_t+90 |262.5593   |180.4049  |243.6435   |163.8554  |
| BNB  | Close_t+1  |28.0218    |17.7690   |36.4836    |23.1028   |
| BNB  | Close_t+7  |96.9937    |60.0029   |65.9900    |46.1683   |
| BNB  | Close_t+30 |121.5361   |100.7396  |116.6160   |98.3403   |
| BNB  | Close_t+90 |199.9403   |174.5667  |198.9639   |174.2621  |
| BTC  | Close_t+1  |7386.9672  |3532.1421 |7557.0492  |3700.3522 |
| BTC  | Close_t+7  |11818.6349 |8935.6509 |10770.6256 |7778.2038 |
| BTC  | Close_t+30 |18585.1626 |15604.1707|19051.8251 |15744.2879|
| BTC  | Close_t+90 |31147.4449 |28158.0800|31081.8367 |28051.5613|
| DOGE | Close_t+1  |0.0176     |0.0085    |0.0166     |0.0097    |
| DOGE | Close_t+7  |0.0415     |0.0250    |0.0358     |0.0219    |
| DOGE | Close_t+30 |0.0804     |0.0485    |0.0762     |0.0443    |
| DOGE | Close_t+90 |0.1068     |0.0613    |0.1069     |0.0618    |
| DOT  | Close_t+1  |0.3714     |0.2578    |0.4844     |0.2864    |
| DOT  | Close_t+7  |1.6431     |1.0098    |1.4937     |0.9805    |
| DOT  | Close_t+30 |4.3402     |2.2486    |4.1702     |2.2698    |
| DOT  | Close_t+90 |9.3163     |5.3740    |8.6623     |4.9161    |
| ETH  | Close_t+1  |114.6955   |74.8926   |109.3778   |78.3089   |
| ETH  | Close_t+7  |267.9336   |196.3715  |298.8299   |217.8919  |
| ...  | ...        | ...       | ...      | ...       | ...      |


---

## Notebooks & Scripts

1. **01_data_preprocessing.ipynb**  
   - Loads raw price data. Basic cleaning and merges.  
2. **02_feature_engineering.ipynb**  
   - Adds technical indicators, shifts `Close` for multi-horizon targets.  
3. **03_baseline_modeling.ipynb**  
   - Runs Naive vs. ARIMA. Logs RMSE/MAE.  
4. **04_advanced_training_tuning.ipynb**  
   - RandomForest/XGBoost with hyperparameter search.  
5. **05_evaluation.ipynb**  
   - Compares baseline vs. advanced predictions. Possibly includes “Actual vs. Predicted” plots.  
6. **06_final_comparison.ipynb**  
   - Summarizes best performance per coin + horizon.  
7. **07_advanced_tuning_no_talib.ipynb**  
   - Alternate advanced script (manual indicators only).  

**Key Scripts**  
- `scripts/finalize_and_persist_models.py`: Re-trains final models for each coin/horizon, saves `.pkl`.  
- `scripts/fetch_and_predict.py`: Accepts `--coin`, `--horizon`, `--api_key` and prints out predicted price.

---

## Deployment & Usage

1. **Local Notebook Execution**  
   - Clone the repo, install dependencies (`pip install -r requirements.txt`).
   - Run notebooks in numerical order to replicate the entire pipeline: from data ingestion through final model comparison.

2. **Model Inference via Command-Line**  
   - `python scripts/fetch_and_predict.py --coin ADA --horizon 7 --api_key <YOUR_CMC_KEY>`
   - Outputs RMSE, chosen model type, plus the predicted price in console.

3. **Flask + React Web App**  
   - **Backend**: 
     - `cd flask_app/`, ensure `.env` has `CMC_API_KEY=xxx`.  
     - `python app.py` (serves on `http://localhost:5000`).  
   - **Frontend**: 
     - `cd ../frontend/`, `npm install`, `npm start` (serves on `http://localhost:3000`).  
   - Visit `http://localhost:3000` => Interact with the UI, pick a coin/horizon => The app calls `http://localhost:5000/api/predict`.

---
## Visuals (Ex from traning, each coin has its own chart notebooks sections of code)
![image](https://github.com/user-attachments/assets/6f6a588a-02d3-4eae-bec6-a6e45c2c6cd2)

![image](https://github.com/user-attachments/assets/f07abaa9-d4bc-4b56-9208-7f736f9c6277)

![image](https://github.com/user-attachments/assets/226e32f9-f22c-4f85-9291-44b1d1206987)

![image](https://github.com/user-attachments/assets/15a9c67e-7cad-4c59-810b-16154e227ac3)


## Future Improvements

- **Additional Indicators**: On-chain metrics, social sentiment, macroeconomic data.  
- **Improved Baselines**: Add a Prophet or LSTM baseline for deeper time-series comparisons.  
- **More Visuals**: Actual vs. Predicted plot overlays, error distribution histograms, shapley-based feature importance.  
- **Dockerization**: Provide Dockerfiles for both the web app and notebooks to ease deployment.  
- **CI/CD Pipeline**: Automate tests for data loading, unit tests for scripts, and style checks.

---

## References & License

- **CoinMarketCap API**: [https://coinmarketcap.com/api/](https://coinmarketcap.com/api/)  
- **Scikit-Learn**, **XGBoost**, **Statsmodels**  
- Project is under an open-source MIT License (or whichever is chosen). See [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or suggestions, please open an issue or reach out to the repo owner:

> **Owner**: [Ibrahim Luke Barhoumeh]  
> **Email**: [lukebarhoumeh11@gmail.com]

Thank you for exploring this **Crypto-Forecasting-Capstone** project! Feel free to fork, experiment, and improve upon this foundation for your own forecasting needs.
