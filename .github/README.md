# Crypto-Forecasting-Capstone

**A comprehensive end-to-end data science project exploring multi-horizon forecasting for various cryptocurrencies.**  
This project integrates rigorous feature engineering, time-series modeling (ARIMA, Naive baseline), machine learning (RandomForest, XGBoost), hyperparameter tuning, and a React + Flask web interface for real-time prediction. Below you'll find an overview of the repository structure, workflow, methodologies, and key results.

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

The **Crypto Forecasting Capstone** aims to tackle the **multi-horizon price forecasting problem** (e.g., predicting `Close_t+1`, `Close_t+7`, `Close_t+30`, `Close_t+90`) for a curated set of cryptocurrencies. By leveraging both **traditional** time-series baselines and **advanced** machine learning models, we seek to:

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

- **CoinMarketCap API** was used to fetch the latest price snapshots.  
- **Historical CSV** files (`coin_<COIN>_features.csv`) were generated or curated for each cryptocurrency.  
- **Feature Engineering**: We manually added RSI(14), ATR(14), SMA(5,50), Bollinger(20) bands, MACD(12,26), Stochastic, etc.  
- **Time-Shifted Targets**: For each coin, we created multiple columns like `Close_t+1`, `Close_t+7`, `Close_t+30`, `Close_t+90` to facilitate multi-horizon forecasting.

**Note**: The final cleaned datasets (one CSV per coin) are stored in the `data/` folder.

---

## Modeling Approaches

### Baseline Models
1. **Naive**: “Tomorrow’s price = Today’s price.”
2. **ARIMA**: Using order `(p, d, q)` auto-selected or tuned with Statsmodels.  

**Why Baselines?**  
Baselines serve as an easy reference point to confirm that advanced ML methods actually yield improvements.

### Advanced Models
1. **RandomForestRegressor**  
2. **XGBRegressor**  

**Hyperparameter Tuning**: We applied **RandomizedSearchCV** over multiple parameters (n_estimators, max_depth, etc.) with 3-fold cross-validation. Our scripts automatically selected the best model (e.g., `RF` or `XGB`) for each coin + horizon combination, then persisted it as a `.pkl` in `models/`.

---

## Notebooks & Key Artifacts

### 01_data_preprocessing.ipynb
- Reads raw historical data, handles missing values, merges columns (Open, High, Low, Close, Volume).
- Basic exploratory analysis: identifying outliers, removing anomalies if any.

### 02_feature_engineering.ipynb
- Adds custom indicators (Moving Averages, RSI, MACD, Bollinger).  
- Illustrates rolling-based transformations and dropping NaNs from shifting windows.

### 03_baseline_modeling.ipynb
- Demonstrates Naive & ARIMA forecasting.  
- Exports baseline metrics (RMSE, MAE) as references for advanced modeling.

### 04_advanced_training_tuning.ipynb
- Trains `RandomForest` & `XGBoost` across multiple coins + horizons.  
- Logs the best hyperparameters, saves final results to `advanced_model_results.csv`.

### 05_evaluation.ipynb
- Consolidates predictions from both baseline and advanced models for final comparison.
- Potential plotting code: “Actual vs. Predicted” line charts, error histograms, etc.

### 06_final_comparison.ipynb
- Merges `baseline_results_statsmodels.csv` and `advanced_model_results.csv`.
- Summarizes best overall model per coin + horizon.
- **Bar charts** or **comparative plots** for final RMSE/MAE across coins.

### 07_advanced_tuning_no_talib.ipynb
- Alternative advanced script **without** TA-Lib, purely manual indicator calculation.
- Final hyperparameter tuning code (similar steps but possibly different libraries).

---

## Results & Highlights

**Performance** varied by coin & horizon. In general:
- **Naive** performed reasonably for short horizon (`t+1`) but quickly degraded for `t+30` or `t+90`.
- **ARIMA** often outperformed Naive on mid-range horizons but struggled with highly volatile altcoins.
- **RandomForest** and **XGB** typically offered improved RMSE vs. ARIMA for multi-horizon tasks, especially beyond `t+7`.

**Sample RMSE** (short snippet, see  `advanced_tuning_results.csv` for full details):
| Coin | Horizon     | Best Model | RMSE         | MAE          |
|------|------------|-----------:|-------------:|-------------:|
| ADA  | Close_t+1  | RF         | 0.026        | 0.0167       |
| BTC  | Close_t+7  | XGB        | 15794.76     | 10278.19     |
| ETH  | Close_t+30 | XGB        | 648.70       | 533.37       |
| DOGE | Close_t+90 | XGB        | 0.105        | 0.0585       |
| ...  | ...        | ...        | ...          | ...          |

**(These figures are purely illustrative and correspond to examples seen in earlier runs.)**

A typical advanced approach (e.g., XGB) yields 10–25% lower RMSE in mid/long horizons relative to ARIMA, validating the advantage of ensemble-based ML methods for crypto price forecasting.


## Project Structure
crypto-forecasting-capstone/
├── data/
│   ├── ADA_features.csv
│   ├── ...
│   └── advanced_tuning_results.csv
├── models/
│   ├── ADA_Close_tplus_7_XGB.pkl
│   ├── ...
│   └── final_feature_columns.json
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── ...
│   └── 07_advanced_tuning_no_talib.ipynb
├── scripts/
│   ├── fetch_and_predict.py
│   ├── finalize_and_persist_models.py
│   └── ...
├── flask_app/
│   └── app.py
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── PredictForm.jsx
│   │   ├── CMCMarketData.jsx
│   │   └── index.css
│   └── ...
├── .env            # contains CMC_API_KEY=...
├── requirements.txt
└── README.md
---

## Usage Instructions

1. **Model Training**  
   - Ensure dependencies: `pip install -r requirements.txt`  
   - Run the notebooks in `notebooks/` in numerical order, or use `scripts/finalize_and_persist_models.py` to finalize best models.

2. **Local Prediction**  
   - `python scripts/fetch_and_predict.py --coin "ADA" --horizon 7 --api_key "<YOUR_CMC_KEY>"`  
   - Watch the console for predicted results.

3. **Flask + React**  
   - In `flask_app/app.py`, run `python app.py` (or `flask run`)  
   - In `frontend/`, run `npm install` then `npm start`  
   - Visit `http://localhost:3000` to interact with the front-end. The predictions are served via `http://localhost:5000/api/predict`.

4. **Deployment**  
   - Host the Flask backend (e.g., Render, Heroku, PythonAnywhere)  
   - Build the React frontend (`npm run build`) and host the static files (GitHub Pages, Netlify, etc.).  
   - Adjust fetch URLs in the React code to point to your hosted Flask domain.

---

## Future Extensions

- **Additional Features**: Incorporate on-chain metrics, social sentiment, or macroeconomic indicators.
- **Extended Horizons**: Evaluate `t+180` or `t+365` for truly long-term predictions.
- **Performance Plots**: Add “actual vs. predicted” charts in notebooks for more intuitive demonstration.
- **CI/CD**: Integrate GitHub Actions to automatically test the notebooks and code on new commits.
- **Dockerization**: Provide a Dockerfile for simpler environment reproducibility and container-based deployment.

---

## License

This project is provided under an open-source license (e.g., MIT). See [LICENSE](LICENSE) for details.  
*Note: If not provided, please create a `LICENSE` file in the repo accordingly.*

---

## Contact

Feel free to open an issue or contact the repo owner if you have any questions or suggestions.  
Happy crypto forecasting!




# Crypto-Forecasting-Capstone

An end-to-end **multi-horizon forecasting** project that marries **time-series analysis** and **machine learning** to predict cryptocurrency prices at multiple future points (`t+1`, `t+7`, `t+30`, `t+90`). In addition to robust feature engineering and advanced model tuning, this project provides a **React + Flask** web app for real-time forecasting demos.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source & Preprocessing](#data-source--preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Modeling Approaches](#modeling-approaches)
   - [Baseline Models](#baseline-models)
   - [Advanced Models](#advanced-models)
5. [Key Results & Highlights](#key-results--highlights)
6. [Notebooks & Scripts](#notebooks--scripts)
7. [Deployment & Usage](#deployment--usage)
8. [Project Structure](#project-structure)
9. [Future Improvements](#future-improvements)
10. [References & License](#references--license)
11. [Contact](#contact)

---

## Project Overview

**Motivation**  
Cryptocurrencies are known for their high volatility and global 24/7 trading availability. Traditional time-series models can struggle with the abrupt price swings in this space, while advanced machine learning algorithms often yield improved predictive power—especially when combined with well-crafted technical indicators and robust hyperparameter tuning.

This project:
- **Compares** simple time-series baselines vs. **ensemble** ML algorithms for various horizon predictions.  
- **Demonstrates** advanced feature engineering (e.g., MACD, RSI, Bollinger Bands).  
- **Implements** an interactive **web UI** that pulls live snapshots via the CoinMarketCap API and runs inferences using pre-trained ML models.

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

=== Saved advanced multi-horizon results to: ../data\advanced_ratio_multihorizon_results_optimized.csv ===
    Coin     Horizon       RF_RMSE        RF_MAE      XGB_RMSE       XGB_MAE
0    ADA   Close_t+1      0.025445      0.016708      0.028275      0.018721
1    ADA   Close_t+7      0.070317      0.048611      0.076173      0.052353
2    ADA  Close_t+30      0.199159      0.127662      0.203981      0.125602
3    ADA  Close_t+90      0.324284      0.217756      0.316206      0.209011
4   AVAX   Close_t+1      2.337897      1.626471      2.633515      1.840575
5   AVAX   Close_t+7      6.405296      4.671014      6.516587      4.870371
6   AVAX  Close_t+30     13.767214     10.297995     13.147630      9.960141
7   AVAX  Close_t+90     25.241952     22.064108     26.568110     21.474070
8    BCH   Close_t+1     19.518295     12.755353     19.133087     12.330786
9    BCH   Close_t+7     49.220721     35.098500     49.544626     32.602816
10   BCH  Close_t+30    101.694522     73.490258     86.247121     60.785779
11   BCH  Close_t+90    262.559334    180.404877    243.643506    163.855384
12   BNB   Close_t+1     28.021811     17.768952     36.483627     23.102802
13   BNB   Close_t+7     96.993655     60.002880     65.989992     46.168318
14   BNB  Close_t+30    121.536073    100.739636    116.616001     98.340332
15   BNB  Close_t+90    199.940286    174.566718    198.963942    174.262128
16   BTC   Close_t+1   7386.967207   3532.142089   7557.049216   3700.352238
17   BTC   Close_t+7  11818.634941   8935.650945  10770.625642   7778.203816
18   BTC  Close_t+30  18585.162594  15604.170734  19051.825121  15744.287903
19   BTC  Close_t+90  31147.444943  28158.079955  31081.836708  28051.561308
20  DOGE   Close_t+1      0.017629      0.008519      0.016633      0.009730
21  DOGE   Close_t+7      0.041519      0.025003      0.035785      0.021859
22  DOGE  Close_t+30      0.080432      0.048503      0.076161      0.044252
23  DOGE  Close_t+90      0.106775      0.061322      0.106922      0.061793
24   DOT   Close_t+1      0.371437      0.257835      0.484433      0.286401
25   DOT   Close_t+7      1.643077      1.009809      1.493748      0.980484
26   DOT  Close_t+30      4.340223      2.248620      4.170168      2.269771
27   DOT  Close_t+90      9.316324      5.374049      8.662280      4.916107
28   ETH   Close_t+1    114.695488     74.892581    109.377821     78.308911
29   ETH   Close_t+7    267.933617    196.371497    298.829880    217.891946




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
