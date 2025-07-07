
# Polymarket and Stock Correlation

**One Sentence Summary**: This repository explores whether data from Polymarket prediction markets can help forecast short-term movements in traditional financial markets like NASDAQ, using time-series regression and classification models.

---

## Overview

**Definition of the Task / Challenge**  
The goal is to determine if there's a statistically meaningful relationship between sentiment expressed in Polymarket prediction markets and future movements in stock indices (e.g., NASDAQ). Specifically, we aim to use past market prediction data to forecast next-day or 3-day price direction or percentage change.

**Approach**  
We collect event market data from Polymarket and combine it with historical stock data from sources like `yfinance`. The problem is approached as both a regression and a binary classification task. Data is aligned and engineered into sequences, and we explore time-series feature extraction followed by machine learning models.

**Performance Summary**  
Initial experiments include linear models and decision trees. Results show modest predictive power. Further exploration with advanced models like LSTMs or transformers is proposed for future iterations.

---

## Summary of Work Done

- Ingested and cleaned Polymarket data and NASDAQ index data
- Engineered lagged features and calculated future price movements
- Merged datasets based on date index
- Performed exploratory analysis and visualizations
- Evaluated predictive performance using regression and classification metrics

---

## Data

**Type**  
- Prediction market event outcomes (e.g., YES/NO market prices over time)  
- Equity index price history (e.g., SPY, NASDAQ Composite)

**Input Format**  
- CSV: Polymarket event prices over time (YES/NO prices)  
- Time series from `yfinance`: OHLC and adjusted close prices

**Size**  
- ~90 days of data (daily resolution) from multiple markets  
- Merged dataset includes ~60–70 matched data points

**Preprocessing / Cleanup**  
- Converted dates to datetime index  
- Normalized event market prices  
- Computed returns and forward-looking deltas for prediction  
- Aligned time ranges between datasets

**Data Visualization**  
- Time series plots of market sentiment and index prices  
- Correlation matrix heatmaps  
- Scatter plots of sentiment vs. price change

---

## Problem Formulation

**Input / Output**  
- Input: Lagged Polymarket market prices, historical index prices  
- Output: Next-day or 3-day price change (numeric or binary direction)

**Models**  
- Linear Regression  
- Decision Tree Regressor  
- Binary classifiers for direction (Logistic Regression, Decision Tree)

**Loss, Optimizer, Hyperparameters**  
- Regression: MSE  
- Classification: Accuracy, ROC AUC  
- Basic sklearn defaults used initially

---

## Training

**How**:  
Trained using Scikit-learn on CPU in Google Colab.  
Training was fast due to small dataset.

**Training Time**:  
Under 1 minute for each model

**Stopping Criteria**:  
No explicit early stopping; experiments run manually for now.

**Difficulties**:  
- Small data size  
- Alignment between datasets  
- Limited variability in Polymarket data

---

## Performance Comparison

**Metrics**  
- Regression: RMSE  
- Classification: Accuracy, ROC AUC

| Model               | Task         | Metric       | Value     |
|--------------------|--------------|--------------|-----------|
| Linear Regression   | Regression   | RMSE         | ~0.015    |
| Decision Tree       | Regression   | RMSE         | ~0.012    |
| Logistic Regression | Classification | Accuracy  | ~61%      |

**Visualizations**  
- ROC curve for classifier  
- Predicted vs. actual price change scatter plots

---

## Conclusions

- Polymarket data shows some predictive signal, but it's weak on its own.
- Combination of sentiment + price trends shows better potential.
- More robust results may require feature engineering or deep learning.

---

## Future Work

- Apply LSTM or transformer-based sequence models  
- Collect more data over longer periods  
- Explore multi-variable event markets (not just binary YES/NO)
- Automate sentiment extraction from event text

---

## How to Reproduce Results

### Reproducing Analysis
1. Open the notebook: `PolymarketProject.ipynb`
2. Run all cells in Google Colab or locally (see below for setup)
3. Data is fetched from `yfinance` and local CSVs, which may need to be replaced

### Applying to Other Data
- Replace Polymarket input with new CSVs
- Modify preprocessing section to adjust date ranges

---

## Overview of Files in Repository

- `PolymarketProject.ipynb`: Main notebook with data ingestion, processing, visualization, and modeling
- `data/`: Folder for any local CSV files (if used)
- `output/`: Directory for saving model results or figures (optional)

---

## Software Setup

### Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance

### Install
```bash
pip install -r requirements.txt
```

_or install manually_:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yfinance
```

---

## Data

- Polymarket data: Exported manually from [polymarket.com](https://polymarket.com)  
- Stock data: Automatically fetched via `yfinance`

---

## Training

Training is executed in-place within the notebook. Models are trained using `sklearn` without GPUs or TPUs.

---

## Performance Evaluation

Run the final cells in the notebook to:
- Generate prediction plots
- Evaluate metrics like RMSE, accuracy, and ROC

---

## Citations

- [Polymarket](https://polymarket.com)  
- [Yahoo Finance via yfinance](https://pypi.org/project/yfinance/)  
- Scikit-learn Documentation: https://scikit-learn.org/
