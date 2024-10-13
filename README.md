# AlgoTrading Project

A Python-based algorithmic trading project leveraging **pandas** for data manipulation and **CatBoost** for machine learning model training. This project is designed to develop, test, and optimize trading strategies based on historical data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
  
        - [0. Get Trading Data](#0-get-trading-data)
	- [1. Data Fix and Sanity Check](#1-data-fix-and-sanity-check)
	- [2. Statistic Check](#2-statistic-check)
	- [3. CatBoost Preparation and Training](#3-catboost-preparation-and-training)
		- [3.1 Example Manufacturing](#31-example-manufacturing)
		- [3.2 Model Training](#32-model-training)
	- [4. Backtest and Optimization](#4-backtest-and-optimization)
- [Installation](#installation)
- [Usage](#usage)
        - [1. Data Fix](#1-run-part-1---data-fix)
	- [2. Statistic Check](#2-run-part-2---Statistic-Check)
	- [3. CatBoost Preparation and Training](#3-run-part-3---catboost)
	- [4. Backtest and Optimization](#4-run-part-4---backtest-and-optimization)

## Overview

### General Definition
Trading gap percentage - the change in price between previous day 'close' price and today's 'open' price.

### Project Definition
Trading gap percentage -
1. Gap Up - today's 'open' price is **higher** than last day **'high'** price -> gap percentage =
   $\frac{open_i - high_{i-1}}{high_{i-1}}$
2. Gap Down - today's 'open' price is **lower** than last day **'low'** price -> gap percentage =
   $\frac{low_{i-1} - open_i}{low_{i-1}}$

### Project Goal
This project aims to create a robust algorithmic trading system that can identify day trading gaps that are bigger than 2% and predict if the next direction of the price will be 'Long' or 'Short'.

Gap Up:

Gap&Go Strategy is in place if the model signals to enter 'Long' position.

![image](https://github.com/user-attachments/assets/6f300fd6-2c9f-406f-9978-b153de9983d9)

Close Gap Strategy is in place if the model signals to enter 'Short' position.

![image](https://github.com/user-attachments/assets/ca139355-f816-4c94-8e72-67a3836dec64)

Gap Down:

Gap&Go Strategy is in place if the model signals to enter 'Short' position.

![image](https://github.com/user-attachments/assets/e4fe09aa-c7af-4af9-970b-af358ef35b5a)

Close Gap Strategy is in place if the model signals to enter 'Long' position.

![image](https://github.com/user-attachments/assets/c4382cd5-a37f-4745-b431-deb8d5c5aec7)



## Project Structure

### 0. Get Trading Data 
- **Sources:** WRDS crsp and yfinance (main source will be WRDS).
-  **Tasks:**
    - Access wrds server , request and download all s&p stocks trading data.
    - Access yfinance api , request and download all s&p stocks trading data.


### 1. Data Fix and Sanity Check

- **Objective:** Clean and preprocess the raw trading data to ensure quality and consistency.
- **Project's data source** - s&p stocks from wrds crsp - 1day interval 'ohlcv' data.
- **Tasks:**
	- Handle missing or null values.
	- Correct data types and formats.
	- Remove or rectify outliers and anomalies.
	- Ensure chronological order and integrity of the dataset.

### 2. Statistic Check

- **Objective:** Analyze the trading patterns to assess their potential profitability.
- **Tasks:**
	- Perform exploratory data analysis (EDA) to understand data distributions.
	- Calculate parameters:

| trade_pct | total gaps | gaps not closed | gaps closed ratio | gaps not closed ratio | Avg days till gap close | Avg drawdown of closed gaps | dd < 2% | 2% <= dd < 5% | 5% <= dd < 7% | 7% <= dd < 10% | 10% <= dd < 15% | 15% <= dd < 20% | 20% <= dd < 30% | dd >= 30% | Max drawdown | -- Avg drawdown - gaps not closed -- | Max drawdown - gaps not closed |
|-----------|------|----------------|--------------------|------------------------|--------------------------|------------------------------|---------|----------------|----------------|------------------|------------------|------------------|------------------|-----------|---------------|------------------------------------|------------------------------|
| 0-2%     |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |
| 2-5%     |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |
| 5-7%     |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |
| 7-10%    |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |
| 10-15%   |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |
| 15-20%   |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |
| 20-30%   |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |
| 30-100%  |      |                |                    |                        |                          |                              |         |                |                |                  |                  |                  |                  |           |               |                                    |                              |

### 3. CatBoost Preparation and Training

#### 3.1 Example Manufacturing

- **Objective:** Generate training examples that the CatBoost model can learn from.
- **Tasks:**
	- Feature engineering to create meaningful input variables.
	- Labeling data based on trading outcomes.


#### 3.2 Model Training

- **Objective:** Train the CatBoost model to predict trading price direction.
- **Tasks:**
	- Splitting data into training(2012-2016) and validation sets(2017-2019).
	- Initialize and configure the CatBoost classifier.
	- Train the model using the prepared dataset.
	- Perform hyperparameter tuning to optimize performance.
	- Validate the model's accuracy and reliability on unseen data(=validation).

### 4. Backtest and Optimization

- **Objective:** Evaluate the trading strategy's performance using historical data and optimize it for better results.

- **Tasks:**
	- Decide which stocks are going to be optimized out of the S&P stocks. We chose the best 20 profitable stocks that behaved similarly.
	- Simulate trades based on model predictions to assess profitability.
	- Calculate key performance metrics.

| Optimization ID | Total Return | Annualized Return | Annualized Sharpe | Sortino Ratio | Max Drawdown | Calmar Ratio |
|-----------------|--------------|-------------------|-------------------|----------------|--------------|---------------|
|                 |              |                   |                   |                |              |               |


## Installation

### 1. **python and libraries installation**
- Download the latest version of Python for Windows from python official website.
- run the installer - !Make sure to check the option "Add Python to PATH" during the installation process!.
- run the next commands:
    ```bash
    python -m pip install --upgrade pip
    pip install pandas numpy catboost scikit-learn matplotlib optuna joblib plotly psutil
  ```

### 2. **Clone the Repository**

```bash
git clone https://github.com/yourusername/AlgoTrading.git
cd AlgoTrading
```

### 3. **Get Trading Data**
#### WRDS
- Get access to WRDS crsp data.
- Search for jupiter notebook server in wrds.
- Upload files from directory 0 and run it.
- Download the zip containing all the S&P stocks to `1- data\wrds 1day Data\original`.
#### yfinance
- Enter `1- data\yfinance 1day Data`.
- Run the `.ipynb` file.

## Usage

### 1. **Run Part 1 - Data Fix**
Get to main directory and run:

```bash
    cd '1- data\wrds 1day Data'
    python data_fix.py
    cd ../..
```

### 2. **Run Part 2 - Statistic Check**

```bash
    cd '2- statistic check'
    python advanced_gap_analysis.py
    python gap_info_analysis.py
    cd ..
```

### 3. **Run Part 3 - CatBoost**

* Enter 'catboost_year_training.py' and define the boolean variables on lines 337-343.

* Main reason - define if to run optimization or training.

* When completed run:
    ```bash
    cd '3- catBoost\examples producing'
    python catboost_gap_analysis.py
    cd 'catBoost Data'
    python gap_per_stock.py
    xcopy "backtest_stocks.xlsx" "../../../4- optimization" /Y
    cd ../..
    python catboost_year_training.py
    cd ..
    ```

### 4. **Run Part 4 - Backtest and Optimization**
In our project - the in-sample period was `2012-2019` and the out-sample period was `2020-2023`.

## - IN-SAMPLE:

- First Optimization:
	1. Enter the symbols you want to optimize inside 'backtest_stocks.xlsx'
	   The Default we chose to use is the top 100 stocks which has max gaps from 2012 to 2019.
	   (the 'backtest_stocks.xlsx' file already contains those stocks symbols).

	2. Enter 'input.xlsx' and define the parameters values.
	   The Default should be

    | Parameter              | Start | End | Step |
    |------------------------|-------|-----|------|
    | z_score_window          | 14    | 14  | 1    |
    | z_score_threshold       | 0     | 0   | 1    |
    | atr_window              | 14    | 14  | 1    |
    | gap&go_sl_tp_ratio      | 1.0   | 1.0 | 1.0  |
    | close_gap_sl_tp_ratio   | 1.0   | 1.0 | 1.0  |


	3. Enter 'main.py' and make sure that 'IN_SAMPLE' variable in line 15 is `True`.
	4. Run:
    ```bash
    cd "4- optimization"
    python main.py
    ```
	5. After Run has finished -> open 'scores_per_stock_1.xlsx' file in 'optimization results' folder.  From the table choose the stocks symbols that you want to continue the optimization with.

- Regular Optimization:

	1. Each Run of 'main.py' creates the files inside 'optimization results' folder -
		* 'logs.txt' - logs of last runs.
		* 'scores_df.xlsx' - table of all combinations results sorted by 'total return' column from high to low.
		  from this table you can choose the combination id and parameters that fits you the most.
		* 'portfolio_value_{opt_id_num}.xlsx' - excel to verify that the sum of the whole portfolio is valid.
		* 'scores_per_stock_{opt_id_num}.xlsx' - excel to examine the results of a parameter combination id on each stock individually. (if one stock was much better than the others - consider run the same parameters without this stock).

  From the tables - decide how to specify the parameters values.

## - Out-SAMPLE:
* After the In-Sample Optimization resulting a set of parameters that suits you - its time to check the results on a different period of time.

* Pay attention that in Out-Sample Opt there is *no parameter tuning* - this will lead to *over-fitting*.

1. Enter 'main.py' and make sure that 'IN_SAMPLE' variable in line 15 is `False`.
2. Run:
```bash
    python main.py
```
3. check results.
