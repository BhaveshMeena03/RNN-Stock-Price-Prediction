# Stock Price Prediction Using RNNs

Sequence models that predict stock prices from historical data across four technology companies — **Amazon (AMZN), Google (GOOGL), IBM (IBM), and Microsoft (MSFT)**.

## Objective

Given the stock prices of the four companies over a fixed window of days, predict their prices for the following period. All four belong to the same sector (Technology), so training on them jointly lets the model capture broader market sentiment rather than the behaviour of a single stock.

## Approach

- Load and align multi-stock OHLC time-series data
- Normalise and window the series into supervised sequences
- Build and train recurrent models (RNN / LSTM / GRU) on the windowed data
- Evaluate forecasts against held-out prices and visualise predicted vs. actual

## Tech

Python, TensorFlow / Keras, pandas, NumPy, Matplotlib (Jupyter Notebook)

## Data

`AMZN_stocks_data.csv`, `GOOGL_stocks_data.csv`, `IBM_stocks_data.csv`, `MSFT_stocks_data.csv` (also under `RNN_Stocks_Data/`).

## Run it

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn jupyter
jupyter notebook RNN_Assg_Stock_Price_Prediction_Starter.ipynb
```
