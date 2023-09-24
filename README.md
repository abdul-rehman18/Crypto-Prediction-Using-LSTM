# Cryptocurrency Price Prediction with LSTM

This project uses Long Short-Term Memory (LSTM) neural networks to predict cryptocurrency prices. Specifically, it focuses on predicting the closing price of Bitcoin (BTC-USD) based on historical price and volume data.

## Getting Started

### Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python (>= 3.6)
- NumPy
- pandas
- Matplotlib
- mplfinance
- scikit-learn
- TensorFlow

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib mplfinance scikit-learn tensorflow
```

### Usage

1. Clone the repository:

```bash
git clone https://github.com/abdul-rehman18/Crypto-Prediction-Using-LSTM.git
cd cryptocurrency-price-prediction
```

2. Download the historical Bitcoin price data in CSV format and place it in the root directory of the project with the filename `BTC-USD.csv`.

3. Run the `crypto_price_prediction.ipynb` script:

```bash
python crypto_price_prediction.ipynb
```

This script loads the data, preprocesses it, trains an LSTM model, and makes predictions.

4. The script will display the Mean Squared Error (MSE) of the predictions, and a candlestick chart with the actual and predicted prices will be generated and displayed using Matplotlib.

## Model Architecture

The model architecture consists of two LSTM layers followed by two dense layers. The LSTM layers are used to capture sequential patterns in the data, and the dense layers perform regression to predict the closing price.

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
```

## Data Preprocessing

The historical price and volume data are preprocessed before feeding them into the model. Features are scaled using Min-Max scaling, and sequences of data are created for training, validation, and testing.

## Results

The model's performance is evaluated using Mean Squared Error (MSE) on the test dataset. Additionally, a candlestick chart is generated to visualize the actual and predicted prices.


**Disclaimer:** This project is for educational and research purposes only and should not be used for financial trading decisions. Cryptocurrency markets are highly volatile, and investing in cryptocurrencies involves risks.
```
