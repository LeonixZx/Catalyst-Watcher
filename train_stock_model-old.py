import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from textblob import TextBlob
from requests.exceptions import ReadTimeout
import time
from xgboost import XGBRegressor
import joblib


# Global variables
SEQUENCE_LENGTHS = [30, 60, 90]
SYMBOLS = [
    # US Stocks for global context
    'AAPL', 'GOOGL', 'MSFT',
    # Malaysian Stocks (Main Market)
    '1818.KL',  # Hartalega Holdings Berhad
    '6012.KL',  # Maxis Berhad
    '7113.KL',  # Top Glove Corporation Bhd
    '5347.KL',  # Tenaga Nasional Berhad (replacing KLSE:TENAGA)
    '5183.KL',  # Petronas Chemicals Group Berhad (replacing KLSE:PCHEM)
    '7277.KL'   # Dialog Group Berhad (replacing KLSE:DIALOG)
]

def get_stock_data(symbols, start_date, end_date, max_retries=3, retry_delay=5):
    data = {}
    for symbol in symbols:
        for attempt in range(max_retries):
            try:
                stock_data = yf.download(symbol, start=start_date, end=end_date, timeout=20)
                if not stock_data.empty:
                    data[symbol] = stock_data
                    print(f"Successfully downloaded data for {symbol}")
                    break
                else:
                    print(f"No data available for {symbol}")
                    break
            except ReadTimeout:
                if attempt < max_retries - 1:
                    print(f"Timeout occurred for {symbol}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to download data for {symbol} after {max_retries} attempts")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                break
    return data

def add_technical_indicators(data):
    """Add technical indicators to the dataset."""
    data['SMA20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    data['SMA50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
    data['EMA12'] = EMAIndicator(close=data['Close'], window=12).ema_indicator()
    data['EMA26'] = EMAIndicator(close=data['Close'], window=26).ema_indicator()
    data['RSI'] = RSIIndicator(close=data['Close']).rsi()
    bb = BollingerBands(close=data['Close'])
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_lower'] = bb.bollinger_lband()
    
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_diff'] = macd.macd_diff()
    
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stoch_k'] = stoch.stoch()
    data['Stoch_d'] = stoch.stoch_signal()
    
    data['OBV'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
    
    data['CCI'] = CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()
    
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
    
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    
    return data
def add_fundamental_data(data, symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        data['P/E'] = info.get('trailingPE', np.nan)
        data['Dividend_Yield'] = info.get('dividendYield', np.nan)
        data['Market_Cap'] = info.get('marketCap', np.nan)
    except:
        print(f"Error fetching fundamental data for {symbol}")
    return data

def add_economic_indicators(data):
    # Placeholder for economic indicators
    data['GDP_Growth'] = 2.0
    data['Inflation_Rate'] = 2.5
    return data

# Update the features list in the preprocess_data function
def add_sentiment_analysis(data, symbol, days=7):
    stock = yf.Ticker(symbol)
    news = stock.news[:days]
    sentiments = []
    for article in news:
        analysis = TextBlob(article['title'])
        sentiments.append(analysis.sentiment.polarity)
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    data['Sentiment'] = avg_sentiment
    return data

def preprocess_data(data_dict, sequence_lengths):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'SMA20', 'SMA50', 'EMA12', 'EMA26', 
                'RSI', 'BB_upper', 'BB_lower', 
                'MACD', 'MACD_signal', 'MACD_diff',
                'Stoch_k', 'Stoch_d', 'OBV', 'CCI', 'ATR',
                'Price_Change', 'Volume_Change', 
                'P/E', 'Dividend_Yield', 'Market_Cap', 'GDP_Growth', 'Inflation_Rate',
                'Sentiment']
    
    X, y = {seq_len: [] for seq_len in sequence_lengths}, {seq_len: [] for seq_len in sequence_lengths}
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    for symbol, data in data_dict.items():
        data = add_sentiment_analysis(data, symbol)  # Add this line
        data_subset = data[features].copy()
        
        # Replace infinity with NaN
        data_subset = data_subset.replace([np.inf, -np.inf], np.nan)
        
        # Remove extreme values (e.g., beyond 5 standard deviations)
        for column in data_subset.columns:
            mean = data_subset[column].mean()
            std = data_subset[column].std()
            data_subset[column] = data_subset[column].mask(data_subset[column].abs() > mean + 5*std)
        
        # Fill NaN values
        data_subset = data_subset.fillna(method='ffill').fillna(method='bfill')
        
        # If any NaN values remain, drop those rows
        data_subset = data_subset.dropna()
        
        if len(data_subset) > max(sequence_lengths):
            data_scaled = scaler.fit_transform(data_subset)
            
            for seq_len in sequence_lengths:
                for i in range(len(data_scaled) - seq_len):
                    X[seq_len].append(data_scaled[i:(i + seq_len)])
                    y[seq_len].append(data_scaled[i + seq_len, features.index('Close')])
        else:
            print(f"Insufficient data for {symbol} after cleaning. Skipping this symbol.")
    
    return {seq_len: np.array(X[seq_len]) for seq_len in sequence_lengths}, \
           {seq_len: np.array(y[seq_len]) for seq_len in sequence_lengths}, \
           scaler


def create_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=100),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_cnn_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(100, return_sequences=True),
        LSTM(100),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_random_forest():
    return RandomForestRegressor(n_estimators=100, random_state=42)

def create_xgboost():
    return XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

def train_sklearn_model(model, X, y):
    model.fit(X, y)
    return model

def train_model(model, X, y, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
        print(f"Fold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Add this function to your script
def add_sentiment_analysis(data, symbol, days=7):
    stock = yf.Ticker(symbol)
    news = stock.news[:days]
    sentiments = []
    for article in news:
        analysis = TextBlob(article['title'])
        sentiments.append(analysis.sentiment.polarity)
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    data['Sentiment'] = avg_sentiment
    return data

def main():
    start_date = '2010-01-01'
    end_date = '2024-09-03'

    data_dict = get_stock_data(SYMBOLS, start_date, end_date)

    for symbol in data_dict:
        data_dict[symbol] = add_technical_indicators(data_dict[symbol])
        data_dict[symbol] = add_fundamental_data(data_dict[symbol], symbol)
        data_dict[symbol] = add_economic_indicators(data_dict[symbol])
        # Note: We don't need to call add_sentiment_analysis here anymore

    X, y, scaler = preprocess_data(data_dict, SEQUENCE_LENGTHS)

    for seq_len in SEQUENCE_LENGTHS:
        if len(X[seq_len]) > 0:
            print(f"Training models for sequence length: {seq_len}")
            
            lstm_model = create_lstm_model((seq_len, X[seq_len].shape[2]))
            cnn_lstm_model = create_cnn_lstm_model((seq_len, X[seq_len].shape[2]))
            rf_model = create_random_forest()
            xgb_model = create_xgboost()
            
            train_model(lstm_model, X[seq_len], y[seq_len], epochs=100, batch_size=32)
            train_model(cnn_lstm_model, X[seq_len], y[seq_len], epochs=100, batch_size=32)
            rf_model = train_sklearn_model(rf_model, X[seq_len].reshape(-1, seq_len * X[seq_len].shape[2]), y[seq_len])
            xgb_model = train_sklearn_model(xgb_model, X[seq_len].reshape(-1, seq_len * X[seq_len].shape[2]), y[seq_len])
            
            lstm_model.save(f'Stock_Predictions_Model_lstm_{seq_len}.keras')
            cnn_lstm_model.save(f'Stock_Predictions_Model_cnn_lstm_{seq_len}.keras')
            joblib.dump(rf_model, f'Stock_Predictions_Model_rf_{seq_len}.joblib')
            joblib.dump(xgb_model, f'Stock_Predictions_Model_xgb_{seq_len}.joblib')
            
            print(f"Models for sequence length {seq_len} saved.")
        else:
            print(f"No data available for sequence length {seq_len} after preprocessing. Skipping this length.")

if __name__ == "__main__":
    main()