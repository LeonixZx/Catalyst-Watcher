import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
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
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
from bs4 import BeautifulSoup
import requests
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import math
import sys

# Constants
SEQUENCE_LENGTHS = [30, 60, 90]
SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT',
    '1818.KL', '6012.KL', '7113.KL', '5347.KL', '5183.KL', '7277.KL'
]

def load_models(model_types):
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        current_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        current_dir = os.path.dirname(os.path.abspath(__file__))

    models = {}
    model_filenames = [
        'Stock_Predictions_Model_lstm_30.keras',
        'Stock_Predictions_Model_cnn_lstm_30.keras',
        'Stock_Predictions_Model_rf_30.joblib',
        'Stock_Predictions_Model_xgb_30.joblib',
        'Stock_Predictions_Model_lstm_60.keras',
        'Stock_Predictions_Model_cnn_lstm_60.keras',
        'Stock_Predictions_Model_rf_60.joblib',
        'Stock_Predictions_Model_xgb_60.joblib',
        'Stock_Predictions_Model_lstm_90.keras',
        'Stock_Predictions_Model_cnn_lstm_90.keras',
        'Stock_Predictions_Model_rf_90.joblib',
        'Stock_Predictions_Model_xgb_90.joblib'
    ]
    for filename in model_filenames:
        model_path = os.path.join(current_dir, filename)
        if os.path.exists(model_path):
            model_type = filename.split('_')[-2]
            if (model_type == 'lstm' and 'LSTM' in model_types) or \
               (model_type == 'cnn' and 'CNN-LSTM' in model_types) or \
               (model_type == 'rf' and 'Random Forest' in model_types) or \
               (model_type == 'xgb' and 'XGBoost' in model_types):
                try:
                    if filename.endswith('.keras'):
                        models[filename] = load_model(model_path, compile=False)
                        models[filename].compile(optimizer='adam', loss='mean_squared_error')
                    elif filename.endswith('.joblib'):
                        models[filename] = joblib.load(model_path)
                    st.success(f"Successfully loaded model: {filename}")
                except Exception as e:
                    st.error(f"Error loading model {filename}: {str(e)}")
        else:
            st.warning(f"Model file not found: {filename}")
    return models


def fetch_stock_data_yfinance(symbol, start_date, end_date, max_retries=5):
    for _ in range(max_retries):
        try:
            data = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
            if not data.empty:
                # Check for missing dates
                all_dates = pd.date_range(start=start_date, end=end_date)
                business_days = pd.bdate_range(start=start_date, end=end_date)
                missing_business_days = business_days.difference(data.index)

                if not missing_business_days.empty:
                    st.warning(f"Missing data for {len(missing_business_days)} business days.")
                    st.info("This could be due to market holidays, trading suspensions, or data availability issues.")

                    # Attempt to fetch missing data
                    for missing_date in missing_business_days:
                        try:
                            missing_data = yf.download(symbol, start=missing_date, end=missing_date + timedelta(days=1), progress=False)
                            if not missing_data.empty:
                                data = pd.concat([data, missing_data])
                        except Exception as e:
                            st.warning(f"Could not fetch data for {missing_date}: {str(e)}")

                data = data.sort_index()
                st.success(f"Successfully fetched data for {symbol} from {data.index[0].date()} to {data.index[-1].date()}")
                st.info(f"Total trading days: {len(data)}, Expected business days: {len(business_days)}")
                return data
            else:
                st.warning(f"No data available for {symbol} between {start_date} and {end_date}. Retrying...")
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {str(e)}. Retrying...")

    st.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    return pd.DataFrame()

def fetch_stock_data_scraping(symbol, start_date, end_date):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/history"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'data-test': 'historical-prices'})
        if table:
            rows = table.find_all('tr')[1:]
            data = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    date = cols[0].text
                    close = cols[4].text.replace(',', '')
                    volume = cols[6].text.replace(',', '')
                    try:
                        data.append({
                            'Date': datetime.strptime(date, '%b %d, %Y'),
                            'Close': float(close),
                            'Volume': int(volume)
                        })
                    except ValueError:
                        continue

            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

            if not df.empty:
                st.success(f"Successfully fetched data for {symbol} up to {df.index[-1].date()} using web scraping")
                return df
    except Exception as e:
        st.error(f"Web scraping error for {symbol}: {str(e)}")
    return pd.DataFrame()

def get_stock_data(symbols, start_date, end_date, use_yfinance=True, max_retries=3, retry_delay=5):
    data = {}
    for symbol in symbols:
        for attempt in range(max_retries):
            try:
                if use_yfinance:
                    stock_data = yf.download(symbol, start=start_date, end=end_date, timeout=20)
                else:
                    # Use your alternative data source here
                    # For example:
                    # stock_data = alternative_data_source.get_data(symbol, start=start_date, end=end_date)
                    raise NotImplementedError("Alternative data source not implemented")

                if not stock_data.empty:
                    data[symbol] = stock_data
                    print(f"Successfully downloaded data for {symbol}")
                    break
                else:
                    print(f"No data available for {symbol}")
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error occurred for {symbol}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to download data for {symbol} after {max_retries} attempts: {str(e)}")
    return data

def clean_data(data):
    original_shape = data.shape

    # Remove columns that are all NaN
    data = data.dropna(axis=1, how='all')
    if data.shape[1] < original_shape[1]:
        st.info(f"Removed {original_shape[1] - data.shape[1]} columns that were all NaN.")

    # Replace infinity with NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    # Remove extreme values (beyond 5 standard deviations)
    for column in data.select_dtypes(include=[np.number]).columns:
        mean = data[column].mean()
        std = data[column].std()
        data[column] = data[column].mask(data[column].abs() > mean + 5*std)

    # Interpolate missing values
    data = data.interpolate(method='time', limit_direction='both', axis=0)

    # If any NaNs remain, forward fill then backward fill
    data = data.ffill().bfill()

    # Convert date column to datetime if it exists
    if 'Date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        if data['Date'].isna().any():
            st.warning("Some date conversions failed. These rows will be dropped.")
            data = data.dropna(subset=['Date'])

    # Ensure all numeric columns are float
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = data[column].astype(float)

    # Log the results
    rows_removed = original_shape[0] - data.shape[0]
    if rows_removed > 0:
        st.info(f"Removed {rows_removed} rows during the cleaning process.")

    st.info(f"Data cleaned. Shape before: {original_shape}, Shape after: {data.shape}")

    return data


def inspect_data(data_scaled, close_prices, features):
    st.subheader("Data Inspection")

    # Convert scaled data back to a DataFrame
    df = pd.DataFrame(data_scaled, columns=features)

    # Calculate the number of rows and columns for the subplot grid
    n_features = len(features)
    n_cols = 3  # You can adjust this number to change the number of columns in the grid
    n_rows = math.ceil(n_features / n_cols)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle("Distribution of Features", fontsize=16)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Plot histograms for each feature
    for i, feature in enumerate(features):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            sns.histplot(df[feature], ax=axes[i], kde=True)
            axes[i].set_title(feature)
            axes[i].set_xlabel('')

    # Remove any unused subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)

    # Display correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
    plt.title("Correlation Heatmap of Features")
    st.pyplot(fig)

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Time series plot of closing prices
    st.subheader("Closing Prices Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(close_prices.index, close_prices.values)
    ax.set_title("Closing Prices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

def calculate_feature_importance(model, X, y, features, n_repeats=10, model_type=''):
    if model_type in ['rf', 'xgb']:
        # Flatten the input for RF and XGB models
        X_flat = X.reshape(1, -1)
        n_features = len(features)
        n_time_steps = X_flat.shape[1] // n_features

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            result = permutation_importance(model, X_flat, y, n_repeats=n_repeats, random_state=42)
            importances = result.importances_mean

        # Aggregate importance for each feature across time steps
        feature_importance = np.zeros(n_features)
        for i in range(n_features):
            feature_importance[i] = np.sum(importances[i::n_features])

        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

    else:  # For LSTM and CNN-LSTM models
        X_orig = X.copy()

        baseline_pred = model.predict(X_orig, verbose=0).flatten()[0]

        importances = []
        for feature_idx in range(X_orig.shape[2]):
            feature_importance = []
            for _ in range(n_repeats):
                X_permuted = X_orig.copy()
                X_permuted[0, :, feature_idx] = np.random.permutation(X_permuted[0, :, feature_idx])
                permuted_pred = model.predict(X_permuted, verbose=0).flatten()[0]
                feature_importance.append(np.abs(baseline_pred - permuted_pred))
            importances.append(np.mean(feature_importance))

        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)

    # Filter out features with zero importance
    feature_importance = feature_importance[feature_importance['importance'] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(10, len(feature_importance))
    ax.bar(feature_importance['feature'][:top_n], feature_importance['importance'][:top_n])
    ax.set_title(f'Top {top_n} Feature Importances for {model_type.upper()} Model')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig, feature_importance



def add_technical_indicators(data):
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
    data['GDP_Growth'] = 2.0
    data['Inflation_Rate'] = 2.5
    return data

def add_sentiment_analysis(data, symbol, days=7):
    stock = yf.Ticker(symbol)
    news = stock.news[:days]
    sentiments = []
    titles = []
    for article in news:
        analysis = TextBlob(article['title'])
        sentiments.append(analysis.sentiment.polarity)
        titles.append(article['title'])
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    data['Sentiment'] = avg_sentiment
    return data, avg_sentiment, titles

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
        data_subset = data[features].copy()
        data_subset = data_subset.replace([np.inf, -np.inf], np.nan)
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

def ensemble_predict(models, data_scaled, sequence_lengths, features, recent_close_price, scaler):
    predictions = []
    for seq_len in sequence_lengths:
        X = np.array([data_scaled[-seq_len:]])
        X = X.reshape(-1, seq_len, len(features))

        for model_name, model in models.items():
            if f'_{seq_len}.' in model_name:
                try:
                    if model_name.endswith('.joblib'):
                        X_reshaped = X.reshape(X.shape[0], -1)
                        expected_features = model.n_features_in_
                        if X_reshaped.shape[1] != expected_features:
                            # Pad or truncate features to match the expected number
                            if X_reshaped.shape[1] < expected_features:
                                X_padded = np.pad(X_reshaped, ((0, 0), (0, expected_features - X_reshaped.shape[1])))
                                pred = model.predict(X_padded)
                            else:
                                pred = model.predict(X_reshaped[:, :expected_features])
                        else:
                            pred = model.predict(X_reshaped)
                    else:
                        # For Keras models
                        input_shape = model.input_shape
                        if input_shape is None:
                            st.warning(f"Model {model_name} input shape is None. Skipping this model.")
                            continue
                        if X.shape[1:] != input_shape[1:]:
                            # Pad or truncate features to match the expected shape
                            if X.shape[2] < input_shape[2]:
                                X_padded = np.pad(X, ((0, 0), (0, 0), (0, input_shape[2] - X.shape[2])))
                                pred = model.predict(X_padded)
                            else:
                                pred = model.predict(X[:, :, :input_shape[2]])
                        else:
                            pred = model.predict(X)
                    predictions.extend(pred.flatten())
                except Exception as e:
                    st.warning(f"Error predicting with model {model_name}: {str(e)}")

    if not predictions:
        st.error("No predictions could be made. Please check if the models are available and compatible with the current data.")
        return None

    avg_prediction = np.mean(predictions)
    unscaled_pred = scaler.inverse_transform([[avg_prediction] + [0]*(len(features)-1)])[0][0]

    return unscaled_pred

def display_model_date_range(data, prediction_date):
    st.subheader("Model Date Range Analysis")
    start_date = data.index[0].date()
    end_date = data.index[-1].date()
    num_trading_days = len(data)

    st.write(f"Data range: {start_date} to {end_date}")
    st.write(f"Number of trading days: {num_trading_days}")
    st.write(f"Prediction for: {prediction_date}")

def plot_price_vs_ma(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['SMA20'], label='SMA20')
    ax.plot(data.index, data['SMA50'], label='SMA50')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

def plot_rsi(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['RSI'])
    ax.axhline(y=70, color='r', linestyle='--')
    ax.axhline(y=30, color='g', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    return fig

def plot_bollinger_bands(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['BB_upper'], label='Upper BB')
    ax.plot(data.index, data['BB_lower'], label='Lower BB')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

def plot_macd(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['MACD'], label='MACD')
    ax.plot(data.index, data['MACD_signal'], label='Signal Line')
    ax.bar(data.index, data['MACD_diff'], label='MACD Histogram')
    ax.set_xlabel('Date')
    ax.set_ylabel('MACD')
    ax.legend()
    return fig


def interpret_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    st.header('Advanced Stock Market Predictor')

    st.sidebar.header('User Input Parameters')
    stock = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL, ^GSPC, ^KLSE)', 'AAPL')
    start_date = st.sidebar.date_input("Start date", date(2010, 1, 1))
    end_date = st.sidebar.date_input("End date", date.today())

    # Add a checkbox for selecting the data source
    use_yfinance = st.sidebar.checkbox("Use yfinance", value=True)

    # Model selection
    st.sidebar.header('Model Selection')
    all_model = st.sidebar.checkbox('All Models', value=True)
    if all_model:
        model_types = ['LSTM', 'CNN-LSTM', 'Random Forest', 'XGBoost']
    else:
        model_types = st.sidebar.multiselect(
            'Select Models',
            ['LSTM', 'CNN-LSTM', 'Random Forest', 'XGBoost'],
            default=['LSTM']
        )

    # Load models
    models = load_models(model_types)

    if st.sidebar.button('Predict'):
        # Show loading spinner
        with st.spinner('Fetching and processing data...'):
            data_dict = get_stock_data([stock], start_date, end_date, use_yfinance=use_yfinance)

        if not data_dict or stock not in data_dict:
            st.error(f"No data available for {stock} between {start_date} and {end_date}")
        else:
            data = data_dict[stock]
            st.subheader('Stock Data')
            st.write(data.tail())

            progress_bar = st.progress(0)
            progress_text = st.empty()

            progress_text.text("Adding technical indicators...")
            data = add_technical_indicators(data)
            progress_bar.progress(25)

            progress_text.text("Adding fundamental data...")
            data = add_fundamental_data(data, stock)
            progress_bar.progress(50)

            progress_text.text("Adding economic indicators...")
            data = add_economic_indicators(data)
            progress_bar.progress(75)

            progress_text.text("Performing sentiment analysis...")
            data, avg_sentiment, news_titles = add_sentiment_analysis(data, stock)
            progress_bar.progress(100)
            progress_text.empty()

            st.subheader('Sentiment Analysis')
            st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
            st.write(f"Interpretation: {interpret_sentiment(avg_sentiment)}")
            st.write("Recent News Titles:")
            for title in news_titles:
                st.write(f"- {title}")

            # After cleaning the data
            data_cleaned = clean_data(data)

            if len(data_cleaned) < len(data):
                st.warning(f"Removed {len(data) - len(data_cleaned)} rows with extreme values or NaN.")

            all_features = ['Open', 'High', 'Low', 'Close', 'Volume',
                            'SMA20', 'SMA50', 'EMA12', 'EMA26',
                            'RSI', 'BB_upper', 'BB_lower',
                            'MACD', 'MACD_signal', 'MACD_diff',
                            'Stoch_k', 'Stoch_d', 'OBV', 'CCI', 'ATR',
                            'Price_Change', 'Volume_Change',
                            'P/E', 'Dividend_Yield', 'Market_Cap', 'GDP_Growth', 'Inflation_Rate',
                            'Sentiment']

            # Add missing features with default values
            for feature in all_features:
                if feature not in data_cleaned.columns:
                    if feature in ['P/E', 'Dividend_Yield', 'Market_Cap']:
                        data_cleaned[feature] = 0
                    elif feature in ['GDP_Growth', 'Inflation_Rate']:
                        data_cleaned[feature] = data_cleaned['Close'].pct_change().rolling(window=252).mean().fillna(0)
                    elif feature == 'Sentiment':
                        data_cleaned[feature] = 0
                    st.warning(f"'{feature}' feature was missing and has been added with a default value.")

            # Check which features are actually available in the cleaned data
            available_features = [f for f in all_features if f in data_cleaned.columns]
            missing_features = set(all_features) - set(available_features)

            if missing_features:
                st.warning(f"The following features are still missing from the data and will not be used: {', '.join(missing_features)}")

            # Initialize the scaler here
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data_cleaned[available_features])

            recent_close_price = data_cleaned['Close'].iloc[-1]
            next_day_price = ensemble_predict(models, data_scaled, SEQUENCE_LENGTHS, available_features, recent_close_price, scaler)

            if next_day_price is not None:
                next_trading_day = data.index[-1] + pd.Timedelta(days=1)
                while next_trading_day.weekday() > 4:  # Skip weekends
                    next_trading_day += pd.Timedelta(days=1)

                display_model_date_range(data, next_trading_day)

                st.subheader(f'Predicted Price for Next Trading Day ({next_trading_day.date()}): ${next_day_price:.2f}')
                st.write(f"Recent closing price: ${recent_close_price:.2f}")
                predicted_change = next_day_price - recent_close_price
                st.write(f"Predicted change: ${predicted_change:.2f} ({(predicted_change / recent_close_price * 100):.2f}%)")
            else:
                st.error("Unable to make a prediction. Please check if the models are available and try again.")

            st.subheader('Data and Model Inspection')
            inspect_data(data_scaled, data_cleaned['Close'], available_features)

            st.subheader('Feature Importance')
            for seq_len in SEQUENCE_LENGTHS:
                st.write(f"### Sequence Length: {seq_len}")
                for model_name, model in models.items():
                    if f'_{seq_len}.' in model_name:
                        model_type = '_'.join(model_name.split('_')[3:-1])
                        st.write(f"{model_type.upper()} Model")

                        if len(data_scaled) >= seq_len:
                            X_imp = data_scaled[-seq_len:].reshape(1, seq_len, len(available_features))
                            y_imp = data_scaled[-1, available_features.index('Close')]

                            st.write(f"Input shape: {X_imp.shape}")
                            st.write(f"Model type: {type(model)}")

                            try:
                                fig, importance_df = calculate_feature_importance(model, X_imp, y_imp, available_features, model_type=model_type.lower())
                                st.pyplot(fig)
                                st.write(importance_df)
                            except Exception as e:
                                st.error(f"An error occurred while calculating feature importance for {model_type} model: {str(e)}")
                                st.error(f"Error details: {type(e).__name__}, {str(e)}")
                        else:
                            st.warning(f"Not enough data for sequence length {seq_len}. Skipping feature importance calculation.")

            # Visualizations
            st.subheader('Price vs Moving Averages')
            fig = plot_price_vs_ma(data_cleaned)
            st.pyplot(fig)

            st.subheader('RSI Indicator')
            fig = plot_rsi(data_cleaned)
            st.pyplot(fig)

            st.subheader('Bollinger Bands')
            fig = plot_bollinger_bands(data_cleaned)
            st.pyplot(fig)

            st.subheader('MACD')
            fig = plot_macd(data_cleaned)
            st.pyplot(fig)

    st.markdown("""
    ## How to use this app:
    1. Enter a stock symbol in the sidebar (e.g., AAPL for Apple, ^GSPC for S&P 500, ^KLSE for Malaysia KLSE).
    2. Select a start date and end date for historical data in the sidebar.
    3. Choose whether to use yfinance or web scraping for data retrieval.
    4. Choose whether to use all models or select specific models for prediction.
    5. Click the 'Predict' button to see the analysis and prediction.
    6. The app will show the predicted price for the next trading day, along with various charts and indicators.
    """)

    st.markdown("""
    **Disclaimer**: This app is for educational purposes only. The predictions are based on historical data and should not be used as financial advice. Always consult with a qualified financial advisor before making investment decisions.
    """)

if __name__ == "__main__":
    main()