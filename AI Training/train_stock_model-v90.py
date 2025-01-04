import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from textblob import TextBlob
from requests.exceptions import ReadTimeout
from xgboost import XGBRegressor
import joblib
import json
import os
import sys


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AITrainingGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Stock Model Training")
        self.geometry("900x885")

        self.SEQUENCE_LENGTHS = [30, 60, 90]
        self.SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', '1818.KL', '6012.KL', '7113.KL', '5347.KL', '5183.KL', '7277.KL']

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        self.create_header()
        self.create_control_panel()
        self.create_settings_panel()
        self.create_info_panel()

        self.training_thread = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.message_queue = queue.Queue()

        self.train_loss_history = []
        self.val_loss_history = []


    def create_header(self):
        header_frame = ctk.CTkFrame(self.main_frame)
        header_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header_frame, text="AI Stock Model Training", font=("Roboto", 24, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")

        buttons_frame = ctk.CTkFrame(header_frame)
        buttons_frame.grid(row=0, column=1, padx=10, pady=10, sticky="e")

        ctk.CTkButton(buttons_frame, text="Save", command=self.save_to_json, width=80).grid(row=0, column=0, padx=5)
        ctk.CTkButton(buttons_frame, text="Load", command=self.load_from_json, width=80).grid(row=0, column=1, padx=5)
        ctk.CTkButton(buttons_frame, text="Reset", command=self.reset_to_default, width=80).grid(row=0, column=2, padx=5)


    def create_control_panel(self):
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        control_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.start_button = ctk.CTkButton(control_frame, text="Start", command=self.start_training, width=100)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.pause_button = ctk.CTkButton(control_frame, text="Pause", command=self.pause_training, width=100, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=5, pady=5)

        self.resume_button = ctk.CTkButton(control_frame, text="Resume", command=self.resume_training, width=100, state="disabled")
        self.resume_button.grid(row=0, column=2, padx=5, pady=5)

        self.stop_button = ctk.CTkButton(control_frame, text="Stop", command=self.stop_training, width=100, state="disabled")
        self.stop_button.grid(row=0, column=3, padx=5, pady=5)

        self.progress_bar = ctk.CTkProgressBar(control_frame)
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(control_frame, text="0.00%")
        self.progress_label.grid(row=1, column=3, padx=5, pady=5)

    def create_constants_editor(self):
        constants_frame = ctk.CTkFrame(self.main_frame)
        constants_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        constants_frame.grid_columnconfigure((0, 1), weight=1)

        # Sequence Lengths
        seq_frame = ctk.CTkFrame(constants_frame)
        seq_frame.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        ctk.CTkLabel(seq_frame, text="Sequence Lengths:").grid(row=0, column=0, columnspan=2)
        self.seq_length_listbox = tk.Listbox(seq_frame, height=3, width=10)
        self.seq_length_listbox.grid(row=1, column=0, columnspan=2, padx=2, pady=2)

        self.seq_length_entry = ctk.CTkEntry(seq_frame, width=50)
        self.seq_length_entry.grid(row=2, column=0, padx=2, pady=2)

        ctk.CTkButton(seq_frame, text="Add", command=self.add_sequence_length, width=50).grid(row=2, column=1, padx=2, pady=2)
        ctk.CTkButton(seq_frame, text="Remove", command=self.remove_sequence_length, width=100).grid(row=3, column=0, columnspan=2, padx=2, pady=2)

        # Stock Symbols
        sym_frame = ctk.CTkFrame(constants_frame)
        sym_frame.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        ctk.CTkLabel(sym_frame, text="Stock Symbols:").grid(row=0, column=0, columnspan=2)
        self.symbol_listbox = tk.Listbox(sym_frame, height=3, width=10)
        self.symbol_listbox.grid(row=1, column=0, columnspan=2, padx=2, pady=2)

        self.symbol_entry = ctk.CTkEntry(sym_frame, width=50)
        self.symbol_entry.grid(row=2, column=0, padx=2, pady=2)

        ctk.CTkButton(sym_frame, text="Add", command=self.add_symbol, width=50).grid(row=2, column=1, padx=2, pady=2)
        ctk.CTkButton(sym_frame, text="Remove", command=self.remove_symbol, width=100).grid(row=3, column=0, columnspan=2, padx=2, pady=2)


        # File Operations
        file_frame = ctk.CTkFrame(constants_frame)
        file_frame.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="ew")

        ctk.CTkButton(file_frame, text="Save", command=self.save_to_json, width=80).grid(row=0, column=0, padx=2, pady=2)
        ctk.CTkButton(file_frame, text="Load", command=self.load_from_json, width=80).grid(row=0, column=1, padx=2, pady=2)
        ctk.CTkButton(file_frame, text="Reset", command=self.reset_to_default, width=80).grid(row=0, column=2, padx=2, pady=2)

        self.update_seq_lengths_listbox()
        self.update_symbols_listbox()

    def create_settings_panel(self):
        settings_frame = ctk.CTkFrame(self.main_frame)
        settings_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        settings_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Sequence Lengths
        seq_frame = ctk.CTkFrame(settings_frame)
        seq_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        seq_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(seq_frame, text="Sequence Lengths", font=("Roboto", 14, "bold")).grid(row=0, column=0, pady=5)
        self.seq_length_listbox = tk.Listbox(seq_frame, height=5, width=15, bg="#2b2b2b", fg="white", selectbackground="#1f538d", font=("Roboto", 10), justify=tk.CENTER)
        self.seq_length_listbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        entry_frame = ctk.CTkFrame(seq_frame)
        entry_frame.grid(row=2, column=0, padx=5, pady=5)
        self.seq_length_entry = ctk.CTkEntry(entry_frame, width=70, font=("Roboto", 10))
        self.seq_length_entry.grid(row=0, column=0, padx=2, pady=2)
        ctk.CTkButton(entry_frame, text="Add", command=self.add_sequence_length, width=70, font=("Roboto", 12)).grid(row=0, column=1, padx=2, pady=2)

        ctk.CTkButton(seq_frame, text="Remove", command=self.remove_sequence_length, width=100, font=("Roboto", 12)).grid(row=3, column=0, padx=5, pady=5)

        # Stock Symbols
        sym_frame = ctk.CTkFrame(settings_frame)
        sym_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        sym_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sym_frame, text="Stock Symbols", font=("Roboto", 14, "bold")).grid(row=0, column=0, pady=5)
        self.symbol_listbox = tk.Listbox(sym_frame, height=5, width=15, bg="#2b2b2b", fg="white", selectbackground="#1f538d", font=("Roboto", 10), justify=tk.CENTER)
        self.symbol_listbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        entry_frame = ctk.CTkFrame(sym_frame)
        entry_frame.grid(row=2, column=0, padx=5, pady=5)
        self.symbol_entry = ctk.CTkEntry(entry_frame, width=70, font=("Roboto", 10))
        self.symbol_entry.grid(row=0, column=0, padx=2, pady=2)
        ctk.CTkButton(entry_frame, text="Add", command=self.add_symbol, width=70, font=("Roboto", 12)).grid(row=0, column=1, padx=2, pady=2)

        ctk.CTkButton(sym_frame, text="Remove", command=self.remove_symbol, width=100, font=("Roboto", 12)).grid(row=3, column=0, padx=5, pady=5)

        # Bulk Symbol Addition
        bulk_frame = ctk.CTkFrame(settings_frame)
        bulk_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        bulk_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(bulk_frame, text="Bulk Add Symbols", font=("Roboto", 14, "bold")).grid(row=0, column=0, padx=5, pady=5)
        self.bulk_symbol_entry = ctk.CTkTextbox(bulk_frame, height=100, width=200, font=("Roboto", 12))
        self.bulk_symbol_entry.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        ctk.CTkButton(bulk_frame, text="Bulk Add", command=self.bulk_add_symbols, width=100, font=("Roboto", 12)).grid(row=3, column=0, padx=5, pady=5)
        
        # Add description for bulk add
        ctk.CTkLabel(bulk_frame, text="Enter symbols separated by commas (eg.: TSLA, 4863.KL)", font=("Roboto", 10)).grid(row=2, column=0, padx=5, pady=(0, 5))
        
        
    def create_info_panel(self):
        info_frame = ctk.CTkFrame(self.main_frame)
        info_frame.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        info_frame.grid_columnconfigure((0, 1), weight=1)
        info_frame.grid_rowconfigure(1, weight=1)

        # Training Log
        log_frame = ctk.CTkFrame(info_frame)
        log_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="nsew")
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_frame, text="Training Log:", font=("Roboto", 14, "bold")).grid(row=0, column=0, pady=5, sticky="w")
        self.terminal = ctk.CTkTextbox(log_frame, height=150, font=("Courier", 12))
        self.terminal.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Graph
        graph_frame = ctk.CTkFrame(info_frame)
        graph_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")
        graph_frame.grid_rowconfigure(1, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(graph_frame, text="Training Progress:", font=("Roboto", 14, "bold")).grid(row=0, column=0, pady=5, sticky="w")
        self.fig, self.ax = plt.subplots(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")


    def bulk_add_symbols(self):
        bulk_symbols = self.bulk_symbol_entry.get("1.0", tk.END).strip()
        new_symbols = [symbol.strip().upper() for symbol in bulk_symbols.split(',') if symbol.strip()]
        self.SYMBOLS.extend([symbol for symbol in new_symbols if symbol not in self.SYMBOLS])
        self.update_symbols_listbox()
        self.bulk_symbol_entry.delete("1.0", tk.END)

    def save_to_json(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            data = {
                "SEQUENCE_LENGTHS": self.SEQUENCE_LENGTHS,
                "SYMBOLS": self.SYMBOLS
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Save Successful", f"Data saved to {file_path}")

    def load_from_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.SEQUENCE_LENGTHS = data.get("SEQUENCE_LENGTHS", self.SEQUENCE_LENGTHS)
            self.SYMBOLS = data.get("SYMBOLS", self.SYMBOLS)
            self.update_seq_lengths_listbox()
            self.update_symbols_listbox()
            messagebox.showinfo("Load Successful", f"Data loaded from {file_path}")

    def reset_to_default(self):
        self.SEQUENCE_LENGTHS = [30, 60, 90]
        self.SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', '1818.KL', '6012.KL', '7113.KL', '5347.KL', '5183.KL', '7277.KL']
        self.update_seq_lengths_listbox()
        self.update_symbols_listbox()
        messagebox.showinfo("Reset Successful", "Values reset to default")

    def add_sequence_length(self):
        length = self.seq_length_entry.get()
        if length.isdigit() and int(length) > 0:
            self.SEQUENCE_LENGTHS.append(int(length))
            self.update_seq_lengths_listbox()
            self.seq_length_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Invalid Input", "Please enter a positive integer.")


    def remove_sequence_length(self):
        selection = self.seq_length_listbox.curselection()
        if selection:
            index = selection[0]
            del self.SEQUENCE_LENGTHS[index]
            self.update_seq_lengths_listbox()

    def add_symbol(self):
        symbol = self.symbol_entry.get().strip().upper()
        if symbol and symbol not in self.SYMBOLS:
            self.SYMBOLS.append(symbol)
            self.update_symbols_listbox()
            self.symbol_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Invalid Input", "Please enter a unique, non-empty symbol.")

    def remove_symbol(self):
        selection = self.symbol_listbox.curselection()
        if selection:
            index = selection[0]
            del self.SYMBOLS[index]
            self.update_symbols_listbox()


    def update_seq_lengths_listbox(self):
        self.seq_length_listbox.delete(0, tk.END)
        for length in self.SEQUENCE_LENGTHS:
            self.seq_length_listbox.insert(tk.END, str(length))

    def update_symbols_listbox(self):
        self.symbol_listbox.delete(0, tk.END)
        for symbol in self.SYMBOLS:
            self.symbol_listbox.insert(tk.END, symbol)

    def save_constants(self):
        constants = {
            "SEQUENCE_LENGTHS": self.SEQUENCE_LENGTHS,
            "SYMBOLS": self.SYMBOLS
        }
        with open("constants.json", "w") as f:
            json.dump(constants, f)
        messagebox.showinfo("Constants Saved", "Constants have been saved to constants.json")

    def load_constants(self):
        if os.path.exists("constants.json"):
            with open("constants.json", "r") as f:
                constants = json.load(f)
            self.SEQUENCE_LENGTHS = constants.get("SEQUENCE_LENGTHS", self.SEQUENCE_LENGTHS)
            self.SYMBOLS = constants.get("SYMBOLS", self.SYMBOLS)
            self.update_seq_lengths_listbox()
            self.update_symbols_listbox()
        else:
            self.reset_constants()

    def reset_constants(self):
        self.SEQUENCE_LENGTHS = [30, 60, 90]
        self.SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', '1818.KL', '6012.KL', '7113.KL', '5347.KL', '5183.KL', '7277.KL']
        self.update_seq_lengths_listbox()
        self.update_symbols_listbox()

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showerror("Training in Progress", "Cannot modify constants while training is in progress.")
            return
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.start()
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.stop_button.configure(state="normal")
        self.process_queue()

    def pause_training(self):
        self.pause_event.set()
        self.pause_button.configure(state="disabled")
        self.resume_button.configure(state="normal")

    def resume_training(self):
        self.pause_event.clear()
        self.resume_button.configure(state="disabled")
        self.pause_button.configure(state="normal")

    def stop_training(self):
        self.stop_event.set()
        self.pause_event.clear()
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.resume_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")

    def run_training(self):
        try:
            logging.debug("Starting run_training method")
            self.update_terminal("Starting data collection...")
            data_dict = self.get_stock_data(self.SYMBOLS, '2010-01-01', '2024-09-03')
            self.update_progress(5)

            logging.debug("Preprocessing data...")
            self.update_terminal("Preprocessing data...")
            X, y, scaler = self.preprocess_data(data_dict)
            self.update_progress(10)

            total_steps = sum(len(X[seq_len]) > 0 for seq_len in self.SEQUENCE_LENGTHS) * 4
            current_step = 0

            for seq_len in self.SEQUENCE_LENGTHS:
                if len(X[seq_len]) > 0:
                    self.update_terminal(f"Training models for sequence length: {seq_len}")

                    models = [
                        ("LSTM", self.create_lstm_model((seq_len, X[seq_len].shape[2]))),
                        ("CNN-LSTM", self.create_cnn_lstm_model((seq_len, X[seq_len].shape[2]))),
                        ("Random Forest", self.create_random_forest()),
                        ("XGBoost", self.create_xgboost())
                    ]

                    for model_name, model in models:
                        if self.stop_event.is_set():
                            self.update_terminal("Training stopped.")
                            return

                        self.update_terminal(f"Training {model_name} model for sequence length {seq_len}...")
                        self.train_model(model, X[seq_len], y[seq_len], model_name, seq_len)

                        current_step += 1
                        progress = 10 + (current_step / total_steps) * 90
                        self.update_progress(progress)

                        self.update_terminal(f"{model_name} model for sequence length {seq_len} saved.")

                else:
                    self.update_terminal(f"No data available for sequence length {seq_len}. Skipping.")

            self.update_terminal("Training completed successfully!")
            self.update_progress(100)
        except Exception as e:
            logging.exception("An error occurred in run_training method")
            self.update_terminal(f"An error occurred: {str(e)}")
        finally:
            logging.debug("run_training method finished")
            self.message_queue.put(("training_finished", None))

    def get_stock_data(self, symbols, start_date, end_date, max_retries=3, retry_delay=5):
        logging.debug("Starting get_stock_data function")
        data = {}
        for symbol in symbols:
            logging.debug(f"Processing symbol: {symbol}")
            for attempt in range(max_retries):
                try:
                    logging.debug(f"Attempting to download data for {symbol} (Attempt {attempt + 1}/{max_retries})")
                    stock_data = yf.download(symbol, start=start_date, end=end_date, timeout=20)
                    logging.debug(f"Download attempt completed for {symbol}")
                    if not stock_data.empty:
                        data[symbol] = stock_data
                        logging.debug(f"Successfully downloaded data for {symbol}")
                        break
                    else:
                        logging.warning(f"No data available for {symbol}")
                        break
                except Exception as e:
                    logging.error(f"Error fetching data for {symbol}: {str(e)}")
                    logging.error(f"Error type: {type(e)}")
                    logging.error(f"Error args: {e.args}")
                    if attempt < max_retries - 1:
                        logging.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"Failed to download data for {symbol} after {max_retries} attempts")
        return data
        

    def add_technical_indicators(self, data):
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

    def add_fundamental_data(self, data, symbol):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            data['P/E'] = info.get('trailingPE', np.nan)
            data['Dividend_Yield'] = info.get('dividendYield', np.nan)
            data['Market_Cap'] = info.get('marketCap', np.nan)
        except Exception as e:
            self.update_terminal(f"Error fetching fundamental data for {symbol}: {str(e)}")
        return data

    def add_economic_indicators(self, data):
        # Placeholder for economic indicators
        data['GDP_Growth'] = 2.0
        data['Inflation_Rate'] = 2.5
        return data

    def add_sentiment_analysis(self, data, symbol, days=7):
        try:
            stock = yf.Ticker(symbol)
            news = stock.news[:days]
            sentiments = []
            for article in news:
                analysis = TextBlob(article['title'])
                sentiments.append(analysis.sentiment.polarity)
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            data['Sentiment'] = avg_sentiment
        except Exception as e:
            self.update_terminal(f"Error in sentiment analysis for {symbol}: {str(e)}")
            data['Sentiment'] = 0  # Default neutral sentiment
        return data

    def preprocess_data(self, data_dict):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA20', 'SMA50', 'EMA12', 'EMA26', 
                    'RSI', 'BB_upper', 'BB_lower', 
                    'MACD', 'MACD_signal', 'MACD_diff',
                    'Stoch_k', 'Stoch_d', 'OBV', 'CCI', 'ATR',
                    'Price_Change', 'Volume_Change', 
                    'P/E', 'Dividend_Yield', 'Market_Cap', 'GDP_Growth', 'Inflation_Rate',
                    'Sentiment']
        
        X, y = {seq_len: [] for seq_len in self.SEQUENCE_LENGTHS}, {seq_len: [] for seq_len in self.SEQUENCE_LENGTHS}
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        for symbol, data in data_dict.items():
            self.update_terminal(f"Preprocessing data for {symbol}...")
            try:
                data = self.add_technical_indicators(data)
                data = self.add_fundamental_data(data, symbol)
                data = self.add_economic_indicators(data)
                data = self.add_sentiment_analysis(data, symbol)
                
                data_subset = data[features].copy()
                
                # Replace infinity with NaN
                data_subset = data_subset.replace([np.inf, -np.inf], np.nan)
                
                # Remove extreme values (e.g., beyond 5 standard deviations)
                for column in data_subset.columns:
                    mean = data_subset[column].mean()
                    std = data_subset[column].std()
                    data_subset[column] = data_subset[column].mask(data_subset[column].abs() > mean + 5*std)
                
                # Fill NaN values
                data_subset = data_subset.ffill().bfill()
                
                # If any NaN values remain, drop those rows
                data_subset = data_subset.dropna()
                
                if len(data_subset) > max(self.SEQUENCE_LENGTHS):
                    data_scaled = scaler.fit_transform(data_subset)
                    
                    for seq_len in self.SEQUENCE_LENGTHS:
                        for i in range(len(data_scaled) - seq_len):
                            X[seq_len].append(data_scaled[i:(i + seq_len)])
                            y[seq_len].append(data_scaled[i + seq_len, features.index('Close')])
                    
                    self.update_terminal(f"Successfully preprocessed data for {symbol}")
                else:
                    self.update_terminal(f"Insufficient data for {symbol} after cleaning. Skipping this symbol.")
            except Exception as e:
                self.update_terminal(f"Error preprocessing data for {symbol}: {str(e)}")
        
        return {seq_len: np.array(X[seq_len]) for seq_len in self.SEQUENCE_LENGTHS}, \
               {seq_len: np.array(y[seq_len]) for seq_len in self.SEQUENCE_LENGTHS}, \
               scaler

    def create_lstm_model(self, input_shape):
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

    def create_cnn_lstm_model(self, input_shape):
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

    def create_random_forest(self):
        return RandomForestRegressor(n_estimators=100, random_state=42)

    def create_xgboost(self):
        return XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    def train_model(self, model, X, y, model_name, seq_len):
        if model_name in ['LSTM', 'CNN-LSTM']:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            tscv = TimeSeriesSplit(n_splits=3)
            
            for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
                self.update_terminal(f"Training {model_name} - Fold {fold}")
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                for epoch in range(100):  # Assuming 100 epochs max
                    if self.stop_event.is_set():
                        self.update_terminal(f"Training of {model_name} stopped.")
                        return
                    
                    if self.pause_event.is_set():
                        self.update_terminal(f"Training of {model_name} paused. Resume to continue.")
                        while self.pause_event.is_set() and not self.stop_event.is_set():
                            time.sleep(0.1)
                        if self.stop_event.is_set():
                            self.update_terminal(f"Training of {model_name} stopped after pause.")
                            return
                        self.update_terminal(f"Resuming training of {model_name}.")
                    
                    history = model.fit(X_train, y_train, epochs=1, batch_size=32, 
                                        validation_data=(X_val, y_val), callbacks=[early_stopping], 
                                        verbose=0)
                    
                    self.message_queue.put(("update_plot", (epoch, history.history['loss'][0], history.history['val_loss'][0])))
                    
                    if early_stopping.stopped_epoch:
                        break
            
            model.save(f'Stock_Predictions_Model_{model_name.lower()}_{seq_len}.keras')
        else:
            self.update_terminal(f"Training {model_name}")
            X_reshaped = X.reshape(X.shape[0], -1)
            model.fit(X_reshaped, y)
            joblib.dump(model, f'Stock_Predictions_Model_{model_name.lower()}_{seq_len}.joblib')

    def update_progress(self, percentage):
        self.message_queue.put(("update_progress", percentage))

    def update_terminal(self, message):
        self.message_queue.put(("update_terminal", message))

    def process_queue(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                if message[0] == "update_progress":
                    self.progress_bar["value"] = message[1]
                    self.progress_label.configure(text=f"{message[1]:.2f}%")
                elif message[0] == "update_terminal":
                    self.terminal.insert(tk.END, message[1] + "\n")
                    self.terminal.see(tk.END)
                elif message[0] == "update_plot":
                    self.update_plot(*message[1])
                elif message[0] == "training_finished":
                    self.start_button.configure(state="normal")
                    self.pause_button.configure(state="disabled")
                    self.resume_button.configure(state="disabled")
                    self.stop_button.configure(state="disabled")
                    return
        except queue.Empty:
            pass
        
        if self.training_thread and self.training_thread.is_alive():
            self.after(100, self.process_queue)
            
    def update_plot(self, epoch, train_loss, val_loss):
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        
        self.ax.clear()
        epochs = range(len(self.train_loss_history))
        self.ax.plot(epochs, self.train_loss_history, label='Train Loss')
        self.ax.plot(epochs, self.val_loss_history, label='Validation Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.ax.set_title('Training Progress')
        
        # Adjust the bottom margin to show full x-axis
        plt.tight_layout()
        self.fig.subplots_adjust(bottom=0.15)
        
        self.canvas.draw()

if __name__ == "__main__":
    try:
        logging.debug("Starting application")
        app = AITrainingGUI()
        app.mainloop()
    except Exception as e:
        logging.exception("An unhandled exception occurred")
        print(f"An error occurred: {str(e)}")