import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def new_func():
    return 1
    
def fetch_data(tickers, start_date, end_date):
    """
    Fetch stock data for multiple tickers and return a dictionary of DataFrames.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = df
    return data

def normalize_data(df, method='first_value', columns=None):
    """
    Normalize stock data using different methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing stock market data
    method : str, optional (default='first_value')
        Normalization method:
        - 'first_value': Normalize by first value (percentage change)
        - 'minmax': Min-max scaling to [0, 1]
        - 'zscore': Z-score normalization
        - 'percentage': Percentage change from previous value
    columns : list, optional
        List of columns to normalize. If None, uses all numeric columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized columns added with '_norm' suffix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for column in columns:
        if method == 'first_value':
            # Normalize by first value (like reference code)
            df_normalized[f'{column}_norm'] = df[column].div(df[column].iloc[0]).mul(100)
            
        elif method == 'minmax':
            # Min-max normalization
            min_val = df[column].min()
            max_val = df[column].max()
            df_normalized[f'{column}_norm'] = (df[column] - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            # Z-score normalization
            mean = df[column].mean()
            std = df[column].std()
            df_normalized[f'{column}_norm'] = (df[column] - mean) / std
            
        elif method == 'percentage':
            # Percentage change
            df_normalized[f'{column}_norm'] = df[column].pct_change() * 100
    
    return df_normalized

def plot_normalized_prices(data_dict, column='Close', method='first_value'):
    """
    Plot normalized prices for multiple stocks.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames containing stock data
    column : str, optional (default='Close')
        Column to normalize and plot
    method : str, optional (default='first_value')
        Normalization method to use
    """
    plt.figure(figsize=(15, 7))
    
    for ticker, df in data_dict.items():
        normalized_df = normalize_data(df, method=method, columns=[column])
        normalized_df[f'{column}_norm'].plot(label=ticker)
    
    plt.title(f'Normalized {column} Prices ({method} normalization)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_data_quality(df):
    """
    Analyze dataset quality including missing values, statistics, and distributions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing stock market data
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics and statistics
    """
    quality_metrics = {
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df)) * 100,
        'descriptive_stats': df.describe(),
        'skewness': df.skew(),
        'kurtosis': df.kurtosis()
    }
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.select_dtypes(include=[np.number]).columns, 1):
        plt.subplot(3, 2, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    
    return quality_metrics

def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect outliers in specified columns using IQR or Z-score method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, optional
        List of columns to check for outliers. If None, uses all numeric columns.
    method : str, optional (default='iqr')
        Method to detect outliers: 'iqr' or 'zscore'
    threshold : float, optional (default=1.5)
        Threshold for IQR method (typically 1.5 or 3)
        
    Returns:
    --------
    dict
        Dictionary containing outlier indices and statistics for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers_dict = {}
    
    for column in columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            
            outliers_dict[column] = {
                'outliers': outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers)
            }
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column]))
            outliers = df[z_scores > threshold][column]
            
            outliers_dict[column] = {
                'outliers': outliers,
                'outlier_count': len(outliers)
            }
    
    return outliers_dict

def plot_outliers(df, columns=None):
    """
    Visualize outliers using box plots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, optional
        List of columns to plot. If None, uses all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(15, 5 * ((len(columns) + 1) // 2)))
    for i, column in enumerate(columns, 1):
        plt.subplot((len(columns) + 1) // 2, 2, i)
        sns.boxplot(y=df[column])
        plt.title(f'Box Plot of {column}')
    plt.tight_layout()
    
def handle_outliers(df, columns=None, method='clip'):
    """
    Handle outliers in the specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, optional
        List of columns to handle outliers. If None, uses all numeric columns.
    method : str, optional (default='clip')
        Method to handle outliers: 'clip', 'remove', or 'winsorize'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'clip':
            df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
            df_clean = df_clean[mask]
        elif method == 'winsorize':
            df_clean[column] = stats.mstats.winsorize(df_clean[column], limits=[0.05, 0.05])
    
    return df_clean

def create_lstm_model(sequence_length, n_features=1, units=50, dropout=0.2, learning_rate=0.001):

    """

    Create LSTM model for stock price prediction.

    

    Parameters:

    -----------

    sequence_length : int

        Number of time steps to look back

    n_features : int, optional (default=1)

        Number of input features

    units : int, optional (default=50)

        Number of LSTM units

    dropout : float, optional (default=0.2)

        Dropout rate

    learning_rate : float, optional (default=0.001)

        Learning rate for Adam optimizer

        

    Returns:
    """

    model = Sequential([

        LSTM(units, activation='relu', input_shape=(sequence_length, n_features), 

             return_sequences=True),

        Dropout(dropout),

        LSTM(units, activation='relu'),

        Dropout(dropout),

        Dense(1)

    ])

    

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model
def train_lstm_model(model, X_train, y_train, epochs=100, batch_size=32, 
                    validation_split=0.1, patience=10, verbose=1):
    """
    Train LSTM model with early stopping.
    
    Parameters:
    -----------
    model : tensorflow.keras.models.Sequential
        LSTM model to train
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training target
    epochs : int, optional (default=100)
        Number of training epochs
    batch_size : int, optional (default=32)
        Batch size for training
    validation_split : float, optional (default=0.1)
        Proportion of training data to use for validation
    patience : int, optional (default=10)
        Number of epochs to wait before early stopping
    verbose : int, optional (default=1)
        Verbosity mode
        
    Returns:
    --------
    tensorflow.keras.callbacks.History
        Training history
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                 restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    return history

def evaluate_lstm_predictions(y_true, y_pred):
    """
    Evaluate LSTM model predictions using multiple metrics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics

def plot_lstm_results(history, y_test, predictions, title="Stock Price Prediction"):
    """
    Plot LSTM training history and predictions.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history
    y_test : numpy.ndarray
        True test values
    predictions : numpy.ndarray
        Model predictions
    title : str, optional (default="Stock Price Prediction")
        Plot title
    """
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def predict_stock_prices_lstm(df, target_col='Close', sequence_length=60, train_split=0.8, units=50, dropout=0.2):
    """
    Complete pipeline for stock price prediction using LSTM.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock data
    target_col : str, optional (default='Close')
        Target column for prediction
    sequence_length : int, optional (default=60)
        Number of time steps to look back
    train_split : float, optional (default=0.8)
        Proportion of data to use for training
    units : int, optional (default=50)
        Number of LSTM units
    dropout : float, optional (default=0.2)
        Dropout rate
        
    Returns:
    --------
    dict
        Dictionary containing model, history, predictions, and evaluation metrics
    """
    print("Processing data...")
    
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[target_col]])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    model = create_lstm_model(sequence_length, units=units, dropout=dropout)
    history = train_lstm_model(model, X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)
    
    # Evaluate predictions
    metrics = evaluate_lstm_predictions(y_test_actual, predictions)
    
    # Plot results
    plot_lstm_results(history, y_test_actual, predictions, 
                     title="Stock Price Prediction")
    
    # Store results
    results = {
        'model': model,
        'history': history,
        'predictions': predictions,
        'actual_values': y_test_actual,
        'metrics': metrics,
        'scaler': scaler
    }
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return results

def predict_stock_prices_gru(df, target_col='Close', sequence_length=60, train_split=0.8, units=50, dropout=0.2):
    """
    Complete pipeline for stock price prediction using GRU.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock data
    target_col : str, optional (default='Close')
        Target column for prediction
    sequence_length : int, optional (default=60)
        Number of time steps to look back
    train_split : float, optional (default=0.8)
        Proportion of data to use for training
    units : int, optional (default=50)
        Number of GRU units
    dropout : float, optional (default=0.2)
        Dropout rate
        
    Returns:
    --------
    dict
        Dictionary containing model, history, predictions, and evaluation metrics
    """
    print("Processing data...")
    
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[target_col]])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    model = create_gru_model(sequence_length, units=units, dropout=dropout)
    history = train_gru_model(model, X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)
    
    # Evaluate predictions
    metrics = evaluate_gru_predictions(y_test_actual, predictions)
    
    # Plot results
    plot_gru_results(history, y_test_actual, predictions, 
                     title="Stock Price Prediction using GRU")
    
    # Store results
    results = {
        'model': model,
        'history': history,
        'predictions': predictions,
        'actual_values': y_test_actual,
        'metrics': metrics,
        'scaler': scaler
    }
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return results

def create_gru_model(sequence_length, n_features=1, units=50, dropout=0.2):
    """
    Create GRU model for stock price prediction.
    
    Parameters:
    -----------
    sequence_length : int
        Number of time steps to look back
    n_features : int, optional (default=1)
        Number of input features
    units : int, optional (default=50)
        Number of GRU units
    dropout : float, optional (default=0.2)
        Dropout rate
        
    Returns:
    --------
    tensorflow.keras.models.Sequential
        GRU model
    """
    model = Sequential([
        GRU(units, activation='relu', input_shape=(sequence_length, n_features), 
             return_sequences=True),
        Dropout(dropout),
        GRU(units, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_gru_model(model, X_train, y_train, epochs=100, batch_size=32, 
                    validation_split=0.1, patience=10, verbose=1):
    """
    Train GRU model with early stopping.
    
    Parameters:
    -----------
    model : tensorflow.keras.models.Sequential
        GRU model to train
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training target
    epochs : int, optional (default=100)
        Number of training epochs
    batch_size : int, optional (default=32)
        Batch size for training
    validation_split : float, optional (default=0.1)
        Proportion of training data to use for validation
    patience : int, optional (default=10)
        Number of epochs to wait before early stopping
    verbose : int, optional (default=1)
        Verbosity mode
        
    Returns:
    --------
    tensorflow.keras.callbacks.History
        Training history
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                 restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    return history

def evaluate_gru_predictions(y_true, y_pred):
    """
    Evaluate GRU model predictions using multiple metrics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics

def plot_gru_results(history, y_test, predictions, title="Stock Price Prediction using GRU"):
    """
    Plot GRU training history and predictions.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history
    y_test : numpy.ndarray
        True test values
    predictions : numpy.ndarray
        Model predictions
    title : str, optional (default="Stock Price Prediction using GRU")
        Plot title
    """
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def predict_stock_prices_tft(
    df, target_col='Close', sequence_length=60, train_split=0.8, num_features=1,
    num_heads=1, d_model=32, num_layers=2, dropout=0.2
):
    """
    Improved pipeline for stock price prediction using Temporal Fusion Transformer (TFT).
    """
    print("Processing data...")

    # Check for missing values
    if df.isnull().any().any():
        df.fillna(method='ffill', inplace=True)  # Forward-fill missing values

    # Feature engineering
    df['SMA_20'] = df[target_col].rolling(window=20).mean()  # 20-day SMA
    df['RSI'] = calculate_rsi(df[target_col])
    df['DayOfWeek'] = df.index.dayofweek  # Day of the week (0=Monday)
    df = df.dropna()  # Drop rows with NaN after rolling window

    # Prepare data
    scaler = StandardScaler()
    features = [target_col, 'SMA_20', 'RSI']
    scaled_data = scaler.fit_transform(df[features].values)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 0])  # Target is the first feature

    X, y = np.array(X), np.array(y)

    # Split data into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train the model
    model = create_tft_model(sequence_length, len(features), num_heads, d_model, num_layers, dropout)
    history = train_tft_model(model, X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(np.hstack([predictions, np.zeros((len(predictions), len(features) - 1))]))[:, 0]
    y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))]))[:, 0]

    # Evaluate predictions
    metrics = evaluate_tft_predictions(y_test_actual, predictions)

    # Store results
    results = {
        'model': model,
        'history': history,
        'predictions': predictions,
        'actual_values': y_test_actual,
        'metrics': metrics,
        'scaler': scaler
    }

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    return results


def create_tft_model(sequence_length, num_features, num_heads=4, d_model=64, num_layers=4, dropout=0.3):
    inputs = tf.keras.Input(shape=(sequence_length, num_features))

    x = inputs
    for _ in range(num_layers):
        # Multi-Head Attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feed-Forward Network
        feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_features)
        ])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + feed_forward(x))

    outputs = tf.keras.layers.Dense(1)(x[:, -1, :])  # Predict the last time step

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

def train_tft_model(model, X_train, y_train, batch_size=32, epochs=50):
    """
    Train the Temporal Fusion Transformer (TFT) model.
    """
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    return history


def evaluate_tft_predictions(y_test_actual, predictions):
    """
    Evaluate the predictions with multiple metrics.
    """
    mae = np.mean(np.abs(y_test_actual - predictions))
    mse = np.mean(np.square(y_test_actual - predictions))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

    metrics = {
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_percentage_error": mape
    }

    return metrics


def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

