import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def fetch_data(tickers, start_date, end_date):
    """
    Fetch stock data for multiple tickers and return a dictionary of DataFrames.
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    dict
        Dictionary with tickers as keys and their respective DataFrames as values
    """
    data = {}
    
    for ticker in tickers:
        # Download data for each ticker
        df = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = df
    
    return data

def normalize_data(data_dict, columns=None, method='first_value', scale=100):
    """
    Normalize stock data using different methods.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing DataFrames for each ticker
    columns : list, optional
        List of columns to normalize. If None, normalizes all numeric columns
    method : str, optional (default='first_value')
        Normalization method:
        - 'first_value': Divide by first value and multiply by scale
        - 'min_max': Min-max scaling to [0, scale]
        - 'zscore': Standardize to zero mean and unit variance
    scale : float, optional (default=100)
        Scale factor for normalization
        
    Returns:
    --------
    dict
        Dictionary containing normalized DataFrames
    """
    normalized_data = {}
    
    for ticker, df in data_dict.items():
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        normalized_df = df.copy()
        
        for column in columns:
            if method == 'first_value':
                # Normalize by first value (like your reference code)
                normalized_df[f'{column}_normalized'] = df[column].div(df[column].iloc[0]).mul(scale)
            
            elif method == 'min_max':
                # Min-max scaling
                min_val = df[column].min()
                max_val = df[column].max()
                normalized_df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val) * scale
            
            elif method == 'zscore':
                # Z-score normalization
                mean = df[column].mean()
                std = df[column].std()
                normalized_df[f'{column}_normalized'] = (df[column] - mean) / std
        
        normalized_data[ticker] = normalized_df
    
    return normalized_data

def plot_normalized_data(normalized_data, column, tickers=None):
    """
    Plot normalized values for comparison across different stocks.
    
    Parameters:
    -----------
    normalized_data : dict
        Dictionary containing normalized DataFrames
    column : str
        Column to plot (without '_normalized' suffix)
    tickers : list, optional
        List of tickers to plot. If None, plots all tickers
    """
    plt.figure(figsize=(15, 7))
    
    if tickers is None:
        tickers = list(normalized_data.keys())
    
    for ticker in tickers:
        df = normalized_data[ticker]
        normalized_column = f'{column}_normalized'
        if normalized_column in df.columns:
            df[normalized_column].plot()
    
    plt.title(f'Normalized {column} Values Comparison')
    plt.legend(tickers)
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