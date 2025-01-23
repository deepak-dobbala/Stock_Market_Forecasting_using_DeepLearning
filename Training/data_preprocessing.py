import pandas as pd
import yfinance as yf

def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = df
    return data

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  
    df['Return'] = df['Close'].pct_change() 
    df.dropna(inplace=True) 
    return df
