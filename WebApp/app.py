from flask import Flask, render_template, request, flash
import torch
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.preprocessing import MinMaxScaler
from config import CONFIG
from exceptions import ModelError, DataError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app with correct template folder path
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Dictionary mapping company codes to their full names
COMPANIES = {
    'GOOGL': 'Google',
    'AMZN': 'Amazon',  # Added Amazon
    'IBM': 'IBM',
    'AAPL': 'Apple'
}


def prepare_data_for_prediction(company_code):
    """Prepare the last 120 days of data for prediction"""
    try:
        logger.debug(f"Preparing data for {company_code}")
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)  # Fetch extra days to ensure we have enough data
        df = yf.download(company_code, start=start_date, end=end_date)
        
        if df.empty:
            logger.error("No data fetched from Yahoo Finance")
            return None
        
        # Process the data
        df.reset_index(inplace=True)
        df['time_idx'] = range(len(df))
        df['group'] = company_code
        df['year'] = df['Date'].dt.year
        
        # Calculate features
        feature_cols = ['High', 'Low', 'Close', 'Volume']
        for col in feature_cols:
            # Add relative change
            df[f'{col}_relative_change'] = df[col].pct_change()
            # Add rolling mean
            df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
        
        # Forward fill and backward fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize the features
        scaler = MinMaxScaler()
        numeric_cols = ['High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df.tail(120).copy()
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        return None

def load_model(company_code):
    """Load the TFT model for the specified company"""
    try:
        model_path = os.path.join(current_dir, 'Models', f'model_{company_code.lower()}.pth')
        logger.debug(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None

        # Define the training dataset parameters
        training_cutoff = datetime.now() - timedelta(days=30)
        max_prediction_length = 30
        max_encoder_length = 120
        
        # Create the training dataset
        training = TimeSeriesDataSet(
            data=pd.DataFrame({
                'time_idx': range(max_encoder_length),
                'group': [company_code] * max_encoder_length,
                'Close': [0.0] * max_encoder_length,
                'High': [0.0] * max_encoder_length,
                'Low': [0.0] * max_encoder_length,
                'Volume': [0.0] * max_encoder_length,
            }),
            time_idx='time_idx',
            target='Close',
            group_ids=['group'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=['group'],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=[
                'Close',
                'High',
                'Low',
                'Volume'
            ],
            target_normalizer=GroupNormalizer(
                groups=['group'],
                transformation='softplus'
            ),
        )

        # Initialize model
        model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=128,        # Updated from 64
        attention_head_size=8,  # Updated from 4
        dropout=0.5,           # Updated from 0.3
        hidden_continuous_size=64,  # Updated from 32
        loss=QuantileLoss()
    )

        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, training

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.exception("Full traceback:")
        return None, None

def predict_stock_price(model, training, data):
    """Generate predictions using the loaded model"""
    try:
        logger.debug("Starting prediction")
        
        # Create prediction dataset using the same parameters as training
        prediction_data = training.predict(
            data,
            mode="raw",
            return_x=True
        )
        
        # Generate predictions
        with torch.no_grad():
            predictions = model.predict(prediction_data)
            
        # Get the median prediction (0.5 quantile)
        median_predictions = predictions.median(dim=1).values.numpy()
        
        return median_predictions
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.exception("Full traceback:")
        return None

def create_prediction_graph(historical_data, predictions, company_code):
    """Create a graph showing historical and predicted stock prices"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 30 days)
    historical_close = historical_data[f'Close_{company_code}'].values[-30:]
    plt.plot(range(30), historical_close, 
             label='Historical', color='blue')
    
    # Plot predictions
    plt.plot(range(29, 29 + len(predictions)), predictions, 
             label='Predicted', color='red', linestyle='--')
    
    plt.title(f'{COMPANIES[company_code]} ({company_code}) Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return base64.b64encode(img_buffer.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            logger.debug("Received POST request")
            company_code = request.form.get("company")
            logger.debug(f"Selected company: {company_code}")
            
            if company_code not in COMPANIES:
                flash("Invalid company selection.")
                return render_template("index.html")
            
            # Prepare data
            data = prepare_data_for_prediction(company_code)
            if data is None:
                flash("Error preparing data for prediction.")
                return render_template("index.html")
            
            # Load model and training configuration
            model, training = load_model(company_code)
            if model is None or training is None:
                flash(f"Model for {company_code} is not available.")
                return render_template("index.html")
            
            # Generate predictions
            predictions = predict_stock_price(model, training, data)
            if predictions is None:
                flash("Error generating predictions.")
                return render_template("index.html")
            
            # Create and return the graph
            graph_url = create_prediction_graph(data, predictions, company_code)
            return render_template("index.html", graph_url=graph_url)
            
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        flash(f"An error occurred: {str(e)}")
        return render_template("index.html")
    
    return render_template("index.html")

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application")
        models_dir = os.path.join(current_dir, 'Models')
        model_path = os.path.join(models_dir, 'model_googl.pth')
        
        # Print detailed debug information
        print("=== Debug Information ===")
        print(f"Current directory: {current_dir}")
        print(f"Model directory exists: {os.path.exists(models_dir)}")
        print(f"Model path: {model_path}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print(f"Model file size: {os.path.getsize(model_path)} bytes")
            try:
                # Try to load the state dict to verify it's valid
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                print("Model state dict loaded successfully")
                print(f"Number of keys in state dict: {len(state_dict.keys())}")
                print(f"First few keys: {list(state_dict.keys())[:5]}")
            except Exception as e:
                print(f"Error loading model state dict: {str(e)}")
        else:
            print(f"WARNING: Model file not found at {model_path}")
            
        # Ensure the Models directory exists
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created Models directory at {models_dir}")
        
        # Check if model files exist
        for company in COMPANIES:
            company_model_path = os.path.join(models_dir, f'model_{company.lower()}.pth')
            if not os.path.exists(company_model_path):
                logger.warning(f"Model file missing: {company_model_path}")
        
        print("\n=== Starting Flask Server ===")
        app.run(debug=True, port=5000)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Critical error: {str(e)}")
