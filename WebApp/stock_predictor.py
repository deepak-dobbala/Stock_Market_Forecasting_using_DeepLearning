import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging
import io
import base64
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.current_dir, 'Models')
        
        # Available companies
        self.COMPANIES = {
            'GOOGL': 'Google',
            'MSFT': 'Microsoft',
            'IBM': 'IBM',
            'AAPL': 'Apple'
        }

    def fetch_historical_data(self, company_code):
        """Fetch the last 120 days of historical data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=120)
            df = yf.download(company_code, start=start_date, end=end_date)
            
            if df.empty:
                logger.error(f"No data fetched for {company_code}")
                return None
                
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None

    def prepare_data(self, df, company_code):
        """Prepare data for the model"""
        try:
            # Add necessary columns
            df['time_idx'] = range(len(df))
            df['group'] = company_code
            
            # Create features
            df['month'] = df['Date'].dt.month
            df['day_of_week'] = df['Date'].dt.dayofweek
            
            # Calculate technical indicators
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            
            # Normalize numerical features
            scaler = MinMaxScaler()
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21']
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
            
            return df
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def load_model(self, company_code):
        """Load the trained model from the Models folder"""
        try:
            model_path = os.path.join(self.models_dir, f'model_{company_code.lower()}.pth')
            logger.info(f"Attempting to load model from: {model_path}")
            
            # Check if Models directory exists
            if not os.path.exists(self.models_dir):
                logger.error(f"Models directory not found at: {self.models_dir}")
                os.makedirs(self.models_dir)
                logger.info(f"Created Models directory at: {self.models_dir}")
                return None, None
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                logger.error(f"Please ensure model_{company_code.lower()}.pth exists in the Models directory")
                return None, None
            
            logger.info(f"Found model file of size: {os.path.getsize(model_path)} bytes")
            
            try:
                # Try to load the model file
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                logger.info("Successfully loaded model state dict")
            except Exception as e:
                logger.error(f"Error loading model file: {str(e)}")
                return None, None
            
            # Create dataset parameters
            training = TimeSeriesDataSet(
                data=pd.DataFrame({
                    'time_idx': range(120),
                    'group': [company_code] * 120,
                    'Close': [0.0] * 120,
                    'Open': [0.0] * 120,
                    'High': [0.0] * 120,
                    'Low': [0.0] * 120,
                    'Volume': [0.0] * 120,
                    'MA7': [0.0] * 120,
                    'MA21': [0.0] * 120,
                    'month': [1] * 120,
                    'day_of_week': [1] * 120
                }),
                time_idx="time_idx",
                target="Close",
                group_ids=["group"],
                max_encoder_length=90,
                max_prediction_length=30,
                static_categoricals=["group"],
                time_varying_known_reals=["month", "day_of_week"],
                time_varying_unknown_reals=[
                    "Close", "Open", "High", "Low", "Volume",
                    "MA7", "MA21"
                ],
                target_normalizer=GroupNormalizer(
                    groups=["group"],
                    transformation="softplus"
                )
            )
            
            # Initialize model
            model = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.001,
                hidden_size=64,
                attention_head_size=4,
                dropout=0.3,
                hidden_continuous_size=32,
                loss=QuantileLoss()
            )
            
            # Load the trained weights
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info("Successfully initialized and loaded model")
            return model, training

        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}")
            logger.exception("Full traceback:")
            return None, None

    def generate_predictions(self, model, training, data):
        """Generate predictions for the next 30 days"""
        try:
            # Prepare prediction data
            prediction_data = training.predict(data, mode="raw", return_x=True)
            
            # Generate predictions
            with torch.no_grad():
                predictions = model.predict(prediction_data)
            
            # Get median predictions
            median_predictions = predictions.median(dim=1).values.numpy()
            return median_predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return None

    def create_graph(self, historical_data, predictions, company_code):
        """Create visualization of historical data and predictions"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            historical_dates = historical_data['Date'].values[-30:]
            historical_prices = historical_data['Close'].values[-30:]
            plt.plot(range(len(historical_dates)), historical_prices, 
                    label='Historical', color='blue')
            
            # Plot predictions
            plt.plot(range(len(historical_dates)-1, len(historical_dates) + len(predictions)-1),
                    predictions, label='Predicted', color='red', linestyle='--')
            
            plt.title(f'{self.COMPANIES[company_code]} Stock Price Prediction')
            plt.xlabel('Days')
            plt.ylabel('Price (Normalized)')
            plt.legend()
            plt.grid(True)
            
            # Convert plot to base64
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plt.close()
            
            return base64.b64encode(img.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            return None 
