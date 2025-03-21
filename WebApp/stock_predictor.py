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
import logging
import os
from config import CONFIG
from exceptions import ModelError, DataError

# Configure logging
def setup_logger():
    """Configure logging with enhanced formatting"""
    log_file = os.path.join(os.path.dirname(__file__), 'stock_predictor.log')
    
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

class StockPredictor:
    # Update COMPANIES dictionary
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.current_dir, 'Models')
        
        # Available companies - Updated list
        self.COMPANIES = {
            'GOOGL': 'Google',
            'AMZN': 'Amazon',  # Added Amazon
            'IBM': 'IBM',
            'AAPL': 'Apple'
        }
        self.model_version = "1.0.0"
        self.model_configs = {
            'GOOGL': {'timestamp': None, 'metrics': None},
            'AAPL': {'timestamp': None, 'metrics': None},
            'IBM': {'timestamp': None, 'metrics': None},
            'AMZN': {'timestamp': None, 'metrics': None}
        }
    from functools import lru_cache

    @lru_cache(maxsize=32)
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
            logger.info(f"Preparing data for {company_code}")
            logger.info(f"Input data shape: {df.shape}")
            
            # Create a copy and reset index
            df = df.copy()
            df = df.reset_index(drop=True)
            
            # Add company code to column names
            df[f'Close_{company_code}'] = df['Close']
            df[f'High_{company_code}'] = df['High']
            df[f'Low_{company_code}'] = df['Low']
            df['Open'] = df['Open']
            df['Volume'] = df['Volume']
            
            # Add time features
            df['time_idx'] = range(len(df))
            df['group'] = company_code
            df['month'] = df['Date'].dt.month
            df['day_of_week'] = df['Date'].dt.dayofweek
            
            # Calculate technical indicators
            df['MA7'] = df[f'Close_{company_code}'].rolling(window=7).mean()
            df['MA21'] = df[f'Close_{company_code}'].rolling(window=21).mean()
            
            # Forward fill missing values
            df = df.ffill()  # Using ffill() instead of fillna(method='ffill')
            
            logger.info(f"Final data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def load_model(self, company_code):
        """Load the trained model from the Models folder"""
        try:
            logger.info(f"Loading model for {company_code}")
            model_path = os.path.join(self.models_dir, f'model_{company_code.lower()}.pth')
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None, None

            # Create sample data with correct structure
            sample_length = 150
            df = pd.DataFrame({
                'time_idx': range(sample_length),
                'group': [company_code] * sample_length,
                f'Close_{company_code}': [0.0] * sample_length,
                f'High_{company_code}': [0.0] * sample_length,
                f'Low_{company_code}': [0.0] * sample_length,
                'Open': [0.0] * sample_length,
                'Volume': [0.0] * sample_length,
                'year': [datetime.now().year] * sample_length,
                'month': [datetime.now().month] * sample_length,
                'day_of_week': [datetime.now().weekday()] * sample_length,
                'MA7': [0.0] * sample_length,
                'MA21': [0.0] * sample_length
            })

            # Updated TimeSeriesDataSet configuration
            training = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target=f"Close_{company_code}",
                group_ids=["group"],
                min_encoder_length=30,  # Reduced from 60
                max_encoder_length=90,  # Reduced from 120
                max_prediction_length=30,
                static_categoricals=["group"],
                time_varying_known_reals=["month", "day_of_week"],
                time_varying_unknown_reals=[
                    f"High_{company_code}", 
                    f"Low_{company_code}", 
                    f"Close_{company_code}",
                    "Open",
                    "Volume",
                    "MA7", 
                    "MA21"
                ],
                target_normalizer=GroupNormalizer(
                    groups=["group"],
                    transformation="softplus"
                )
            )

            # Initialize model with matched parameters
            model = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.001,
                hidden_size=64,
                attention_head_size=4,
                dropout=0.3,
                hidden_continuous_size=32,
                loss=QuantileLoss(),
                reduce_on_plateau_patience=4,
                optimizer="ranger",
                lstm_layers=2
            )

            try:
                # Load model weights with proper device handling
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model = model.to(device)
                model.eval()
                
                logger.info(f"Model loaded successfully to {device}")
                return model, training

            except Exception as e:
                logger.error(f"Error loading model weights: {str(e)}")
                logger.exception("Full traceback:")
                return None, None

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
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot historical data with dates
            historical_dates = historical_data['Date'].values[-30:]
            historical_prices = historical_data['Close'].values[-30:]
            
            # Create date range for predictions
            last_date = historical_dates[-1]
            future_dates = pd.date_range(start=last_date, periods=len(predictions)+1)[1:]
            
            plt.plot(historical_dates, historical_prices, 
                    label='Historical', color='blue', marker='o', markersize=4)
            
            plt.plot(future_dates, predictions,
                    label='Predicted', color='red', linestyle='--', marker='x', markersize=4)
            
            plt.title(f'{self.COMPANIES[company_code]} Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price (Normalized)')
            plt.legend()
            plt.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert plot to base64
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plt.close()
            
            return base64.b64encode(img.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            return None
    def verify_model_files(self):
        """Verify all model files exist"""
        missing_models = []
        for company in self.COMPANIES.keys():
            model_path = os.path.join(self.models_dir, f'model_{company.lower()}.pth')
            if not os.path.exists(model_path):
                missing_models.append(company)
                logger.error(f"Model file missing: {model_path}")
            else:
                logger.info(f"Model file found: {model_path}")
        
        if missing_models:
            logger.error(f"Missing model files for: {', '.join(missing_models)}")
            return False
        return True