import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import yfinance as yf
from datetime import datetime, timedelta


def predict_future_prices(company_code, max_encoder_length=120, max_prediction_length=30):
    
    try:
        # 1. Load historical data
        end_date = datetime.now()
        # Get more days than needed to ensure we have enough after weekends/holidays
        start_date = end_date - timedelta(days=max_encoder_length * 2)
        df = yf.download(company_code, start=start_date, end=end_date)
        
        print(f"Downloaded {len(df)} days of data for {company_code}")
        
        if len(df) < max_encoder_length:
            raise ValueError(f"Not enough data available. Got {len(df)} days but need at least {max_encoder_length}")
        
        # Fix MultiIndex columns
        df.columns = [f"{col[0]}_{company_code}" if isinstance(col, tuple) else col 
                     for col in df.columns]
        df.reset_index(inplace=True)
        
        # 2. Prepare data format
        df['time'] = range(len(df))
        df['group'] = company_code
        df['year'] = df['Date'].dt.year
        
        # Get last values for display
        last_date = df['Date'].iloc[-1]
        last_known_price = df[f'Close_{company_code}'].iloc[-1]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Enable CUDA optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # 3. Create training dataset (needed to initialize the model structure)
        training_cutoff = len(df) - max_prediction_length
        training_data = df[:training_cutoff].copy()
        
        training_dataset = TimeSeriesDataSet(
            training_data,
            time_idx="time",
            target=f"Close_{company_code}",
            group_ids=["group"],
            min_encoder_length=max_encoder_length//2,  # More flexible
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group"],
            static_reals=["year"],
            time_varying_known_reals=["time"],
            time_varying_unknown_reals=[
                f"High_{company_code}", 
                f"Low_{company_code}", 
                f"Close_{company_code}"
            ],
            target_normalizer=GroupNormalizer(
                groups=["group"],
                transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # 4. Load trained model and ensure it's on CUDA
        model_path = f"C:/Users/deepa/OneDrive/Documents/Project Documents/Stock_Market_Forecasting_using_DeepLearning/WebApp/Models/model_{company_code.lower()}.pth"
        
        # Initialize model with same parameters as training
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.001,
            hidden_size=128,
            attention_head_size=8,
            dropout=0.5,
            hidden_continuous_size=64,
            loss=QuantileLoss()
        ).to(device)  # Move model to CUDA immediately
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        tft.load_state_dict(state_dict)
        tft.eval()
        
        # 5. Create prediction dataset
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, 
            df[-max_encoder_length:],
            predict=True,
            stop_randomization=True
        )
        
        # Create dataloader with CUDA pinned memory for faster transfer
        prediction_dataloader = prediction_dataset.to_dataloader(
            batch_size=1, 
            shuffle=False,
            pin_memory=True,  # Enable pinned memory for faster CUDA transfer
            num_workers=2,    # Reduced from 4 to 2 to avoid potential issues
            persistent_workers=True  # Add this to speed up dataloader worker initialization
        )
        
        # 6. Make predictions - FIXED: proper handling of TFT prediction output
        predictions = tft.predict(
            prediction_dataloader,
            mode="prediction",
            return_x=True,
            return_y=True,
            return_index=True
        )
        
        # 7. Extract predictions - FIXED: correctly access predictions
        # The predictions object from TFT is a Prediction object with specific attributes
        prediction_values = predictions.output.cpu().numpy()
        
        # Calculate confidence intervals
        # TFT by default returns quantiles, so let's extract them
        quantiles = tft.loss.quantiles
        
        # Find the indices of the 10th and 90th percentiles
        q10_idx = np.abs(np.array(quantiles) - 0.1).argmin()
        q50_idx = np.abs(np.array(quantiles) - 0.5).argmin()
        q90_idx = np.abs(np.array(quantiles) - 0.9).argmin()
        
        # Extract the specific quantiles
        lower_bound = prediction_values[..., q10_idx]
        median_predictions = prediction_values[..., q50_idx]
        upper_bound = prediction_values[..., q90_idx]
        
        # 8. Create future dates starting from today for visualization
        future_dates = []
        start_date = datetime.now()
        for i in range(max_prediction_length):
            days_to_add = i + 1
            future_date = start_date + timedelta(days=days_to_add)
            # Skip weekends
            while future_date.weekday() > 4:
                future_date = future_date + timedelta(days=1)
                days_to_add += 1
            future_dates.append(future_date)
        
        # 9. Visualize results
        plt.figure(figsize=(12, 6))

        # Generate new dates starting from 20-03-2025 for the historical data
        def generate_trading_days(start_date, num_days):
            dates = []
            current_date = start_date
            while len(dates) < num_days:
                if current_date.weekday() < 5:  # Monday to Friday
                    dates.append(current_date)
                current_date += timedelta(days=1)
            return dates

        # Plot historical data with new dates
        historical_prices = df[f'Close_{company_code}'].values[-30:]
        start_date = datetime.now().date()
        new_dates = generate_trading_days(start_date, len(historical_prices))
        plt.plot(new_dates, historical_prices, label='Prediction', color='blue', marker='o')

        # Plot historical data only
        plt.title(f'{company_code} Stock Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{company_code}_historical.png")
        plt.show()
                
        # 10. Return prediction data
        prediction_results = {
            'company': company_code,
            'last_date': last_date,
            'future_dates': future_dates,
            'predictions': median_predictions.flatten(),
            'lower_bound': lower_bound.flatten(),
            'upper_bound': upper_bound.flatten(),
            'last_known_price': last_known_price
        }
        
        return prediction_results
        
    except Exception as e:
        print(f"Error predicting future prices for {company_code}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


# Example usage
if __name__ == "__main__":
    companies_list =['GOOGL','AAPL','IBM','AMZN']
    company = 'AAPL'
    predictions = predict_future_prices(company, max_encoder_length=120, max_prediction_length=30)
    