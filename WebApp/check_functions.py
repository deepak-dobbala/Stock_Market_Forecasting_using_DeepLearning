from stock_predictor import StockPredictor
import pandas as pd
import numpy as np
import os

def run_tests():
    print("\n=== Testing StockPredictor Functions ===\n")
    
    predictor = StockPredictor()
    
    # Verify model files first
    print("Checking model files...")
    if not predictor.verify_model_files():
        print("❌ Some model files are missing")
        return
    print("✅ All model files present")
    
    print("1. Testing fetch_historical_data...")
    # Test fetch_historical_data
    predictor = StockPredictor()

    # Test valid company
    df = predictor.fetch_historical_data('GOOGL')
    if df is not None:
        print("✅ fetch_historical_data works for valid company")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("❌ fetch_historical_data failed for valid company")

    # Test invalid company
    df_invalid = predictor.fetch_historical_data('INVALID')
    if df_invalid is None:
        print("✅ fetch_historical_data correctly handles invalid company")
    else:
        print("❌ fetch_historical_data failed for invalid company")
    
    print("\n2. Testing prepare_data...")
    # Test prepare_data
    if df is not None:  # Using df from previous test
        prepared_df = predictor.prepare_data(df, 'GOOGL')
        if prepared_df is not None:
            print("✅ prepare_data works")
            print("Checking required columns...")
            required_cols = ['time_idx', 'group', 'month', 'day_of_week', 'MA7', 'MA21']
            missing_cols = [col for col in required_cols if col not in prepared_df.columns]
            if not missing_cols:
                print("✅ All required columns present")
            else:
                print(f"❌ Missing columns: {missing_cols}")
        else:
            print("❌ prepare_data failed")
    
    print("\n3. Testing load_model...")
    # Test load_model
    model, training = test_load_model(predictor)
    
    print("\n4. Testing generate_predictions...")
    # Test generate_predictions
    if all([model, training, prepared_df is not None]):
        predictions = predictor.generate_predictions(model, training, prepared_df)
        if predictions is not None:
            print("✅ generate_predictions works")
            print(f"Number of predictions: {len(predictions)}")
            print(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
        else:
            print("❌ generate_predictions failed")
    else:
        print("❌ Cannot test generate_predictions due to previous failures")
    
    print("\n5. Testing create_graph...")
    # Test create_graph
    if df is not None and predictions is not None:
        graph = predictor.create_graph(df, predictions, 'GOOGL')
        if graph is not None:
            print("✅ create_graph works")
            print("Graph generated successfully (base64 string)")
        else:
            print("❌ create_graph failed")
    else:
        print("❌ Cannot test create_graph due to previous failures")

def test_load_model(predictor):
    """Test loading the model with detailed feedback"""
    print("\n3. Testing load_model...")
    
    # Check Models directory
    if not os.path.exists(predictor.models_dir):
        print(f"❌ Models directory not found at: {predictor.models_dir}")
        return None, None
    
    # Check model file
    model_path = os.path.join(predictor.models_dir, f'model_googl.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print("Please ensure you have trained model files in the Models directory")
        return None, None
    
    # Try loading the model
    model, training = predictor.load_model('GOOGL')
    if model is not None and training is not None:
        print("✅ load_model works")
        print("Checking model attributes...")
        if hasattr(model, 'predict'):
            print("✅ Model has predict method")
        else:
            print("❌ Model missing predict method")
    else:
        print("❌ load_model failed")
        print("Make sure model files exist in the Models directory")
        print(f"Expected model path: {model_path}")
    
    return model, training

if __name__ == "__main__":
    run_tests() 