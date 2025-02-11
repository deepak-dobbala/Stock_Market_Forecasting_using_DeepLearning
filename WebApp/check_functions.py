from stock_predictor import StockPredictor
import pandas as pd
import numpy as np

def run_tests():
    print("\n=== Testing StockPredictor Functions ===\n")
    
    predictor = StockPredictor()
    
    print("1. Testing fetch_historical_data...")
    # [fetch_historical_data tests from above]
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
            required_cols = ['time_idx', 'group', 'month', 'day_of_week', 'MA7', 'MA21', 'RSI']
            missing_cols = [col for col in required_cols if col not in prepared_df.columns]
            if not missing_cols:
                print("✅ All required columns present")
            else:
                print(f"❌ Missing columns: {missing_cols}")
        else:
            print("❌ prepare_data failed")
    
    print("\n3. Testing calculate_rsi...")
    # Test calculate_rsi
    test_prices = pd.Series([10, 11, 10, 11, 12, 11, 12, 13, 12, 11])
    rsi = predictor.calculate_rsi(test_prices, periods=3)

    if rsi is not None:
        print("✅ calculate_rsi works")
        print(f"RSI values range: {rsi.min():.2f} to {rsi.max():.2f}")
        if all((rsi >= 0) & (rsi <= 100)):
            print("✅ RSI values are in correct range (0-100)")
        else:
            print("❌ RSI values outside expected range")
    else:
        print("❌ calculate_rsi failed")
    
    print("\n4. Testing load_model...")
    # Test load_model
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
    
    print("\n5. Testing generate_predictions...")
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
    
    print("\n6. Testing create_graph...")
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
if __name__ == "__main__":
    run_tests() 