from tensorflow.keras.models import load_model

def evaluate_model(model_path, X_test, y_test):
    model = load_model(model_path)
    predictions = model.predict(X_test)
    
    mse = np.mean((predictions - y_test) ** 2)
    print(f'Mean Squared Error: {mse}')
    
    return predictions

# Example usage:
# predictions = evaluate_model('Models/AAPL_best_model.h5', X_test_AAPL, y_test_AAPL)
