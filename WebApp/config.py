CONFIG = {
    'MODEL_PARAMS': {
        'learning_rate': 0.001,
        'hidden_size': 128,
        'attention_head_size': 8,
        'dropout': 0.5,
        'hidden_continuous_size': 64
    },
    'DATA_PARAMS': {
        'max_encoder_length': 120,
        'max_prediction_length': 30,
        'training_cutoff': 120
    },
    'FEATURE_ENGINEERING': {
        'moving_averages': [7, 21],
        'technical_indicators': ['RSI', 'MACD']
    }
}