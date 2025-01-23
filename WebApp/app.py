from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the best model (assumed to be a single best model for simplicity)
model = load_model('Models/model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    
    # Fetch and preprocess the latest stock data for prediction (pseudo code)
    
    prediction = model.predict(processed_input_data)  # Use your preprocessing function here
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
