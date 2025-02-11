from flask import Flask, render_template, request, flash
import torch
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

def load_model(company_code):
    # Build the model path using lowercase company code (e.g., "model_googl.pth")
    model_path = f"models/model_{company_code.lower()}.pth"
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    else:
        return None

def predict_stock_price(model):
    # Replace with your actual input data preparation
    # This is a dummy input tensor, adjust the shape and values as needed.
    input_tensor = torch.randn(1, 10, 128)  # Example shape: (batch_size, sequence_length, feature_size)
    with torch.no_grad():
        predictions = model(input_tensor)
    # Convert predictions to a list; adjust if your model returns a different structure
    return predictions.numpy().flatten().tolist()

def create_prediction_graph(predictions, company_code):
    plt.figure(figsize=(8, 5))
    plt.plot(predictions, marker='o', linestyle='-', color='b', label=f"{company_code} Predicted Prices")
    plt.xlabel("Future Time Steps")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Predictions for {company_code}")
    plt.legend()
    plt.grid()
    # Save plot to a BytesIO buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    # Encode image to base64 string so it can be embedded in HTML
    graph_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return graph_base64

@app.route("/", methods=["GET", "POST"])
def index():
    graph_url = None
    if request.method == "POST":
        company_code = request.form.get("company", "").strip().upper()
        if not company_code:
            flash("Please enter a valid company code.")
            return render_template("index.html")
        
        model = load_model(company_code)
        if model is None:
            flash(f"Model for '{company_code}' not found.")
            return render_template("index.html")
        
        predictions = predict_stock_price(model)
        graph_url = create_prediction_graph(predictions, company_code)
    return render_template("index.html", graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)
