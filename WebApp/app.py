from flask import Flask, render_template, request, flash
import os
import logging
from stock_predictor import StockPredictor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'your_secret_key_here'

# Initialize StockPredictor
predictor = StockPredictor()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            company_code = request.form.get("company")
            if company_code not in predictor.COMPANIES:
                flash("Invalid company selection.")
                return render_template("index.html")
            
            # Fetch and prepare data
            historical_data = predictor.fetch_historical_data(company_code)
            if historical_data is None:
                flash("Error fetching historical data.")
                return render_template("index.html")
            
            prepared_data = predictor.prepare_data(historical_data, company_code)
            if prepared_data is None:
                flash("Error preparing data.")
                return render_template("index.html")
            
            # Load model and generate predictions
            model, training = predictor.load_model(company_code)
            if model is None or training is None:
                flash("Error loading model.")
                return render_template("index.html")
            
            predictions = predictor.generate_predictions(model, training, prepared_data)
            if predictions is None:
                flash("Error generating predictions.")
                return render_template("index.html")
            
            # Create and return visualization
            graph_url = predictor.create_graph(historical_data, predictions, company_code)
            if graph_url is None:
                flash("Error creating visualization.")
                return render_template("index.html")
            
            return render_template("index.html", graph_url=graph_url)
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            flash(f"An error occurred: {str(e)}")
            return render_template("index.html")
    
    return render_template("index.html")

if __name__ == '__main__':
    # Ensure Models directory exists
    models_dir = os.path.join(current_dir, 'Models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Start the application
    app.run(debug=True, port=5000)
