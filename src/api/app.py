# src/api/app.py

from flask import Flask
import logging # Import logging library

# --- ADD PROMETHEUS EXPORTER ---
from prometheus_flask_exporter import PrometheusMetrics
# --- END PROMETHEUS EXPORTER ---

# Import the blueprint from the controller
from .controllers.prediction_controller import prediction_bp
# Import the service module to trigger model loading on startup
from .services import prediction_service

def create_app():
    """Creates and configures the Flask application."""
    
    app = Flask(__name__)

    # --- INITIALIZE PROMETHEUS METRICS ---
    # This will automatically create a /metrics endpoint
    # By default, it groups metrics by endpoint and provides common HTTP metrics
    metrics = PrometheusMetrics(app)
    # You can add static info metrics about your app
    metrics.info('app_info', 'HVS Prediction API Information', version='1.0.0', major_release='1')
    # --- END INITIALIZE PROMETHEUS METRICS ---

    # --- Configuration ---
    # Set configuration variables if needed
    # app.config['SECRET_KEY'] = 'a_secure_random_secret_key' # Example if using sessions etc.
    
    # Configure logging
    # Log to stdout/stderr, which is common for containerized apps
    logging.basicConfig(level=logging.INFO, # Log INFO level and above
                        format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
    
    app.logger.info("Flask app created.")
    if hasattr(metrics, 'app'): # Check if metrics initialized correctly
            app.logger.info("Prometheus metrics exporter initialized.")


    # --- Register Blueprints ---
    # Register the prediction blueprint created in the controller file
    app.register_blueprint(prediction_bp)
    app.logger.info("Registered prediction blueprint.")

    # --- Initial Model Load Trigger ---
    # Attempt to load the model when the app starts
    # The service module already calls load_model_and_scaler() on import,
    # but we can add an explicit check/call here if needed, or rely on the
    # check within the prediction endpoint.
    if prediction_service.MODEL is None or prediction_service.SCALER is None:
        app.logger.warning("Model/Scaler not loaded on initial app creation. Will attempt load on first request.")
        # Optionally attempt loading again here, but be mindful of app context
        # with app.app_context():
        #     prediction_service.load_model_and_scaler()
    else:
         app.logger.info(f"Model version {prediction_service.MODEL_VERSION} and scaler loaded successfully at startup.")


    # --- Basic Root Route (Optional) ---
    @app.route('/')
    @metrics.do_not_track() # Example: Don't track metrics for the root endpoint if it's just a health check
    def index():
        app.logger.info("Root endpoint '/' accessed.")
        return "HVS Prediction API is running!"

    return app

# --- Run the App (for local development) ---
# This block allows running the app directly using 'python src/api/app.py'
# For production, use a proper WSGI server like Gunicorn or Waitress.
if __name__ == '__main__':
    app = create_app()
    # Runs the development server
    # Debug=True automatically reloads on code changes but SHOULD NOT be used in production
    app.run(host='0.0.0.0', port=5001, debug=False)

