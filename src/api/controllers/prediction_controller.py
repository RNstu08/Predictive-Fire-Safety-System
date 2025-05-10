# src/api/controllers/prediction_controller.py

from flask import Blueprint, request, jsonify, current_app # Import Blueprint and request/jsonify
from pydantic import ValidationError # To catch validation errors

# Import schemas and the service function
from ..schemas.prediction_schema import PredictionInput, PredictionResponse, PredictionResult
from ..services import prediction_service # Import functions from the service module

# --- ADD PROMETHEUS CLIENT FOR CUSTOM METRICS ---
from prometheus_client import Counter
# --- END PROMETHEUS CLIENT ---

# Create a Blueprint for prediction routes
# Blueprints help organize routes, especially in larger applications
prediction_bp = Blueprint('prediction', __name__, url_prefix='/predict')

# --- DEFINE CUSTOM METRIC ---
# This counter will track the number of predictions made, labeled by the predicted class.
PREDICTION_OUTCOME_COUNTER = Counter(
    'hvs_api_prediction_outcomes_total', # Metric name
    'Total number of hazard predictions made by the API', # Description
    ['predicted_label_text'] # Labels to categorize (e.g., "Normal", "Hazardous")
)
# --- END CUSTOM METRIC ---

# Define the prediction endpoint
@prediction_bp.route('', methods=['POST']) # Route is '/predict' (POST request)
def predict():
    """
    Endpoint to receive sensor readings and return hazard predictions.
    Expects JSON input matching the PredictionInput schema.
    """
    # Get JSON data from the request
    input_data = request.get_json()
    
    # --- Input Validation using Pydantic ---
    try:
        # Validate the input data against the PredictionInput schema
        validated_input = PredictionInput(**input_data)
        # Convert Pydantic models to list of dictionaries for the service
        input_list_of_dicts = [reading.model_dump() for reading in validated_input.readings]
        current_app.logger.info(f"Received valid prediction request for {len(input_list_of_dicts)} modules.")
    except ValidationError as e:
        # If validation fails, return a 400 Bad Request error
        current_app.logger.error(f"Input validation failed: {e}")
        return jsonify({"error": "Invalid input format", "details": e.errors()}), 400
    except Exception as e:
        current_app.logger.error(f"Error processing input JSON: {e}")
        return jsonify({"error": "Could not parse input JSON"}), 400

    # --- Call the Prediction Service ---
    try:
        # Ensure model/scaler are loaded (service attempts this on import, but check)
        if prediction_service.MODEL is None or prediction_service.SCALER is None:
             # Attempt to reload if not loaded (e.g., if server restarted worker)
             if not prediction_service.load_model_and_scaler():
                  raise RuntimeError("Model/Scaler could not be loaded.")

        # Preprocess the validated input data
        scaled_features, module_ids = prediction_service.preprocess_input(input_list_of_dicts)

        # Make predictions
        predictions, probabilities = prediction_service.predict_hazard(scaled_features)

        # --- Format the Response ---
        results = []
        for i, pred in enumerate(predictions):
            # --- INCREMENT CUSTOM METRIC ---
            predicted_label_text = "Hazardous" if pred == 1 else "Normal"
            PREDICTION_OUTCOME_COUNTER.labels(predicted_label_text=predicted_label_text).inc()
            # --- END INCREMENT ---

            result = PredictionResult(
                Module_ID=module_ids[i] if i < len(module_ids) else "Unknown",
                hazard_prediction=int(pred), # Ensure prediction is int
                hazard_probability=float(probabilities[i]) if probabilities is not None else None # Ensure prob is float
            )
            results.append(result)

        response_data = PredictionResponse(
            predictions=results,
            model_version=prediction_service.MODEL_VERSION # Get version from service
        )
        current_app.logger.info(f"Successfully generated {len(results)} predictions.")
        # Use .model_dump_json() for Pydantic v2+ or .json() for v1 to serialize
        # Use .model_dump() for dict output
        return jsonify(response_data.model_dump()), 200

    except ValueError as e: # Catch specific errors from preprocessing/service
        current_app.logger.error(f"Value error during prediction: {e}")
        return jsonify(PredictionResponse(predictions=[], error=str(e)).model_dump()), 400
    except RuntimeError as e: # Catch errors like model not loaded
         current_app.logger.error(f"Runtime error during prediction: {e}")
         return jsonify(PredictionResponse(predictions=[], error=str(e)).model_dump()), 500
    except Exception as e:
        # Catch any other unexpected errors during preprocessing or prediction
        current_app.logger.error(f"Unexpected error during prediction: {e}", exc_info=True) # Log traceback
        return jsonify(PredictionResponse(predictions=[], error="An internal error occurred during prediction.").model_dump()), 500

