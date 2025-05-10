
---

### Phase 6: API Development (Flask)

**1. Goal of this Phase:**
To build a robust, scalable, and low-latency web API that can:
* Receive new, incoming sensor data for one or more battery modules in real-time.
* Load the selected best-performing trained machine learning model and its corresponding data scaler (identified from MLflow runs in Phase 5).
* Preprocess the incoming data using the loaded scaler.
* Utilize the loaded model to make a hazard prediction (Normal/Hazardous) and provide a probability score.
* Return these predictions in a structured JSON format.
* Implement proper input validation and error handling.

**2. Rationale (Why this phase is critical):**
A trained machine learning model is only useful if it can be integrated into other systems to make predictions on new data. An API (Application Programming Interface) is the standard way to make an ML model's functionality available as a service.
* **Operationalization:** Transforms the ML model from a research artifact into a usable component in a larger system (e.g., a battery monitoring dashboard, an automated alert system).
* **Real-time Inference:** Enables on-demand predictions as new sensor data arrives, which is crucial for an early warning system. Your project description specified a low-latency (<50ms) requirement.
* **Decoupling:** The API separates the model from the applications that consume its predictions. This means the model can be updated, retrained, or even replaced without requiring changes to the consuming applications, as long as the API contract (input/output format) remains consistent.
* **Scalability & Integration:** Web APIs are inherently designed for network communication and can be scaled out (by running multiple instances behind a load balancer) to handle many concurrent requests.
* **Addressing Feedback:** This phase directly addresses the feedback you received about project structure by implementing a service layer separate from the API controller and ensuring input validation.

**3. Approach & Key Activities:**
We chose **Flask**, a lightweight and flexible Python web framework, to build the API. The API was structured with a clear separation of concerns for better organization and scalability. The key files are located in the `src/api/` directory.

* **Directory Structure for API (`src/api/`):**
    * `app.py`: Main Flask application factory and entry point (initializes Flask, registers blueprints, sets up logging, initializes Prometheus metrics).
    * `controllers/prediction_controller.py`: Handles incoming HTTP requests for the `/predict` endpoint. It's responsible for parsing request data, calling the prediction service, and formatting the HTTP response. It does *not* contain the core ML logic.
    * `services/prediction_service.py`: Contains the "business logic" – loading the ML model and scaler, preprocessing input data, making predictions using the model, and any complex logic related to the prediction task. This separation ensures the controller remains lean.
    * `schemas/prediction_schema.py`: Defines the expected structure and data types for API input and output using **Pydantic**. This enables automatic and robust input validation.

* **Key Implementation Details:**
    * **Input Validation (`schemas/prediction_schema.py` - ID: `api_schema_v1`):**
        * `ModuleFeatures(BaseModel)`: Defined the expected fields (all 30+ sensor and engineered features our model was trained on) and their data types for a single module's input.
        * `PredictionInput(BaseModel)`: Defined that the API expects a JSON object with a `readings` key, which is a list of `ModuleFeatures` objects. This allows for batch predictions.
        * `PredictionResult(BaseModel)` and `PredictionResponse(BaseModel)`: Defined the structure for the JSON response.
        * **Rationale:** Using Pydantic for schema definition and validation automatically handles parsing incoming JSON and ensures that the data passed to the service layer conforms to the expected format, preventing errors further down the line. This directly addresses the "Missing input validation" feedback.
    * **Model & Scaler Loading (`services/prediction_service.py` - ID: `api_service_v4_direct_load` or similar):**
        * **Global Variables:** `MODEL`, `SCALER`, `MODEL_VERSION`, `FEATURE_NAMES` were defined globally within the service module.
        * **`load_model_and_scaler()` function:**
            * This function is called when the service module is first imported (i.e., when the Flask app starts) to load the model and scaler into the global variables only once, avoiding the overhead of reloading them on every API request.
            * It reads the `BEST_MLFLOW_RUN_ID` environment variable (which would be set when running the API, e.g., in Docker or Kubernetes) to identify which MLflow run's artifacts to use.
            * **Artifact Loading Strategy (Iterative Debugging):** We encountered issues loading artifacts from MLflow `runs:/` URIs directly within Docker due to path resolution problems between the host (Windows) where runs were logged and the container (Linux). The final successful approach involved:
                1.  Ensuring the host `mlruns` directory was mounted into the container at `/app/mlruns`.
                2.  Setting the `MLFLOW_TRACKING_URI` environment variable inside the Docker container to `file:///app/mlruns`.
                3.  In `prediction_service.py`, manually constructing the full path to the artifacts *inside the container* based on the `MLFLOW_TRACKING_URI`, Experiment ID (hardcoded after discovery), `BEST_MODEL_RUN_ID`, and known artifact names (e.g., `scaler.joblib`, `tuned_xgboost_model/`).
                4.  Loading the scaler using `joblib.load()` from its constructed local path.
                5.  Loading the ML model using `mlflow.xgboost.load_model()` (or the appropriate flavor) from its constructed local directory path.
            * `MODEL_VERSION` was set to the `BEST_MLFLOW_RUN_ID`.
            * `FEATURE_NAMES` were extracted from the loaded scaler (`scaler.feature_names_in_`) to ensure incoming data is processed with the correct features in the correct order.
    * **Preprocessing Input (`services/prediction_service.py`):**
        * The `preprocess_input()` function takes the list of dictionaries (validated by Pydantic in the controller).
        * Converts it to a Pandas DataFrame.
        * **Feature Alignment:** Selects and reorders columns based on `FEATURE_NAMES` to match exactly what the model was trained on. Raises an error if expected features are missing.
        * Fills any remaining NaNs (e.g., with 0, as done in Phase 3).
        * Applies the loaded `SCALER` to transform the features.
    * **Prediction (`services/prediction_service.py`):**
        * The `predict_hazard()` function takes the scaled features.
        * Calls `MODEL.predict()` to get binary (0/1) predictions.
        * Calls `MODEL.predict_proba()[:, 1]` to get the probability of the positive class (hazardous).
    * **API Endpoint (`controllers/prediction_controller.py` - ID: `api_controller_v3_fix_nameerror`):**
        * A Flask `Blueprint` (`prediction_bp`) was used to define the `/predict` route, accepting `POST` requests.
        * **Request Handling:**
            1.  Gets JSON data from the request.
            2.  Validates it using `PredictionInput(**input_data)`. If validation fails, returns a 400 error with Pydantic's error details.
            3.  Calls the service functions (`preprocess_input`, `predict_hazard`).
            4.  Formats the predictions into the `PredictionResponse` schema.
            5.  Returns the JSON response with a 200 OK status.
        * **Error Handling:** Includes `try-except` blocks to catch `ValidationError`, `ValueError`, `RuntimeError` (e.g., if model isn't loaded), and generic `Exception` to return appropriate JSON error responses and HTTP status codes (400 for client errors, 500/503 for server errors).
        * **Logging:** Uses `current_app.logger` for request logging and error logging.
    * **Flask App Initialization (`app.py` - ID: `api_app_v2_metrics`):**
        * `create_app()` factory pattern.
        * Initializes Flask (`app = Flask(__name__)`).
        * Initializes `PrometheusMetrics(app)` for exposing metrics (Phase 9 preparation).
        * Registers the `prediction_bp` blueprint.
        * Includes a basic root route `/` for health checks.
        * The `if __name__ == '__main__':` block runs the Flask development server on `0.0.0.0` (making it accessible externally) and port `5001`.

**4. Technical Insights & Decisions:**
* **Framework Choice (Flask):** Flask is lightweight, flexible, and well-suited for building microservices like this prediction API. Its simplicity allows for quick development.
* **Separation of Concerns:**
    * **Controller:** Handles HTTP concerns (request parsing, response formatting, routing).
    * **Service:** Contains the core ML prediction logic (data transformation, model inference). This makes the code more modular, testable, and easier to maintain, directly addressing project feedback.
    * **Schemas:** Pydantic handles data validation cleanly and declaratively.
* **Model Loading Strategy:** Loading the model and scaler once at service startup (when the module is first imported) is crucial for performance, avoiding the overhead of loading from disk on every prediction request. The challenges with MLflow path resolution in Docker highlighted the importance of understanding how tools interact with different filesystems and environments. The direct path construction (using the known structure of the mounted `mlruns` volume) was a pragmatic solution for the local Docker setup. For a production system, a dedicated MLflow Tracking Server or Model Registry would be preferred for fetching models.
* **Input Validation:** Using Pydantic provides robust, automatic validation of the request payload, ensuring the service layer receives data in the expected format and preventing many potential runtime errors.
* **Error Handling:** Implementing proper `try-except` blocks and returning meaningful JSON error responses and HTTP status codes makes the API more robust and easier for clients to integrate with.
* **Low Latency Consideration:** While not explicitly optimized with advanced techniques in this phase, the design (loading model once, using efficient libraries like NumPy/Pandas for preprocessing, relatively simple model inference) aims for reasonable latency. Further optimizations (e.g., ONNX runtime, more optimized preprocessing) would be considered if the <50ms target wasn't met under load.

**5. Key Commands/Code Snippets:**
* **Installing Dependencies:**
    ```powershell
    pip install Flask pydantic joblib mlflow xgboost scikit-learn pandas numpy prometheus-flask-exporter
    ```
* **Running the Flask API Locally:**
    1.  Set the environment variable for the chosen model:
        *(PowerShell)* `$env:BEST_MLFLOW_RUN_ID = "your_run_id"`
    2.  Execute the app:
        ```powershell
        python src/api/app.py
        ```
* **Example Pydantic Schema (`prediction_schema.py`):**
    ```python
    from pydantic import BaseModel, Field
    from typing import List

    class ModuleFeatures(BaseModel):
        Module_Avg_Surface_Temp_C: float
        # ... all 30 features ...
    
    class PredictionInput(BaseModel):
        readings: List[ModuleFeatures]
    ```
* **Example Flask Route (`prediction_controller.py`):**
    ```python
    from flask import Blueprint, request, jsonify
    prediction_bp = Blueprint('prediction', __name__, url_prefix='/predict')
    @prediction_bp.route('', methods=['POST'])
    def predict():
        # ... get data, validate, call service, format response ...
        return jsonify(response_data.model_dump()), 200
    ```
* **Example Service Logic (`prediction_service.py`):**
    ```python
    # Global MODEL, SCALER
    def load_model_and_scaler(): # Loads from MLflow artifacts
        # ...
    def preprocess_input(data): # Uses SCALER
        # ...
    def predict_hazard(features): # Uses MODEL
        # ...
    ```

**6. Outcome of Phase 6:**
* A functional Flask API with a `/predict` endpoint.
* The API successfully loads a specified trained model and scaler from MLflow run artifacts.
* Incoming prediction requests are validated against a Pydantic schema.
* The API preprocesses input data, makes predictions, and returns them in a structured JSON format.
* A clear separation of concerns between controllers, services, and schemas, promoting maintainability.
* The API is ready to be containerized in the next phase.

This phase is a major step towards making the machine learning model operational and accessible to other systems.

---