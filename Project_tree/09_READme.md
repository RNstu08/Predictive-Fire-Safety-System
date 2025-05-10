
---

### Phase 9: API Instrumentation for Monitoring & Conceptual Monitoring Setup (Prometheus & Grafana)

**1. Goal of this Phase:**
To instrument our Flask API to expose key performance indicators (KPIs) and custom application metrics in a format suitable for Prometheus, a leading open-source monitoring system. We will then conceptually cover how Prometheus and Grafana would be deployed and configured in our Kubernetes environment to collect, visualize, and alert on these metrics.

**2. Rationale (Why this phase is critical):**
Running an application without monitoring is like driving a car without a dashboard – you don't know your speed, fuel level, or if the engine is overheating until it's too late.
* **Operational Visibility:** Monitoring provides real-time insights into the API's behavior (latency, throughput, error rates) and resource usage.
* **Performance Optimization:** By tracking metrics, we can identify performance bottlenecks and areas for optimization.
* **Proactive Issue Detection:** Setting up alerts based on metrics (e.g., high error rate, high latency, low prediction throughput) allows us to detect and address problems often before users are impacted.
* **Resource Management:** Understanding CPU and memory usage helps in right-sizing our Kubernetes deployments.
* **Basic Model Monitoring:** Tracking the distribution of predictions (Normal vs. Hazardous) can give early warnings about potential issues with incoming data or model drift.
* **Debugging:** Historical metric data is invaluable for troubleshooting and post-mortem analysis of incidents.
This phase directly addresses your project goal of *"Established monitoring and alerting with Prometheus and Grafana, visualizing key performance indicators (API latency, request throughput, prediction distributions)..."*

**3. Approach & Key Activities:**

This phase has two main parts:
* **Part 1 (Practical): Instrumenting the Flask API.** This involves code changes to our API.
* **Part 2 (Conceptual): Deploying and Configuring Prometheus & Grafana.** This involves infrastructure setup, which we'll describe conceptually.

**Part 1: Instrumenting the Flask API (Practical Steps)**

* **A. Add `prometheus-flask-exporter` to `requirements.txt`:**
    * This Python library makes it easy to expose Prometheus metrics from a Flask application.
    * **Action:** Ensure your `requirements.txt` includes:
        ```text
        # ... other requirements ...
        prometheus-flask-exporter>=0.18,<0.23
        # ...
        ```
    * If you modify `requirements.txt`, reinstall dependencies in your virtual environment:
        ```powershell
        pip install -r requirements.txt
        ```

* **B. Initialize `PrometheusMetrics` in `src/api/app.py`:**
    * This integrates the exporter with your Flask application.
    * **Code (as in `api_app_v2_metrics`):**
        ```python
        # src/api/app.py
        from flask import Flask
        from prometheus_flask_exporter import PrometheusMetrics # Import
        # ... other imports ...

        def create_app():
            app = Flask(__name__)
            metrics = PrometheusMetrics(app) # Initialize metrics exporter
            metrics.info('app_info', 'HVS Prediction API Information', version='1.0.0') # Add static app info
            # ... rest of app factory ...
            return app
        ```
    * This automatically creates a `/metrics` endpoint on your API and starts tracking standard HTTP request metrics (count, latency, status codes) per endpoint.

* **C. Add Custom Metrics (e.g., Prediction Outcomes) in `src/api/controllers/prediction_controller.py`:**
    * While `prometheus-flask-exporter` gives many default metrics, we often want to track application-specific events, like the distribution of our model's predictions.
    * **Code (as in `api_controller_v3_fix_nameerror`):**
        ```python
        # src/api/controllers/prediction_controller.py
        from prometheus_client import Counter # Import Counter
        # ... other imports ...

        # Define the custom counter (global or within app context if preferred)
        PREDICTION_OUTCOME_COUNTER = Counter(
            'hvs_api_prediction_outcomes_total', # Metric name
            'Total number of hazard predictions made by the API', # Description
            ['predicted_label_text'] # Label to distinguish 'Normal' vs 'Hazardous'
        )

        # ... inside the predict() function, after getting predictions_numeric ...
        for i, single_prediction_value in enumerate(predictions_numeric):
            predicted_label_text = "Hazardous" if single_prediction_value == 1 else "Normal"
            PREDICTION_OUTCOME_COUNTER.labels(predicted_label_text=predicted_label_text).inc() # Increment counter
            # ... rest of loop ...
        ```
    * This counter, `hvs_api_prediction_outcomes_total`, will track how many "Normal" and "Hazardous" predictions are made.

* **D. Rebuild Docker Image:**
    * Since we've changed Python code (`app.py`, `prediction_controller.py`) and potentially `requirements.txt`, the Docker image must be rebuilt to include these changes.
    * **Command (from project root):**
        ```powershell
        docker build -t hvs-prediction-api:latest .
        ```

* **E. Test `/metrics` Endpoint Locally with Docker (Before Kubernetes):**
    * This is to verify the instrumentation works correctly in the containerized environment.
    * **Run the new image:**
        ```powershell
        # Stop and remove old container if it exists
        docker stop hvs-api
        docker rm hvs-api
        # Run the new image (replace YOUR_SELECTED_RUN_ID)
        docker run -p 5001:5001 -e BEST_MLFLOW_RUN_ID="YOUR_SELECTED_RUN_ID" -v ${PWD}/mlruns:/app/mlruns --name hvs-api hvs-prediction-api:latest
        ```
    * **Verify:**
        1.  Open your browser to `http://localhost:5001/metrics`. You should see a text page listing various metrics (e.g., `flask_http_requests_total`, `app_info`).
        2.  Send a few POST requests to `http://localhost:5001/predict` using `Invoke-RestMethod` with your `test_payload.json` and `test_payload_hazardous.json`.
        3.  Refresh the `http://localhost:5001/metrics` page. You should see the values for `flask_http_requests_total{endpoint="/predict", method="POST"}` increase. You should also see your custom `hvs_api_prediction_outcomes_total{predicted_label_text="Normal"}` and `hvs_api_prediction_outcomes_total{predicted_label_text="Hazardous"}` counters with their respective counts.
    * **Outcome:** This confirms the API correctly exposes metrics.

**Part 2: Deploying and Configuring Prometheus & Grafana on Kubernetes (Conceptual Steps)**

After confirming your API exposes metrics correctly within a Docker container, the next step in a full MLOps setup would be to deploy Prometheus and Grafana to your Kubernetes cluster (e.g., Docker Desktop K8s) and configure them.

* **A. Deploy Prometheus & Grafana:**
    * **Method:** Typically using Helm (a Kubernetes package manager) with the `kube-prometheus-stack` chart. This chart bundles Prometheus (for metrics collection), Grafana (for visualization), Alertmanager (for handling alerts), and various exporters for Kubernetes cluster metrics.
    * **Conceptual Commands (after installing Helm):**
        ```powershell
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        kubectl create namespace monitoring # If it doesn't exist
        helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring
        ```
    * **Verification:** `kubectl get pods -n monitoring` should show all components running. This step can be resource-intensive on a local machine.

* **B. Configure Prometheus to Scrape Your API Metrics:**
    * **Method:** The `kube-prometheus-stack` typically uses a Custom Resource Definition (CRD) called `ServiceMonitor`. You create a `ServiceMonitor` YAML file that tells Prometheus which Kubernetes Services to scrape metrics from.
    * **Conceptual `k8s/hvs-api-servicemonitor.yaml`:**
        ```yaml
        apiVersion: monitoring.coreos.com/v1
        kind: ServiceMonitor
        metadata:
          name: hvs-api-monitor
          namespace: monitoring # Where Prometheus operator looks for ServiceMonitors
          labels:
            release: prometheus # Standard label for kube-prometheus-stack
        spec:
          selector: # Selects the Service to scrape
            matchLabels:
              app: hvs-api # Must match label on your hvs-api-service
          namespaceSelector: # Namespace of your hvs-api-service
            matchNames:
            - default
          endpoints:
          - port: http-api # Name of the port in your hvs-api-service (must match)
            path: /metrics   # Path to scrape metrics from
            interval: 15s    # Scrape interval
        ```
    * **Apply:** `kubectl apply -f k8s/hvs-api-servicemonitor.yaml`
    * **Verify in Prometheus UI:** Port-forward to the Prometheus service (`kubectl port-forward svc/prometheus-operated 9090:9090 -n monitoring`), open `http://localhost:9090`, go to `Status > Targets`. You should see your API pods listed as targets and their state as `UP`.

* **C. Visualize Metrics in Grafana:**
    * **Access Grafana:** Port-forward to the Grafana service (`kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring`), open `http://localhost:3000`. Login (default usually `admin`/`prom-operator` for this stack).
    * **Add Prometheus Data Source:** If not auto-configured by the Helm chart, add Prometheus as a data source, pointing to the internal Kubernetes DNS name of the Prometheus service (e.g., `http://prometheus-operated.monitoring.svc.cluster.local:9090`).
    * **Create Dashboards:** Build dashboards with panels using PromQL queries to visualize:
        * API Request Rate (e.g., `sum(rate(flask_http_request_total{job="<your-job-label>", endpoint="/predict"}[1m]))`)
        * API Latency (e.g., `histogram_quantile(0.95, sum(rate(flask_http_request_duration_seconds_bucket{job="<your-job-label>", endpoint="/predict"}[5m])) by (le))`)
        * Error Rates (e.g., based on `flask_http_request_total{status=~"5.."}`)
        * Prediction Outcomes (e.g., `sum(increase(hvs_api_prediction_outcomes_total{job="<your-job-label>", predicted_label_text="Hazardous"}[5m]))`)
        * Pod CPU/Memory (requires `kube-state-metrics` and `node_exporter`, usually included in the stack).

* **D. Alerting (Conceptual):**
    * Once metrics are in Grafana/Prometheus, you can define alert rules.
    * **Grafana:** Define alerts directly on dashboard panels (e.g., if 95th percentile latency > 500ms for 5 minutes).
    * **Prometheus & Alertmanager:** Define alert rules in Prometheus. Prometheus sends alerts to Alertmanager, which handles routing to email, Slack, etc.

**4. Technical Insights & Decisions:**
* **Choice of Exporter:** `prometheus-flask-exporter` is a convenient library for Flask apps, providing many standard metrics automatically.
* **Custom Metrics:** Defining custom metrics (like `PREDICTION_OUTCOME_COUNTER`) is crucial for tracking application-specific behavior beyond generic HTTP metrics. This helps in basic model performance monitoring (e.g., observing prediction distributions).
* **Service Discovery in Kubernetes:** `ServiceMonitor` CRDs (used by Prometheus Operator) provide a Kubernetes-native way to automatically discover and configure scrape targets, which is much more robust than static IP configurations.
* **PromQL:** Learning PromQL (Prometheus Query Language) is essential for effectively querying and visualizing metrics in Prometheus and Grafana.
* **Resource Intensity:** A full monitoring stack (Prometheus, Grafana, etc.) can be resource-intensive, especially on a local machine with Docker Desktop.

**5. Key Commands/Code Snippets (Illustrative):**
* **Install Prometheus/Grafana Stack via Helm:**
    ```powershell
    helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
    ```
* **Apply ServiceMonitor:**
    ```powershell
    kubectl apply -f k8s/hvs-api-servicemonitor.yaml
    ```
* **Port-forward Grafana:**
    ```powershell
    kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
    ```
* **Example PromQL for RPS:**
    `sum(rate(flask_http_request_total{endpoint="/predict"}[1m]))`

**6. Outcome of Phase 9:**
* **Application Side (Practical):**
    * Flask API instrumented to expose a `/metrics` endpoint.
    * Docker image updated with the instrumented API.
    * Local verification that the `/metrics` endpoint is working and custom metrics are being generated.
* **Infrastructure Side (Conceptual for Full K8s Setup, but you could attempt the Helm install):**
    * Understanding of how Prometheus would be deployed and configured to scrape the API in Kubernetes.
    * Understanding of how Grafana would be used to connect to Prometheus and build dashboards for visualizing key API and model metrics.
    * Conceptual understanding of how alerting could be set up based on these metrics.

This phase establishes the observability for your deployed API, a critical component for maintaining a healthy and performant ML system in production.

---