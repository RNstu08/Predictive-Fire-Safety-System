
---

### Phase 11: Data Drift Detection & Retraining Strategy (Conceptual)

**1. Goal of this Phase:**
To design a robust strategy for:
* **Detecting Data Drift:** Identifying when the statistical properties of incoming live data (used for predictions) significantly deviate from the data the model was trained on.
* **Detecting Concept Drift (Indirectly):** Identifying when the underlying relationship between input features and the likelihood of a hazard changes.
* **Triggering Retraining:** Defining clear conditions under which the model should be retrained with new or more relevant data.
* **Automated Retraining Pipeline:** Conceptualizing an automated workflow to retrain, evaluate, validate, and potentially redeploy the model.

This phase directly addresses your project description point: *"Designed data drift detection mechanisms (KS-test) and conceptualized automated retraining pipelines triggered by performance degradation or significant drift, ensuring continuous model relevance and improvement."*

**2. Rationale (Why this phase is critical for long-term success):**
Machine learning models are not "train once, deploy forever" solutions, especially in dynamic environments like battery systems.
* **Model Degradation (Staleness):** Models trained on historical data can become less accurate over time if the characteristics of the live data they encounter change. This is known as model staleness or performance degradation.
* **Changing World:**
    * **Data Drift:** The statistical distribution of input features can change due to various factors:
        * Sensor aging or calibration drift.
        * Changes in battery usage patterns (e.g., different charging/discharging cycles).
        * Seasonal variations in ambient temperature affecting battery temperatures.
        * Introduction of new battery types or configurations into the warehouse.
    * **Concept Drift:** The fundamental relationship between the sensor readings and the actual likelihood of a hazardous event might evolve. For example:
        * New, previously unseen failure modes might emerge in the batteries.
        * Changes in maintenance practices might alter how precursors manifest.
        * The definition of what constitutes an "actionable hazard" might change.
* **Maintaining Trust & Reliability:** A model that makes increasingly inaccurate predictions loses its value and can even be detrimental (e.g., missing real hazards or causing too many false alarms). Continuous monitoring and retraining are essential to maintain trust and reliability.
* **MLOps Principle:** This phase is a core component of a mature MLOps practice, focusing on the "Operate" and "Monitor" stages of the ML lifecycle and enabling continuous improvement.

**3. Approach & Key Activities (Conceptual Design):**

* **A. Understanding Drift Types:**
    * **Data Drift (Covariate Shift):** The distribution of input features `P(X)` changes. The model was trained on `P_train(X)` but now sees `P_live(X)`, where `P_live(X) != P_train(X)`. The relationship `P(Y|X)` might still be the same.
    * **Concept Drift:** The conditional probability `P(Y|X)` changes. The meaning of the features in relation to the outcome shifts. This is often harder to detect directly without fresh labels.

* **B. Designing Data Drift Detection Mechanisms:**
    The goal is to compare a "reference" dataset (e.g., the training data, or a stable recent period of production data) with a "current" dataset (recent live data used for predictions).
    1.  **Monitoring Summary Statistics:**
        * **What:** Track basic statistics (mean, median, min, max, standard deviation, percentage of missing values, number of unique values for categorical features) for each input feature on incoming batches of data (e.g., collected daily or weekly).
        * **How:** Compare these live statistics against the statistics from the reference dataset. Significant deviations trigger alerts.
        * **Tools:** Can be implemented with custom scripts, or monitoring systems that can process logged feature data.
    2.  **Statistical Distribution Comparison Tests:**
        * **Kolmogorov-Smirnov (KS) Test (2-sample):** (Mentioned in your project description). For continuous numerical features, this test compares the cumulative distribution functions (CDFs) of the feature in the reference dataset versus the current dataset. The test statistic quantifies the maximum difference between the two CDFs. A small p-value (e.g., < 0.05) suggests the distributions are significantly different.
            * **Code Snippet (Conceptual):**
              ```python
              from scipy.stats import ks_2samp
              # feature_data_reference = ... (e.g., from training set)
              # feature_data_current = ... (e.g., from recent live data)
              # ks_statistic, p_value = ks_2samp(feature_data_reference, feature_data_current)
              # if p_value < 0.05: print(f"Drift detected for feature (p={p_value:.3f})")
              ```
        * **Population Stability Index (PSI):** Compares the distribution of a variable across predefined bins (e.g., deciles) between two datasets. It provides a single numerical score indicating the magnitude of the shift.
            * Rules of thumb for PSI: < 0.1 (no significant shift), 0.1-0.2 (moderate shift, monitor), > 0.2 (significant shift, action needed).
        * **Chi-Squared Test:** Can be used for categorical features to test if the distribution of categories has changed significantly.
    3.  **Adversarial Validation:** Train a classifier to distinguish between the reference dataset and the current dataset. If the classifier can do this with high accuracy, it means the datasets are systematically different, indicating drift. The features most important to this classifier are often the ones that have drifted the most.
    4.  **Monitoring Prediction Probabilities:** Track the distribution of the probabilities output by your deployed model. A significant shift in this distribution (e.g., model suddenly becomes much more or less confident, or predicts a different ratio of classes) can indicate that the input data has changed in a way that affects the model, or that concept drift is occurring.

* **C. Detecting Concept Drift (Often Indirectly):**
    * Direct detection is hard without new ground truth labels for recent data.
    * **Primary Method: Performance Monitoring:** This is where Phase 9 (Monitoring & Alerting) plays a crucial role. Continuously track key model performance metrics (Recall for class 1, Precision for class 1, F1-score, AUC-PR) on any newly labeled data that becomes available (even if it's a small, sampled set). A sustained drop in these metrics is a strong indicator of concept drift (or severe data drift that the model can't handle).
    * **Error Analysis:** Regularly analyze the model's prediction errors (False Positives, False Negatives) on recent data. Are there new types of errors appearing? This can provide qualitative insights into concept drift.

* **D. Defining Retraining Triggers:**
    Clear, pre-defined triggers are needed to initiate the retraining pipeline.
    1.  **Data Drift Thresholds:**
        * For KS-test: p-value consistently below a threshold (e.g., < 0.01 or < 0.05) for several key features or a certain percentage of features.
        * For PSI: PSI value exceeding a threshold (e.g., > 0.2) for critical features.
        * For summary statistics: Change exceeding a certain percentage (e.g., mean changes by > 2 standard deviations of the reference mean).
    2.  **Model Performance Degradation Thresholds:**
        * Recall for class 1 drops below a critical value (e.g., < 0.85, if target was 0.92).
        * PR-AUC drops by more than a certain percentage from its baseline.
        * False Negative rate increases significantly.
    3.  **Scheduled Retraining:** A fixed schedule (e.g., monthly, quarterly) to ensure the model is refreshed with recent data, even if no explicit triggers are hit. This can help proactively address gradual drift.
    4.  **Manual Trigger:** An option for data scientists/ML engineers to initiate retraining based on new insights, significant new data availability, or business needs.

* **E. Conceptualizing an Automated Retraining Pipeline:**
    This pipeline would be orchestrated by a workflow management tool (e.g., Airflow, Kubeflow Pipelines, Argo Workflows) and triggered by the conditions above.
    1.  **Trigger:** Detection of drift or performance drop, or scheduled trigger.
    2.  **Data Ingestion & Preparation:**
        * Collect new data since the last training run (or a relevant window of recent data). This data needs to be sourced from where the live prediction requests and their features are logged.
        * **Labeling New Data:** This is often the most challenging step for automation.
            * If ground truth labels become available for recent data (e.g., through human review of incidents or alerts), these should be incorporated.
            * If not, strategies might include using proxy labels, semi-supervised learning, or focusing retraining on data drift adaptation without new labels for the target (more advanced). For our HVS system, actual confirmed hazardous events would be rare, so any new labeled hazardous instance is highly valuable.
        * Combine new data with a relevant portion of historical data (to avoid catastrophic forgetting).
        * Run the **Preprocessing and Feature Engineering Pipeline** (from `src/preprocessing/pipelines.py`) on this new combined dataset.
    3.  **Model Retraining:**
        * Execute the training script (e.g., `src/training/train_mlflow.py` or `tune_model.py` if re-tuning is part of the pipeline).
        * **Log to MLflow:** Every retraining attempt should be a new run in MLflow, logging parameters, metrics, and the new model artifact. This allows comparison with previously deployed models.
    4.  **Model Evaluation & Validation (Shadow Mode or A/B Testing):**
        * Evaluate the retrained model on a held-out validation set (derived from the new data).
        * **Compare with Current Production Model:** Metrics (Recall, Precision, F1, PR-AUC) of the new model are compared against the currently deployed model (whose performance might have degraded).
        * **Business Logic Validation:** Does the new model meet minimum safety and performance criteria?
        * **Shadow Deployment (Optional):** Deploy the new model alongside the old one, feeding it live traffic but not using its predictions for actual actions. Compare its predictions to the old model's and to any ground truth that becomes available.
        * **A/B Testing (Optional):** Route a small percentage of live traffic to the new model and compare its performance directly.
    5.  **Model Deployment Decision:**
        * **Automated:** If the new model is demonstrably better (e.g., significantly higher Recall on validation data without too much loss in Precision) and passes all checks, the pipeline could automatically promote it.
        * **Human-in-the-Loop:** More commonly, a data scientist or ML engineer reviews the validation results in MLflow and gives manual approval for deployment.
    6.  **Deployment:**
        * If approved, the CI/CD pipeline (Phase 10) is triggered.
        * The `BEST_MLFLOW_RUN_ID` (or a similar configuration pointing to the new model version/URI from the MLflow Model Registry) is updated.
        * The CI/CD pipeline builds/configures the API with the new model and deploys it to Kubernetes.

**4. Technical Insights & Decisions:**
* **Choosing Drift Detection Metrics:** The choice depends on the data type and sensitivity required. KS-test is good for continuous data, PSI for binned data. Monitoring basic stats is a good first line of defense.
* **Thresholds for Drift/Retraining:** Setting appropriate thresholds for drift detection and performance degradation requires experimentation and domain knowledge. Too sensitive, and you'll retrain too often; too insensitive, and the model will perform poorly for too long.
* **Importance of Labeled Data for Retraining:** The quality and availability of fresh, accurately labeled data are paramount for effective retraining, especially for concept drift adaptation. This is often a major operational challenge.
* **MLflow's Role:** MLflow is central to this process for tracking all retraining experiments, versioning the retrained models, comparing their performance against production models, and potentially managing the lifecycle of models through its Model Registry (promoting from staging to production).
* **Automation Tools:** Workflow orchestrators are key for automating the multi-step retraining pipeline.

**5. Key Commands/Code Snippets (Conceptual):**
* **KS-Test (Python):**
    ```python
    from scipy.stats import ks_2samp
    # ks_statistic, p_value = ks_2samp(training_data_feature, live_data_feature)
    # if p_value < 0.05: # Drift detected
    ```
* **MLflow - Getting Run Info (Python, for comparing models):**
    ```python
    # client = mlflow.tracking.MlflowClient()
    # production_model_run = client.get_run("RUN_ID_OF_PROD_MODEL")
    # new_model_run = client.get_run("RUN_ID_OF_NEW_MODEL")
    # prod_recall = production_model_run.data.metrics.get("recall_class1")
    # new_recall = new_model_run.data.metrics.get("recall_class1")
    # if new_recall > prod_recall: # Deploy new model (simplified logic)
    ```

**6. Outcome of Phase 11 (Conceptual):**
* A well-defined strategy for monitoring data and model performance in production.
* Clear criteria (triggers) for when model retraining is necessary.
* A conceptual design for an automated retraining pipeline that includes data collection, labeling considerations, model retraining, validation, and deployment steps, leveraging MLflow for tracking and versioning.
* An understanding of the challenges involved, particularly around data labeling and setting appropriate drift/performance thresholds.

This phase ensures that the HVS prediction system is not just a one-time deployment but a dynamic system designed for continuous improvement and long-term reliability.

---