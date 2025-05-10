### Phase 3: Data Preprocessing & Feature Engineering

**1. Goal:**
To clean the raw (dummy) data and transform it into a structured, high-quality dataset ready for machine learning. This involves handling imperfections identified during EDA and creating new, potentially more informative features.

**2. Rationale (Why?):**
* **Data Quality for Models:** Machine learning algorithms are sensitive to the quality of input data. Missing values can cause errors or lead to biased models. Outliers can disproportionately influence model training. Noise can obscure underlying patterns.
* **Enhanced Predictive Power:** Raw sensor data might not always be in the most optimal format for models to learn complex patterns. Feature engineering aims to create new features that explicitly represent relationships or trends (e.g., rate of temperature change) that are highly indicative of the target event (hazardous state). This can significantly improve model performance.
* **Consistency:** Ensures all data fed to the model goes through a standardized preparation process.

**3. Approach & Key Activities:**
This phase was implemented in the script `src/preprocessing/pipelines.py`.
* **Loading Data:** The raw dummy data CSV (e.g., `dummy_fault_RackA2_Module01_200rows.csv`) from `data/raw/` was loaded.
* **Timestamp Handling:** The `Timestamp` column was converted to datetime objects, and the DataFrame was sorted by `Module_ID` and then `Timestamp`. This ordering is crucial for time-series operations like forward fill and calculating rolling window features correctly within each module's individual timeline.
* **Missing Value Imputation:**
    * **Strategy:** For the sensor columns, **forward fill (`ffill`) followed by backward fill (`bfill`) within each `Module_ID` group** was chosen.
    * **Rationale:** In time-series data, especially sensor readings, it's often reasonable to assume that a missing value is likely similar to the last known reading (`ffill`). `bfill` handles any NaNs at the very beginning of a module's series. Grouping by `Module_ID` prevents filling data from one module into another. Any remaining NaNs (e.g., if an entire module's data for a feature was missing, which shouldn't happen with our generator) were filled with 0 as a final catch-all.
* **Outlier Handling:**
    * **Strategy:** **Quantile-based clipping (Winsorization)** was applied to most sensor columns (excluding gas proxies, which are expected to have legitimate large spikes in the dummy data). Values below the 1st percentile (`lower_quantile=0.01`) and above the 99th percentile (`upper_quantile=0.99`) *for each module individually* were replaced with the 1st or 99th percentile value, respectively.
    * **Rationale:** Clipping helps reduce the influence of extreme outliers without entirely removing the data points (which would disrupt the time series). Performing this per module respects the individual operating characteristics or fault severities.
* **Feature Engineering:**
    * **Delta Features (Rate of Change):**
        * Calculated the difference between the current sensor reading and the reading from one time step prior (`window=1`, which is 10 seconds in our case). This was done for key features like `Module_Avg_Surface_Temp_C`, `Module_Voltage_V`, and `Module_SoC_percent`.
        * New columns were named like `Delta_1_Module_Avg_Surface_Temp_C`.
        * **Rationale:** The rate of change (e.g., how fast temperature is rising) is often a much stronger predictor of thermal runaway than the absolute value alone.
    * **Rolling Window Statistics:**
        * Calculated rolling mean and rolling standard deviation for key features (`Module_Avg_Surface_Temp_C`, `Module_Voltage_V`, `Module_Current_A`).
        * Used multiple window sizes (e.g., 6 steps = 1 minute, 30 steps = 5 minutes, 60 steps = 10 minutes) to capture trends and volatility over different time scales.
        * New columns were named like `Rolling_mean_6_Module_Avg_Surface_Temp_C`, `Rolling_std_60_Module_Voltage_V`.
        * **Rationale:** Rolling means smooth out short-term noise and reveal underlying trends. Rolling standard deviations quantify the recent volatility or stability of a sensor reading.
* **Saving Processed Data:** The resulting DataFrame, now cleaned and augmented with new features, was saved to a new CSV file in the `data/processed/` directory (e.g., `processed_dummy_fault_RackA2_Module01_200rows.csv`).

**4. Technical Insights & Decisions:**
* **Grouped Operations:** A key decision was to perform most preprocessing steps (imputation, outlier clipping) and feature engineering (deltas, rolling stats) *per `Module_ID`*. This is critical because each module is an independent unit, and operations should not bleed across different modules. Pandas' `groupby()` followed by `transform()` or `apply()` was used for this.
* **Order of Operations:** Sorting by `Module_ID` and `Timestamp` before time-dependent operations (`ffill`, `diff`, `rolling`) is essential for correctness.
* **Handling NaNs from Feature Engineering:** Both `diff()` and `rolling()` operations introduce NaNs at the beginning of each group (as there are no prior data points for the window). These were subsequently filled with 0, assuming that at the start of a series, the rate of change or rolling stats are effectively zero or baseline.
* **Choice of Imputation/Outlier Strategy:** `ffill` is a common choice for sensor data. Quantile clipping is a robust way to handle outliers without data removal. These are reasonable starting points; more sophisticated methods could be explored if needed.
* **Script vs. Notebook:** While EDA is interactive in a notebook, the preprocessing pipeline was implemented as a Python script (`pipelines.py`) to make it reusable and executable as part of a larger data pipeline (e.g., in a CI/CD workflow or an orchestration tool).

**5. Key Commands/Code Snippets (Illustrative from `pipelines.py`):**
* **Running the Preprocessing Script:**
    ```powershell
    # Ensure RAW_DATA_FILENAME in the script's main block is correct
    python src/preprocessing/pipelines.py
    ```
* **Grouped Forward Fill (Conceptual):**
    ```python
    df[columns_to_impute] = df.groupby('Module_ID')[columns_to_impute].ffill()
    df[columns_to_impute] = df.groupby('Module_ID')[columns_to_impute].bfill() # Handle initial NaNs
    ```
* **Grouped Delta Feature (Conceptual):**
    ```python
    df[new_delta_col_name] = df.groupby('Module_ID')[col].transform(lambda x: x.diff(periods=window))
    ```
* **Grouped Rolling Mean (Conceptual):**
    ```python
    df[new_rolling_col_name] = df.groupby('Module_ID')[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    ```

**6. Outcome:**
* A cleaned and feature-enriched dataset saved in `data/processed/`.
* This processed dataset has missing values handled, outliers managed, and new, potentially more predictive features (deltas and rolling statistics).
* It is now in a much better state for feeding into machine learning models in the next phase.
* A reusable preprocessing script (`src/preprocessing/pipelines.py`).

This phase transforms the raw data into a high-quality input for model training, directly impacting the potential performance and robustness of the final predictive system.

---

### Phase 4: Baseline Model Development, Training & Evaluation

**1. Goal:**
To train several standard classification models on the preprocessed data, evaluate their baseline performance without extensive hyperparameter tuning, and establish initial benchmarks. This phase also incorporates strategies to handle the previously identified class imbalance.

**2. Rationale (Why?):**
* **Establish Baselines:** Before spending significant time on complex models or extensive tuning, it's crucial to know how well simpler, standard models perform. This provides a reference point. If a complex model doesn't significantly outperform a simpler one, the added complexity might not be justified.
* **Initial Imbalance Handling:** Given the severe class imbalance (few hazardous samples), applying basic techniques to address this from the outset is important for getting meaningful initial results, especially for the minority class.
* **Model Candidate Selection:** The performance of these baseline models helps in selecting one or two promising candidates for more intensive optimization (like hyperparameter tuning) in later stages.
* **Proof of Concept:** Demonstrates that the engineered features and processed data can indeed be used to train models that have some predictive capability for the hazardous state.

**3. Approach & Key Activities:**
This phase was implemented in a script like `src/training/train.py` (which later evolved into `train_mlflow.py`).
* **Load Processed Data:** Loaded the output CSV from Phase 3 (e.g., `processed_dummy_fault_RackA2_Module01_200rows.csv`).
* **Feature and Target Selection (X, y):**
    * The `Hazard_Label` column was selected as the target variable `y`.
    * All other relevant numerical columns (original sensor readings + engineered delta and rolling features) were selected as input features `X`. Identifier columns (`Timestamp`, `Module_ID`, `Rack_ID`) were excluded.
* **Train/Test Split:**
    * The data was split into training and testing sets (e.g., 80% train, 20% test).
    * **Stratified Splitting:** `StratifiedShuffleSplit` (or `train_test_split` with `stratify=y`) was used. This ensures that the proportion of hazardous (1) and normal (0) samples is maintained in both the training and test sets, which is critical for reliable evaluation on imbalanced data.
* **Data Scaling:**
    * `StandardScaler` from `scikit-learn` was used.
    * The scaler was **fit only on the training data (`X_train`)** to learn the mean and standard deviation of each feature.
    * Then, this *same fitted scaler* was used to **transform both the training data (`X_train`) and the test data (`X_test`)**. This prevents data leakage from the test set into the training process. Scaling is particularly important for models like Logistic Regression.
* **Imbalance Handling (Class Weighting):**
    * This was the primary strategy for handling imbalance at this stage.
    * For `LogisticRegression` and `RandomForestClassifier`, the `class_weight='balanced'` parameter was used. This mode automatically adjusts weights inversely proportional to class frequencies in the input data (i.e., giving a higher weight to the minority class).
    * For `XGBClassifier`, `scale_pos_weight` was calculated as `count(negative_class_samples_in_train) / count(positive_class_samples_in_train)` and passed to the model. This tells XGBoost to give more weight to the positive (hazardous) class.
* **Model Training:**
    * Instantiated three common classification models:
        1.  `LogisticRegression` (a good linear baseline).
        2.  `RandomForestClassifier` (an ensemble of decision trees).
        3.  `XGBClassifier` (a powerful gradient boosting algorithm).
    * Each model was trained (using its `.fit()` method) on the scaled training data (`X_train_scaled`, `y_train`).
* **Prediction & Evaluation:**
    * For each trained model, predictions were made on the scaled *test data* (`X_test_scaled`).
    * `model.predict()` was used to get binary class predictions (0 or 1).
    * `model.predict_proba()` was used to get probability scores for each class (we are interested in the probability of class 1).
    * **Evaluation Metrics:**
        * **Accuracy:** Overall correctness (often misleading for imbalanced data).
        * **Confusion Matrix:** A table showing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). Visualized using `ConfusionMatrixDisplay`.
        * **Classification Report:** Provides Precision, Recall, and F1-score *for each class*, as well as macro and weighted averages. **Recall for class 1 (hazardous)** and **Precision for class 1** were key metrics.
        * **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to discriminate between positive and negative classes across all classification thresholds.
        * **AUC-PR (Area Under the Precision-Recall Curve):** More informative than AUC-ROC for highly imbalanced datasets, as it focuses on the performance of the positive (minority) class.

**4. Technical Insights & Decisions:**
* **Importance of Stratified Splitting:** Ensures that the rare hazardous class is represented in the test set, allowing for a meaningful evaluation of how well the model predicts it.
* **Fitting Scaler on Training Data Only:** This is a fundamental best practice to prevent data leakage from the test set, which would lead to overly optimistic performance estimates.
* **Choice of Imbalance Strategy (Class Weighting):** Class weighting is often a good first approach as it's directly supported by many model implementations and is less complex to set up than data resampling techniques like SMOTE.
* **Focus on Relevant Metrics:** For this problem (predicting rare hazardous events), overall accuracy is not the primary concern. High **Recall for class 1** (minimizing missed hazards) is paramount, followed by reasonable **Precision for class 1** (minimizing false alarms). AUC-PR is also a very good summary metric.
* **Model Selection Considerations:** XGBoost is often a strong performer on tabular data and handles imbalances relatively well with `scale_pos_weight`. Random Forest is also robust. Logistic Regression provides a good linear baseline.

**5. Key Commands/Code Snippets (Illustrative from `train.py` / `train_mlflow.py`):**
* **Stratified Split:**
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ```
* **Scaling:**
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
* **Class Weighting Examples:**
    ```python
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(class_weight='balanced', solver='liblinear')

    from xgboost import XGBClassifier
    # neg_count = sum(y_train == 0); pos_count = sum(y_train == 1)
    # scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1
    xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight_val)
    ```
* **Evaluation:**
    ```python
    from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of positive class
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba)}")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    print(f"AUC-PR: {auc(recall, precision)}")
    ```

**6. Outcome:**
* Baseline performance metrics established for Logistic Regression, Random Forest, and XGBoost.
* An initial understanding of how well each model type can identify the hazardous class given the current features and class weighting strategy.
* A script (`src/training/train.py` or `src/training/train_mlflow.py`) for reproducible training and evaluation.
* Identification of which model(s) are good candidates for further optimization (hyperparameter tuning) or for experiments with other imbalance handling techniques (like SMOTE).

This phase transitions us from data preparation to actual model building and provides the first quantitative assessment of our predictive capabilities.

---
