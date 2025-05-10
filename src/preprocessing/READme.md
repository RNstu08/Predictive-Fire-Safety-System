# Phase 3: Data Preprocessing and Feature Engineering

## Recap

We've completed Phase 1 (Data Generation - using the dummy data script) and Phase 2 (EDA - analyzing the data). We should now have a file like `dummy_fault_RackA2_Module01_200rows.csv` in our `data/raw` folder, and our EDA notebook (`notebooks/01_EDA_Scenario_Analysis.ipynb`) should show that this data contains both normal (0) and hazardous (1) labels, along with injected anomalies and imperfections.

## Goal of Phase 3

To take the raw, imperfect data and transform it into a clean, informative dataset suitable for training machine learning models. This involves two main parts:

* **Preprocessing:** Cleaning the data by handling missing values and potential outliers.
* **Feature Engineering:** Creating new features from the existing sensor data that might capture patterns more effectively (like rates of change or rolling statistics).

## Why is this important?

* **Data Quality:** Machine learning models are sensitive to missing data and extreme outliers. Cleaning the data improves model stability and performance.
* **Information Extraction:** Raw sensor readings might not always be the most informative representation. Engineered features can highlight trends, volatility, or rates of change that are crucial for prediction (especially for time-series anomaly detection like ours). For example, the rate at which temperature increases is often a much stronger indicator of thermal runaway than the absolute temperature alone.

## Steps in Phase 3:

1.  **Load Raw Data:** Read the CSV file.
2.  **Handle Timestamp:** Ensure the 'Timestamp' column is parsed correctly as datetime objects and potentially set it as the index for time-series operations.
3.  **Handle Missing Values:** Apply an imputation strategy.
4.  **Handle Outliers (Optional but Recommended):** Apply a strategy to manage extreme values.
5.  **Feature Engineering:** Create new features based on domain knowledge and EDA insights.
6.  **Select Features (Initial):** Choose the columns (original and engineered) to keep for modeling.
7.  **Save Processed Data:** Store the cleaned and engineered data in the `data/processed/` directory.

We will implement this logic primarily within the `src/preprocessing/pipelines.py` file, creating functions that can be called later.

## Explanation of the Code (`src/preprocessing/pipelines.py`):

* **Configuration:** Defines lists of expected columns and columns containing sensor readings that need processing.
* **`load_data(file_path)`:** Loads the CSV, checks for existence, and performs a basic validation for expected columns.
* **`handle_timestamp(df)`:** Converts the 'Timestamp' column to datetime objects and sorts the DataFrame by `Module_ID` and `Timestamp`. Sorting is crucial for time-series operations like forward fill and calculating differences/rolling windows correctly within each module's sequence. (Note: Timestamp is kept as a regular column to simplify grouping operations later).
* **`impute_missing_values(df, ...)`:** Takes the DataFrame, columns to process, and an imputation method.
    * `method='ffill'` (Forward Fill): This is chosen as the default. It performs the ffill within each `Module_ID` group to prevent filling across different modules. It also includes a `bfill` (backward fill) to handle any NaNs remaining at the very beginning of a module's data.
    * Other methods (mean, median) are included as options, calculating the statistic per module.
    * A final check fills any remaining NaNs (e.g., if an entire module had missing data) with 0.
    * Prints which columns had values imputed.
* **`handle_outliers(df, ...)`:** Takes the DataFrame, columns to process, and quantile thresholds.
    * **Clipping/Winsorization:** It calculates the lower (e.g., 1st percentile) and upper (e.g., 99th percentile) boundaries for each module individually. Any value below the lower boundary is replaced with the lower boundary value, and any value above the upper boundary is replaced with the upper boundary value.
    * This is generally preferred over removing outliers in time-series, as removing rows disrupts the sequence. Clipping retains extreme values related to faults but limits the influence of potentially erroneous sensor spikes.
    * It's applied to sensor columns but not the gas proxies, as those are expected to have legitimate large spikes during faults in our dummy data.
* **`calculate_delta_features(df, ...)`:** Calculates the difference between consecutive time steps (window=1) for specified columns (e.g., temperature, voltage).
    * Operates within each `Module_ID` group using `groupby().transform(lambda x: x.diff())`.
    * Creates new columns named like `Delta_1_Module_Avg_Surface_Temp_C`.
    * Fills the initial NaN values (which occur because the first reading has no previous value) with 0.
    * This captures the rate of change, a very important feature for detecting rapid anomalies.
* **`calculate_rolling_features(df, ...)`:** Calculates rolling statistics (mean and standard deviation by default) over specified time windows (e.g., 6 steps = 1 min, 30 steps = 5 min, 60 steps = 10 min).
    * Operates within each `Module_ID` group.
    * Creates new columns like `Rolling_mean_6_Module_Avg_Surface_Temp_C`, `Rolling_std_60_Module_Voltage_V`, etc.
    * Fills initial NaNs with 0.
    * These features help capture trends (rolling mean) and volatility/stability (rolling standard deviation) over different time scales.
* **`run_preprocessing_pipeline(raw_data_path, processed_data_path)`:** This function orchestrates the entire process by calling the steps in sequence: load, handle timestamp, impute, handle outliers, engineer delta features, engineer rolling features.
    * It defines which columns to apply specific steps to.
    * It saves the final processed DataFrame to the specified output path in the `data/processed/` directory.
* **`if __name__ == '__main__':` block:** Allows you to run this script directly from the command line for testing or processing a single file.
    * It defines the input and output file paths relative to the project structure. Make sure `RAW_DATA_FILENAME` matches your dummy data file.
    * Calls `run_preprocessing_pipeline`.
    * Prints info about the processed DataFrame and checks for any remaining NaNs.

## Action for You:

1.  **Save the Code:** Copy the entire code block (from the original source for `src/preprocessing/pipelines.py`) into your `src/preprocessing/pipelines.py` file and save it.
2.  **Run the Script:**
    * Make sure your virtual environment is active.
    * Navigate to your project root directory (`hvs_fire_prediction`) in the terminal.
    * Crucially, ensure the `RAW_DATA_FILENAME` variable inside the `if __name__ == '__main__':` block in `pipelines.py` matches the name of your dummy data CSV file (e.g., `"dummy_fault_RackA2_Module01_200rows.csv"`).
    * Execute the script:
        ```powershell
        python src\preprocessing\pipelines.py
        ```
3.  **Check the Output:**
    * Watch the console output. It should show the steps being performed (Loading, Imputing, Clipping, Calculating Features...).
    * Verify that a new CSV file appears in your `data/processed/` directory (e.g., `processed_dummy_fault_RackA2_Module01_200rows.csv`).
    * Check the final "Check NaNs" output in the console - it should report 0 remaining NaNs.
    * Look at the "Processed DataFrame Head" output. You should see the original columns plus the new `Delta_...` and `Rolling_...` columns.

Once this runs successfully, you will have a processed dataset ready for the next phase: Model Development, Training & Evaluation (Phase 4).