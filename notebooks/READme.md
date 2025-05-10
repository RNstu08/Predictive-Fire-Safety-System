We'll follow the structure laid out in the "Phase 2: Exploratory Data Analysis (EDA) for HVS Project" document (ID: `eda_hvs_project`).

**Step 1: Setting Up the EDA Environment and Loading Data** (from `01_EDA_Scenario_Analysis.ipynb`)

* **Code:** The first code block in `01_EDA_Scenario_Analysis.ipynb` (lines starting with `# Import necessary libraries...` down to `if not df.empty: print(f"\nDataset shape: {df.shape}")`).
* **Action:**
    1.  Make sure the `DATA_FILE_NAME` variable in the code is set to `"internalfault_aggressive_racka2_module01_isc_aggr_run1.csv"`.
    2.  Ensure `FAILING_MODULE_ID` is `"RackA2_Module01"`. This is critical because Scenario 4 in `simulator.py` was designed to make this specific module fail.
    3.  Run this code block.
* **Expected Output & What to Look For:**
    * `Loading data from: ..\data\raw\internalfault_aggressive_racka2_module01_isc_aggr_run1.csv`
    * `Data loaded successfully.`
    * `Dataset shape: (some_number_of_rows, 14)` (or similar number of columns based on the final `simulator.py`).
        * **Rows:** The number of rows will be `(total_simulation_duration_seconds / time_step_seconds) * number_of_modules`. For Scenario 4 (6 hours, 10s step, 64 modules): `(6*3600 / 10) * 64 = 2160 * 64 = 138,240 rows`. This is a substantial dataset for one scenario!
        * **Columns:** You should have around 14 columns: `Timestamp`, `Module_ID`, `Rack_ID`, various sensor readings (`Module_Avg_Surface_Temp_C`, etc.), simulated gas proxies, and `Hazard_Label`.
    * **Why this matters:** This step confirms the data is loaded correctly and gives you a sense of its size. If the shape is unexpected (e.g., very few rows or columns), something went wrong with data generation or loading.


**Step 2: Initial Data Inspection** (from `01_EDA_Scenario_Analysis.ipynb`)

* **Code:** The second code block in `eda_hvs_project` (lines starting with `if not df.empty: print("\n--- First 5 rows of the dataset: ---")` down to `print(f"Total unique modules in this dataset: {df['Module_ID'].nunique()}")`).
* **Action:** Run this code block.
* **Expected Output & What to Look For:**
    * **`df.head()` and `df.tail()`:**
        * You'll see the first and last 5 rows. Look at the column names.
        * `Timestamp`: Should look like dates and times (e.g., `2024-01-01 00:00:00`, `2024-01-01 00:00:10`, etc.).
        * `Module_ID`: Will show IDs like `RackA1_Module01`, `RackA2_Module01`, etc.
        * Sensor values: Will be numbers, possibly with decimals.
        * `Hazard_Label`: Will be 0 or 1.
    * **`df.info()`:**
        * `Timestamp`: Should be `datetime64[ns]` type. If it's `object`, it wasn't parsed correctly (though `pd.read_csv` usually handles our format).
        * Sensor columns (e.g., `Module_Avg_Surface_Temp_C`): Should be `float64`.
        * `Module_ID`, `Rack_ID`: Should be `object` (which means string).
        * `Hazard_Label`: Should be `int64`.
        * **Non-Null Counts:** For sensor columns, you should see counts slightly less than the total number of rows for that module type, indicating the presence of *missing values* (NaNs) that our simulator intentionally added. For example, if there are 2160 entries for `RackA1_Module01`, a sensor column for it might show 2158 non-null entries.
    * **`df[numerical_cols].describe()`:**
        * This shows statistics for all numerical columns.
        * **`Module_Avg_Surface_Temp_C`**:
            * `mean`: Might be around 20-30°C, depending on ambient and load.
            * `std`: Standard deviation – gives a sense of temperature variability.
            * `min`: Should be a reasonable ambient temperature.
            * `max`: **This is important!** For the aggressive fault scenario, the `max` temperature could be very high (e.g., >100°C, possibly much higher for the failing module). Other modules should have lower max temps.
            * `25%`, `50%` (median), `75%`: Show the distribution.
        * **`Module_Voltage_V`**: Values typically between ~45V and ~58V for a 14s module. Look for `min` values that might indicate a fault (e.g., a significant drop).
        * **`Module_Current_A`**: Will have positive (charge) and negative (discharge) values. The `min` and `max` will reflect your charge/discharge C-rates.
        * **`Sim_OffGas...proxy`**: Min should be 0. Max could be high for the failing module.
        * **`Hazard_Label`**: `mean` will be very low (e.g., 0.00something) because '1's are rare. `min`=0, `max`=1.
    * **`df.isnull().sum()`:**
        * You'll see a list of columns and the count of NaNs in each. Columns like `Module_Avg_Surface_Temp_C`, `Module_Voltage_V`, etc., should have some missing values (e.g., a few dozen to a hundred or so per sensor column across the whole dataset, depending on the `SENSOR_MISSING_RATE_PERCENT`). This confirms our data imperfection mechanism is working.
    * **`df['Module_ID'].unique()` and `nunique()`:**
        * You should see a list of all 64 unique module IDs (e.g., `['RackA1_Module01' 'RackA1_Module02' ... 'RackB4_Module08']`).
        * `Total unique modules`: Should be 64.
    * **Why this matters:** This step is your first detailed check. You're verifying data types, getting a feel for the typical and extreme values of your sensors, confirming that data imperfections (missing values) are present, and ensuring all modules are represented. Any major discrepancies here (e.g., temperature of -1000°C, wrong number of modules) would indicate a problem in `simulator.py` or the scenario definition.


**Step 3: Analyzing the Target Variable (`Hazard_Label`)** (from `01_EDA_Scenario_Analysis.ipynb`)

* **Code:** The third code block in `eda_hvs_project` (lines starting with `if not df.empty and 'Hazard_Label' in df.columns:`).
* **Action:** Run this code block.
* **Expected Output & What to Look For:**
    * **`label_counts` (printed percentages):**
        * `Hazard_Label`
        * `0    99.xxxx %` (A very high percentage for the "Normal" class)
        * `1     0.yyyy %` (A very small percentage for the "Hazardous" class)
        * For Scenario 4 (6-hour simulation, 1 failing module, 10-minute prediction window):
            * Total data points for one module = `6 * 3600 / 10 = 2160`.
            * Hazardous data points for the failing module = `10 * 60 / 10 = 60`.
            * Percentage of hazardous points for the *failing module alone* = `(60 / 2160) * 100 \approx 2.78%`.
            * Across all 64 modules, the overall percentage of '1's will be `(60 / (2160 * 64)) * 100 \approx 0.043%`.
            * So, `yyyy` will be a very small number.
    * **Plot (`sns.countplot`):**
        * You'll see a bar chart with two bars. The bar for "Normal (0)" will be extremely tall, and the bar for "Hazardous (1)" will be very, very short, almost invisible in comparison without looking at the y-axis scale carefully. The percentages printed on top will confirm the imbalance.
    * **Why this matters:** This visually and numerically confirms the **class imbalance**. This is critical because most machine learning models, by default, perform poorly on imbalanced datasets (they tend to just predict the majority class). Knowing this early tells us we'll need strategies like class weighting, oversampling the minority class, or undersampling the majority class during model training (Phase 4).


**Step 4: Time-Series Visualization of Sensor Data** (from `01_EDA_Scenario_Analysis.ipynb`)

This is the most visual part of EDA for time-series.

**A. Isolate Data for the Failing Module and a Sample Normal Module**

* **Code:** Subsection "A. Isolate Data..." in `eda_hvs_project`.
* **Action:** Run this code.
* **Expected Output & What to Look For:**
    * `Data isolated for failing module: RackA2_Module01 (Shape: (2160, 14))` (or similar row count for 6 hours).
    * `Data isolated for sample normal module: RackA1_Module01 (Shape: (2160, 14))` (or some other module ID that isn't `RackA2_Module01`).
    * The shapes should have the same number of columns, and the number of rows should correspond to the total number of time steps in the simulation (e.g., 2160 for a 6-hour simulation with 10s steps).
    * **Why this matters:** We need to look at the failing module in detail to see the failure signature. Comparing it to a normal module helps us understand what "normal" looks like and how the failing one deviates.

**B. Plotting Key Sensor Data for the Failing Module (`RackA2_Module01`)**

* **Code:** Subsection "B. Plotting Key Sensor Data..." in `eda_hvs_project`.
* **Action:** Run this code. This will generate a multi-panel plot.
* **Expected Output & What to Look For (for `RackA2_Module01` in Scenario 4):**
    * **General Appearance:** Each subplot will show a sensor reading on the y-axis and `Timestamp` on the x-axis. Lines might look a bit "noisy" or jagged due to the `SENSOR_NOISE_STD_DEV_ABS` we added. You might spot occasional missing data points (tiny gaps in the line).
    * **`Module_Avg_Surface_Temp_C` & `Module_Max_Surface_Temp_C`:**
        * For the first hour (before the fault at `1 * SECONDS_IN_HOUR`), temperature should show normal fluctuations (cycling with charge/discharge, influenced by ambient).
        * After the fault starts (1 hour in), you should see the temperature *start to rise gradually* for `RackA2_Module01`.
        * As the "Aggressive ISC" fault progresses (over the next 0.75 hours according to `time_to_max_heat_s`), this temperature rise should become **much steeper and more dramatic**, eventually reaching very high values (potentially >100-150°C or more, depending on the exact heat parameters and thermal mass).
        * **Hazard Window Line:** You should see an orange vertical dashed line labeled "Hazard Window Start". This line marks the beginning of the `Hazard_Label == 1` period. **Crucially, observe if the sharp temperature increase starts *before or around* this line.** Ideally, the line appears just as the temperature is clearly deviating from normal and starting its rapid ascent.
    * **`Module_Voltage_V`:**
        * Normally, voltage cycles with SoC (higher when charged, lower when discharged).
        * When the ISC fault becomes severe in `RackA2_Module01`, the voltage might **drop significantly and erratically** as the internal short bypasses the normal cell path or damages cells.
    * **`Module_Current_A`:**
        * Normally, current follows the charge/discharge profile (e.g., +25A charge, -35A discharge, 0A idle).
        * With a severe ISC, the module might internally discharge very rapidly through the short. This could manifest as a large negative current if the sensor is capturing total module current, or the external current might drop to zero if the BMS disconnects it, while internal current through the short is high (which we don't directly "sense" with an external sensor). Our current `simulator.py` model for current is primarily driven by `assigned_load_current_a`. The effect of an ISC on *measured* current might need more sophisticated modeling in the simulator if this feature is critical. For now, focus on how the *other* parameters change.
    * **`Sim_OffGas_CO_ppm_proxy`, `Sim_OffGas_H2_ppm_proxy`, `Sim_OffGas_VOC_ppm_proxy`:**
        * These should be near zero during normal operation.
        * As the internal temperature of `RackA2_Module01` (driven by the ISC fault) crosses their respective trigger thresholds (`OFFGAS_CO_TRIGGER_INTERNAL_TEMP_C = 75.0`, `H2 = 95.0`, `VOC = 110.0`), these proxy values should **start to rise**.
        * **These are key!** Ideally, these gas proxies should start rising *before* the surface temperature shows its most dramatic spike, making them valuable *early* indicators. Check if their rise aligns with or precedes the "Hazard Window Start" line.
    * **`Ambient_Temp_Rack_C`:** For `RackA2_Module01`, this should generally follow the `typical_daily_ambient_temp_profile` unless there's a separate rack-level scenario (not the case for the *failing module's own rack* in Scenario 4). It should be relatively stable or slowly cycling compared to the module's own temperature during a fault.
    * **`Module_SoC_percent`:**
        * Normally cycles with charge/discharge.
        * When the ISC fault in `RackA2_Module01` becomes severe, the SoC might **plummet rapidly** as the internal short drains the cells. This should be a very clear signal.
    * **Why this matters:** These plots are your primary way to visually confirm that the simulated fault behaves as intended and produces a detectable signature in the sensor data. You're looking for clear deviations from normal *before or during* the labeled hazardous window. If the "Hazard Window Start" line appears *after* all the dramatic changes, then our labeling window or PNR definition might need adjustment.

**C. Plotting Comparison: Failing vs. Normal Module (Temperature)**

* **Code:** Subsection "C. Plotting Comparison..." in `01_EDA_Scenario_Analysis.ipynb`.
* **Action:** Run this code.
* **Expected Output & What to Look For:**
    * You'll see two lines on the plot:
        * One for `Module_Avg_Surface_Temp_C` of `RackA2_Module01` (Failing).
        * One for `Module_Avg_Surface_Temp_C` of a sample normal module (e.g., `RackA1_Module01`), shown with a dashed line.
    * **Comparison:**
        * For the initial period (before the fault in `RackA2_Module01` at 1 hour), both lines should show similar normal operating temperatures, cycling with load and ambient conditions.
        * After the fault starts in `RackA2_Module01`, its temperature line should start to rise and then spike upwards dramatically.
        * The normal module's temperature line should continue its normal behavior, remaining relatively low and stable compared to the failing one.
        * The orange "Hazard Window Start" line should clearly show when the failing module's temperature is entering the anomalous, rapidly rising phase that we want to predict.
    * **Why this matters:** This plot directly visualizes the difference we are trying to teach the ML model: how to distinguish a module that's heading for thermal runaway from one that's operating normally. The clearer the separation, the easier the ML task (in principle).


**Step 5: Data Quality Deep Dive - Missing Values & Outliers** (from `01_EDA_Scenario_Analysis.ipynb`)

* **Code:** The fifth code block in `eda_hvs_project` (lines starting with `if not df.empty: print("\n--- Analyzing Missing Values Further ---")`).
* **Action:** Run this code.
* **Expected Output & What to Look For:**
    * **Missing Values Analysis:** The code currently comments out the heatmap, which is fine. `df.isnull().sum()` from Step 2 already gave us the counts.
    * **Boxplot of `Module_Avg_Surface_Temp_C` (All Modules):**
        * The main "box" will represent the bulk of normal operating temperatures (e.g., between 20°C and 40°C).
        * The "whiskers" will extend from this box.
        * You should see several points plotted as individual dots beyond the upper whisker. These are outliers.
            * Many of these will be the high temperatures from `RackA2_Module01` as it undergoes thermal runaway.
            * A few might be the random "spurious outliers" we added via `introduce_outlier_randomly` on other modules' temperature readings.
    * **Boxplots for Key Features (Failing Module: `RackA2_Module01`):**
        * This will create separate boxplots for `Module_Avg_Surface_Temp_C`, `Module_Voltage_V`, and `Module_Current_A`, but *only using data from `RackA2_Module01`*.
        * **Temperature:** The box might be wider, or the distribution skewed, because it includes both normal operation and the high fault temperatures for this single module. The very high fault temperatures will likely appear as outliers relative to its own "normal" operating range before the fault.
        * **Voltage/Current:** Look for values that are far from the typical range during the fault.
    * **Why this matters:**
        * Boxplots help visualize the distribution and spread of your data and are particularly good at highlighting outliers.
        * Understanding outliers is crucial for preprocessing. Are they genuine extreme values from a fault (which we want to keep and learn from), or are they erroneous sensor readings (which we might want to cap, remove, or transform)? Our simulator creates both types.


**Step 6: Preliminary Thoughts & Next Steps after EDA** (from `01_EDA_Scenario_Analysis.ipynb`)

* **Action:** Now, reflect on what you've seen, using the "Key questions to reflect on" from the `01_EDA_Scenario_Analysis.ipynb` document.
* **Interpreting for Scenario 4:**
    1.  **Realism:** The aggressive fault should show a clear, rapid temperature increase, SoC drop, and gas proxy rise for `RackA2_Module01`. This should feel "plausible" for a severe fault.
    2.  **Predictive Signal:** Yes, there should be very clear visual differences for `RackA2_Module01` in temperature, SoC, and gas proxies during its `Hazard_Label == 1` window compared to its earlier normal operation or other normal modules.
    3.  **Data Quality Impact:** You've seen the missing values via `.isnull().sum()`. The noise should make the line plots somewhat jagged. The boxplots should have highlighted outliers. These will definitely require preprocessing.
    4.  **Feature Importance (Gut Feel):** `Module_Avg_Surface_Temp_C`, `Sim_OffGas_..._proxy` values, and `Module_SoC_percent` likely showed the most dramatic and clear changes for the failing module. `Module_Voltage_V` might also be a strong indicator.
    5.  **Labeling Adequacy:** For Scenario 4 (aggressive fault), the 10-minute `PREDICTION_WINDOW_SECONDS` should capture the steep rise in temperature and other indicators. Observe if the "Hazard Window Start" line appears at a point where these signals are already clearly deviating. If the fault develops *extremely* fast, the 10-minute window might even seem a bit long, or just right. If it developed slower, you'd want to ensure the window isn't too late.