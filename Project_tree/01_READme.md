## HVS Predictive Fire Safety System: A Detailed Project Walkthrough

**Project Goal:** To architect and implement an end-to-end machine learning system capable of providing early warnings for potential fire hazards (specifically thermal runaway precursors) in High-Voltage Battery Storage (HVS) systems. The project emphasizes MLOps best practices, from data generation to conceptual deployment and monitoring.

---

### Phase 1: Data Generation (Simulated & Dummy)

**1. Goal of this Phase:**
The primary goal was to produce a structured time-series dataset that would serve as the foundation for training and evaluating our predictive models. This dataset needed to represent:
* Normal operational behavior of multiple battery modules (initially planned for 64 modules in a simulated warehouse context).
* At least one module undergoing a simulated fault condition that realistically progresses towards thermal runaway.
* Clear, unambiguous labels (`Hazard_Label`) distinguishing normal operation (0) from a critical "hazardous" window (1) that precedes the Point of No Return (PNR) of thermal runaway.
* Simulated real-world data imperfections, such as sensor noise, occasional missing values, and outliers, to make the subsequent data cleaning and model robustness development more realistic.

**2. Rationale (Why this phase is critical):**
Supervised machine learning models, which we aimed to use for prediction, fundamentally learn from labeled examples. Without a dataset containing examples of both normal operation and the specific hazardous pre-failure conditions we want to detect, building an effective predictive system is impossible.
* **Data Scarcity:** Real-world data detailing the precise sensor readings leading up to HVS thermal runaway events is extremely rare, often proprietary, expensive to acquire (as it involves destructive testing), and not readily available in public datasets.
* **Controlled Scenarios:** Synthetic data generation allows us to create controlled fault scenarios, vary parameters, and ensure we have enough instances of the rare "hazardous" class for the model to learn from.
* **Understanding Failure Signatures:** The process of trying to simulate failures forces a deeper understanding of the underlying physics and the sensor patterns that might indicate an impending hazard.

**3. Approach & Key Activities (This was an iterative process):**

* **Initial Approach - Physics-Based Simulator (`src/data_generation/simulator.py`):**
    * The first strategy was to develop a Python-based simulator to model simplified battery physics. The ambition was to create data that was as close to realistic physical behavior as possible within reasonable complexity.
    * **`BatteryModule` Class:** This class was designed to be the core of the simulation. Each instance would represent a single battery module and track its state variables over time:
        * **Electrical State:** State of Charge (SoC), terminal voltage, current (charge/discharge), internal resistance.
        * **Thermal State:** Internal cell temperature (crucial for runaway), surface temperature (what external sensors might measure), rate of temperature change.
        * **Health Status:** A categorical state like "Normal," "Degrading," "PreRunaway," "Runaway."
        * **Off-Gassing Proxies:** Simulated values for CO, H2, and VOCs, triggered by internal temperature thresholds, as off-gassing is a key early indicator.
    * **Simplified Physics Models:**
        * *Thermal Model:* Included equations for heat generation ($I^2R$ ohmic losses from current flow through internal resistance, plus additional heat injected during a simulated fault like an Internal Short Circuit - ISC). It also modeled heat dissipation from the module's core to its surface, and from the surface to the ambient air, considering defined thermal masses and heat transfer coefficients.
        * *Electrical Model:* Implemented a simplified Open Circuit Voltage (OCV) to SoC relationship, and modeled how internal resistance might change with SoC and temperature.
    * **Fault Injection Mechanism:** A method (`introduce_fault`) was created to trigger an ISC in a specific module at a chosen simulation time. The ISC was modeled to generate an increasing amount of heat as the fault progressed over time.
    * **Point of No Return (PNR) Definition:** PNR conditions were defined to mark the onset of uncontrollable thermal runaway. These were based on:
        * Absolute internal temperature exceeding a critical threshold (e.g., `PNR_INTERNAL_TEMP_C = 140.0` °C).
        * OR The rate of internal temperature increase exceeding a critical threshold (e.g., `PNR_DELTA_T_INTERNAL_PER_SECOND_C = 0.5` °C/s), but only if the internal temperature was already significantly elevated (e.g., above `OFFGAS_VOC_TRIGGER_INTERNAL_TEMP_C = 110.0` °C).
    * **Labeling Logic (`get_label` method):**
        * If PNR was triggered for a module at a specific `pnr_triggered_time_s`, a "hazardous" window was defined *before* this time. The `Hazard_Label` would be set to `1` for data points falling within this window.
        * The window was defined by `PREDICTION_WINDOW_SECONDS` (e.g., 10, then 20, then ~33 minutes) and `GUARD_BUFFER_SECONDS` (e.g., 30-60 seconds) to prevent labeling too close to the PNR itself.
    * **Data Imperfections:** Functions were designed to add random noise to sensor readings, introduce `np.nan` for missing values, and inject occasional large outliers to mimic real-world sensor data.
    * **`WarehouseSimulator` Class:** This class was designed to manage the 64 `BatteryModule` instances, orchestrate the simulation steps, and handle global conditions like ambient temperature profiles and load profiles.

* **Challenges & Extensive Debugging of the Physics-Based Simulator:**
    * **Reliably Triggering PNR:** This proved to be the most significant challenge. Despite many iterations trying to tune fault parameters (e.g., `base_isc_heat_W`, `time_to_max_heat_s`, `max_isc_heat_W` for the "Aggressive Fault" scenario) and even temporarily lowering the `PNR_INTERNAL_TEMP_C` to as low as 60°C, the simulated fault often did not generate enough sustained net heat within the module's core to consistently meet the defined PNR conditions.
    * **Interaction of Parameters:** The balance between injected fault heat, ohmic heating, the module's thermal mass (how much energy it takes to heat up), and the heat dissipation rates (core-to-surface, surface-to-ambient) was very delicate. Small changes in one could significantly impact whether runaway was achieved.
    * **Debugging Process:** This involved:
        * Adding numerous `print` statements inside `simulator.py` (especially within `_update_thermal_model` and `_update_failure_progression_and_offgassing` in the `BatteryModule` class) to meticulously trace:
            * If and when a fault was being activated (`introduce_fault`).
            * The calculated `p_gen_fault_W` at each step to ensure it was non-zero and ramping up as intended.
            * The `internal_cell_temp_c` and `delta_internal_temp_per_s` for the failing module.
            * The conditions being checked for PNR.
        * Identifying and fixing several bugs:
            1.  A `NameError` in a debug print statement.
            2.  An incorrect order of method calls in `update_state` (where `fault_progression_time_s` was incremented *after* the thermal model had already used its value for the current step).
            3.  A critical mismatch between the `fault_name` (e.g., `'ISC_EXTREME'`) defined in the scenario and the hardcoded check `self.fault_type == 'ISC'` inside `_update_thermal_model`, which prevented the fault heat calculation block from being entered.
    * Even after these fixes, the default thermal parameters made PNR difficult to achieve without making the fault heat so extreme that it felt less like a subtle onset and more like an instant explosion, or making the PNR temperature unrealistically low for initial testing.

* **Pivot to Dummy Data Generation (`src/data_generation/generate_dummy_data.py` - ID: `dummy_data_generator`):**
    * **Decision Rationale:** To ensure the project could move forward to the EDA, preprocessing, and modeling phases with a dataset that reliably contained the desired hazardous labels and clear anomalous signals, we made a pragmatic decision to use a simpler, more direct "dummy" data generation script. This prioritized unblocking the MLOps workflow stages over perfecting the complex physics simulation at this point.
    * **Script Functionality (`generate_dummy_data.py`):**
        * Uses Pandas and NumPy to directly construct the DataFrame.
        * Generates baseline "normal" data for all 64 modules over a 6-hour period (10-second intervals). This normal data includes slight randomization per module and gradual drifts over time.
        * For a designated `faulty_module` (specifically `"RackA2_Module01"`), it explicitly defines a window of **200 rows** (approximately 33 minutes) where hazardous conditions are injected. This window typically started 1 hour into the simulation.
        * **Hazard Injection Logic:** Within this window, the script programmatically and significantly alters the sensor readings for the faulty module:
            * Temperature (`Module_Avg_Surface_Temp_C`, `Module_Max_Surface_Temp_C`) is increased (e.g., `+= (10 + 60 * progression**2)`, where `progression` goes from 0 to 1 across the window).
            * Voltage (`Module_Voltage_V`) and SoC (`Module_SoC_percent`) are decreased.
            * Gas proxies (`Sim_OffGas_CO_ppm_proxy`, etc.) are substantially increased.
            * For these specific 200 rows for the faulty module, `Hazard_Label` is explicitly set to `1`. All other rows get `Hazard_Label = 0`.
        * **Data Imperfections:** The script also incorporates the previously designed logic to add random noise, introduce missing values (`np.nan`), and inject occasional outliers into the sensor readings for all modules.
    * **Output File:** Produces a CSV file, for example, `data/raw/dummy_fault_RackA2_Module01_200rows.csv`.

**4. Technical Insights & Decisions:**
* **The Simulation vs. Generation Trade-off:** The initial goal of a physics-based simulator is ideal for realism. However, when the complexity of tuning such a simulator becomes a bottleneck for an end-to-end project aimed at learning the full MLOps cycle, a more controlled "data generation" script (like the dummy data script) can be a practical way to ensure a usable dataset. This dataset, while less physically pure, still serves the purpose of testing data pipelines, preprocessing, model training (especially for imbalanced data), and API deployment.
* **Importance of Controlled Labeling:** The dummy script gives precise control over how many "hazardous" samples are generated and what their characteristics are, which is useful for initial model development and testing.
* **Retaining Imperfections:** It was important that even the dummy data generator included noise, NaNs, and outliers, so the subsequent preprocessing phase would be meaningful and address realistic data quality issues.
* **Iterative Debugging:** The process of trying to get the physics-based simulator to work, even if ultimately set aside temporarily, provided valuable insights into the sensitivities of the parameters and the importance of detailed logging and step-by-step verification.

**5. Key Commands/Code Snippets:**
* **Running the (attempted) Physics-Based Simulator:**
    ```powershell
    python src/data_generation/simulator.py
    ```
    *(This involved many iterations with parameter changes and adding debug prints as described above.)*
* **Running the Dummy Data Generator (the approach we proceeded with):**
    ```powershell
    # In an activated venv, from the project root
    python src/data_generation/generate_dummy_data.py
    ```
* **Key snippet from `generate_dummy_data.py` for hazard injection:**
    ```python
    # ... inside loops for module and timestamp ...
    if module == faulty_module and hazard_label_start_idx <= i < hazard_label_end_idx:
        progression = (i - hazard_label_start_idx) / num_hazardous_steps
        row['Module_Avg_Surface_Temp_C'] += (10 + 60 * progression**2) # Example anomaly
        # ... other anomalies for voltage, SoC, gas ...
        row['Hazard_Label'] = 1
    else:
        row['Hazard_Label'] = 0
    # ... then add noise, NaNs, outliers ...
    ```

**6. Outcome of Phase 1:**
* A functional Python script (`src/data_generation/generate_dummy_data.py`) capable of producing a CSV dataset with specified characteristics, including a defined number of hazardous samples for a target module.
* A CSV dataset (e.g., `dummy_fault_RackA2_Module01_200rows.csv`) located in the `data/raw/` directory. This dataset contains:
    * Time-series data for 64 modules over a 6-hour simulation period.
    * Approximately 200 rows explicitly labeled as `Hazard_Label = 1` for module `RackA2_Module01`, with corresponding anomalous sensor readings.
    * All other data points labeled as `Hazard_Label = 0`.
    * Simulated sensor imperfections (noise, missing values, outliers) across all data.
* This dataset became the primary input for the subsequent EDA (Phase 2) and Preprocessing (Phase 3) stages, allowing the project to move forward.
* The original `simulator.py` remains as a development artifact that could be revisited for more advanced, physics-based data generation in the future.

---
