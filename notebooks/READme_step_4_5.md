Now that we have the dummy data with both 'Normal' (0) and 'Hazardous' (1) labels, let's dive into what to look for in the EDA steps, particularly Step 4 and Step 5, and then discuss class imbalance in detail.

---

**Understanding EDA Step 4: Time-Series Visualization**

The goal here is to *visually confirm* that the data for the faulty module (`RackA2_Module01`) looks different during the hazardous period compared to its own normal operation or compared to other healthy modules.

* **A. Plotting Key Sensor Data for the Failing Module (`RackA2_Module01`)**
    * **What the Code Does:** This part generates multiple subplots, each showing one sensor reading (like temperature, voltage, etc.) for *only* `RackA2_Module01` over the entire 6-hour simulation time. It also marks the start of the hazardous window (where `Hazard_Label` should be 1, roughly 1.00h to 1.56h in the dummy data).
    * **What to Look For (Specifically in the 1.00h - 1.56h window):**
        * **Temperature Plots (`Module_Avg/Max_Surface_Temp_C`):** You should see a clear and significant *increase* in temperature starting around the 1-hour mark and continuing through the window. The dummy data adds `(10 + 60 * progression**2)`, so the temperature should rise quite sharply compared to the baseline ~25°C.
        * **Voltage Plot (`Module_Voltage_V`):** You should see a distinct *drop* in voltage during this window compared to the baseline ~51V. The dummy data subtracts `(2 + 15 * progression)`.
        * **SoC Plot (`Module_SoC_percent`):** You should see a noticeable *decrease* in the State of Charge during this window, faster than the slight downward drift seen in normal operation. The dummy data subtracts `(5 + 40 * progression)`.
        * **Gas Proxy Plots (`Sim_OffGas...`):** These should show a very large *increase* during the hazardous window compared to their baseline near ~2 ppm. The dummy data adds hundreds of ppm, increasing with `progression`.
        * **Current Plot (`Module_Current_A`):** In the *dummy data*, this was just set to random noise around 0. You likely won't see a significant change here unless you modify the dummy generator to simulate high current during the fault.
        * **Hazard Window Line:** The vertical orange line should appear at the start of these anomalies (around 1 hour). The anomalies should persist until around the 1.56-hour mark.
    * **Why it Matters:** This visually confirms that the hazardous state (`Hazard_Label == 1`) corresponds to measurable, anomalous sensor readings in our dataset. It gives us confidence that a machine learning model *might* be able to learn these patterns. If the plots showed no difference during the hazardous window, our dummy data generation would be flawed.

* **B. Plotting Comparison: Failing vs. Normal Module (Temperature)**
    * **What the Code Does:** This plots the average temperature of the failing module (`RackA2_Module01`) and a sample normal module (e.g., `RackA1_Module01`) on the *same graph*.
    * **What to Look For:** You should see the two temperature lines tracking each other relatively closely for the first hour. Then, around the 1-hour mark, the line for `RackA2_Module01` should dramatically diverge upwards, while the line for the normal module continues its typical, much lower fluctuation. The orange line marks the start of this divergence for the failing module.
    * **Why it Matters:** This directly highlights the difference the model needs to learn – distinguishing the faulty module's thermal signature from a healthy one operating under similar ambient conditions.

---

**Understanding EDA Step 5: Data Quality Deep Dive (Boxplots)**

The goal here is to understand the distribution of values for each sensor and identify potential outliers, both across all data and specifically for the failing module.

* **A. Boxplot of Temperature (All Modules)**
    * **What the Code Does:** Creates a boxplot showing the distribution of `Module_Avg_Surface_Temp_C` for *all 138,240 data points*.
    * **What to Look For:**
        * The main "box" will represent the typical temperature range during normal operation (e.g., 20-35°C).
        * You should see many points plotted as dots far above the upper "whisker". These are outliers.
        * Most of these high outliers will correspond to the high temperatures reached by `RackA2_Module01` during its hazardous/fault phase.
        * There might be a few other outliers due to the random `introduce_outlier_randomly` function applied to other modules.
    * **Why it Matters:** This confirms that the fault condition creates values that are statistically outliers compared to the overall dataset. It also shows the presence of other random outliers that our preprocessing will need to handle.

* **B. Boxplots for Key Features (Failing Module Only)**
    * **What the Code Does:** Creates separate boxplots for Temperature, Voltage, and Current, using *only* the data points for `RackA2_Module01`.
    * **What to Look For:**
        * **Temperature:** The box itself might be wider or higher than for a normal module, and the upper whisker will likely extend much further. The highest temperatures during the fault will probably still show as outliers *even relative to this module's own data*, because it spends most of its time in the normal range.
        * **Voltage/SoC:** Similarly, the lower whisker might be extended downwards, and the lowest values during the fault might appear as outliers below the lower whisker.
    * **Why it Matters:** This helps understand the range of values *within* the faulty module's lifecycle. It shows how distinct the hazardous period values are compared to its normal operation.

---

**Understanding Class Imbalance**

* **Is the data imbalanced?**
    * **Yes, absolutely.** Your Step 3 output confirms this:
        ```
        Hazard_Label
        0    138040 (99.855%)
        1       200 ( 0.145%)
        ```
    * The "Normal" class (0) vastly outnumbers the "Hazardous" class (1).

* **Why is it imbalanced?**
    * This is **inherent to the problem domain**. Battery thermal runaway events (and the hazardous states preceding them) are, thankfully, **rare** compared to the vast amount of time batteries operate normally.
    * Our simulation reflects this reality. Even though we simulated only 6 hours, the hazardous window for the single failing module is only about 33 minutes (200 steps * 10s/step = 2000s), while the total time simulated across all modules is 6 hours * 64 modules = 384 module-hours.

* **Why is imbalance a problem for Machine Learning?**
    * **Default Model Behavior:** Most standard ML algorithms (like Logistic Regression, basic Neural Networks, etc.) aim to minimize overall error or maximize overall accuracy across *all* data points.
    * **Misleading Accuracy:** With 99.85% normal data, a lazy model could achieve 99.85% accuracy by simply *always* predicting "Normal" (label 0). It completely ignores the rare, but critical, hazardous class.
    * **Ignoring the Minority Class:** Because the hazardous class contributes so little to the overall error calculation, the model might not learn its patterns effectively. It prioritizes getting the majority class right.
    * **Poor Real-World Performance:** Such a model would be useless in practice because it would fail to predict the actual fire hazards (high **False Negatives**), which is the entire point of the system!
    * **Need for Better Metrics:** Simple accuracy is not a good metric here. We need metrics that focus on the positive (hazardous) class performance, such as:
        * **Recall (Sensitivity):** Out of all *actual* hazardous events, how many did the model correctly identify? (We want this to be very high, like the 92% mentioned in your project description). `Recall = True Positives / (True Positives + False Negatives)`
        * **Precision:** Out of all events the model *predicted* as hazardous, how many were *actually* hazardous? (We want this to be reasonably high to avoid too many false alarms, like the 89% mentioned). `Precision = True Positives / (True Positives + False Positives)`
        * **F1-Score:** The harmonic mean of Precision and Recall, providing a single balanced measure.
        * **Precision-Recall Curve (PR Curve) & AUC-PR:** Visualizes the trade-off between precision and recall; the Area Under the Curve (AUC-PR) summarizes this. Better for imbalanced data than the standard ROC curve.

* **What to do about Imbalance? (Strategies for Phase 4 - Modeling)**
    * **1. Data-Level Approaches (Resampling):**
        * **Oversampling:** Create copies of the minority class (label 1) examples or generate synthetic minority examples (like using **SMOTE** - Synthetic Minority Over-sampling Technique) to balance the dataset *before* training.
        * **Undersampling:** Randomly remove examples from the majority class (label 0). This is faster but risks losing valuable information about normal behavior.
        * **Combination/Hybrid:** Often, combining oversampling the minority and undersampling the majority works well.
    * **2. Algorithm-Level Approaches (Cost-Sensitive Learning):**
        * **Class Weighting:** Modify the learning algorithm's objective function to penalize misclassifications of the minority class more heavily than misclassifications of the majority class. Many algorithms (Logistic Regression, SVMs, Random Forests, XGBoost, LightGBM) support a `class_weight` or `scale_pos_weight` parameter. For example, if class 1 is 0.14% of the data, you might give it a weight roughly 1/0.0014 ≈ 700 times higher than class 0.
    * **3. Evaluation Metrics:** As mentioned above, use metrics like Recall, Precision, F1-score, and AUC-PR instead of just Accuracy.

* **Do we need to increase the number of hazardous rows (e.g., from 200)?**
    * **Not necessarily required right now.** While 200 rows is a small absolute number compared to 138k, the *percentage* (0.14%) represents a realistic, highly imbalanced scenario.
    * The techniques above (especially Class Weighting and SMOTE) are specifically designed to work even with such low numbers of minority samples. Trying to handle the imbalance using these methods is standard practice and a key part of building a robust model for rare event prediction.
    * Artificially generating *many* more hazardous rows in the dummy data might make the initial modeling seem easier, but it wouldn't reflect the real-world data challenge as accurately.
    * **Recommendation:** Let's proceed with the current dataset (138040 vs 200). It forces us to correctly apply imbalance handling techniques in the modeling phase, which is an important learning objective based on your project description and likely interview questions. We can always generate more data later if we find the model *really* struggles even with these techniques.

---

**Summary & Next Step:**

- We've successfully generated usable dummy data with the critical hazardous labels included. 
- We should now finalize your visual inspection (EDA Steps 4 & 5) to confirm the anomalies are present as expected in the dummy data. 
- The key takeaway is the confirmation of significant class imbalance, which dictates our strategy for model training and evaluation later.
