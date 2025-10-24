## Data Science Project Summary: Predicting 'y' using Feature Engineering and XGBoost

### Overview
This project involved a regression task to predict the target variable **'y'** from the training data, which initially contained features **'w'** and **'x'**. The key insight to solving the problem lay in identifying a **non-linear relationship** between the target variable and a combination of the input features.

---

### Key Steps and Findings

#### 1. Hidden Relationship Discovery (Sinusoidal)

The initial visualization and analysis of the dataset hinted at a complex relationship for direct features 'w' and 'x' against 'y'.

* **Crucial Insight:** By visualizing the relationship between the target **'y'** and the engineered feature **'wx'** (i.e., the product of 'w' and 'x'), a clear **sinusoidal (sine wave) pattern** was revealed: `y` $\approx \sin(z)$ where $z = w \cdot x$. This was the **hidden secret** to unlocking the predictive power of the model.

#### 2. Feature Engineering

A new feature **'z'** was created by multiplying the original features 'w' and 'x'.

$$\mathbf{z} = \mathbf{w} \cdot \mathbf{x}$$

This transformed the complex non-linear problem in the original feature space ('w', 'x') into a simpler, albeit still non-linear, problem in the one-dimensional feature space ('z').

---

#### 3. Data Sampling Strategy

Given the large dataset and the requirement to capture the continuous, cyclical nature of the sinusoidal relationship, a **clever sampling method** was employed:

* The training data was **sorted by the new feature 'z'**.
* **Equally spaced points** were then sampled from the sorted data (1% of the total dataset) to create `X_sampled` and `y_sampled`.
* **Purpose:** This ensured that the training subset adequately covered the full range of the sinusoidal curve, including its peaks, troughs, and zero-crossing points, preventing the model from missing the true underlying function.

#### 4. Model Training and Hyperparameters

An **XGBoost Regressor** was chosen for its ability to model complex non-linear relationships effectively.

* **Model:** `xgboost.XGBRegressor`
* **Hyperparameter Tuning:** The model was configured with a set of parameters:
    * `objective='reg:squarederror'` (Standard for regression)
    * `n_estimators=200` (Number of boosting rounds)
    * `learning_rate=0.1` (Step size shrinkage)
    * `max_depth=6` (Maximum depth of a tree)
    * `random_state=42` (For reproducibility)

#### 5. Evaluation and Prediction

The model was trained on the sampled data and evaluated on a hold-out test set from the sampled data.

* **Root Mean Squared Error (RMSE):**
    $$\text{RMSE} = 0.07356$$
* **Final Prediction:** The trained XGBoost model (`xgb_model_z`) was used to generate predictions for the original `test_df` after applying the same feature engineering step ($z = w \cdot x$). The predicted 'y' values were saved into the `test_df` under the column `'y_predicted'`.
