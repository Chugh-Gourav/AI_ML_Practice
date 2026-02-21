# %%
# Predict Package Upsells (Flight + Hotel)

# %%
"""
```python
# Revision: Predicting Package Upsells (Flight + Hotel)
```
**Goal:** Predict whether a user's intent is to book a 'Package' (`is_package = 1`) based on their search contextual features.

*(Note: You must upload the `Expedia_travel.csv` file into this Colab session before running!)*
"""
# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# %%
"""
## Exploratory Data Analysis (EDA)
Before building our models, we need to understand the shape and structure of our dataset. As a Product Manager, asking the right questions here is crucial:
* **How much data do we have?** (Rows and Columns)
* **What does the data look like?** (A quick preview)
* **Is the data imbalanced?** (How often do users naturally book a package vs. just a flight/hotel?)
"""
# %%

print("--- Exploratory Data Analysis ---")

# Load a chunk of the dataset for exploration
df_eda = pd.read_csv('../Datasets/Expedia_travel.csv', nrows=50000)

# 1. Basic Information
print("\n1. Dataset Info:")
df_eda.info()

# 2. Check for Missing Values
print("\n2. Missing Values Summary (Top 5 columns with most missing):")
missing_data = df_eda.isnull().sum().sort_values(ascending=False)
print(missing_data[missing_data > 0].head())

# 3. Target Variable Distribution (Class Imbalance check)
print("\n3. How often do users book a package? (Our Target Variable: 'is_package')")
package_counts = df_eda['is_package'].value_counts()
package_props = df_eda['is_package'].value_counts(normalize=True) * 100

for val in package_counts.index:
    print(f"Class {val}: {package_counts[val]} searches ({package_props[val]:.2f}%)")

# PM Note: If Class 1 (Packages) is very low (e.g., < 20%), our dataset is imbalanced.
# This means our accuracy metric later might look artificially high!

# %%
"""
## Stage 1: Data Preparation & Feature Engineering
"""
# %%

print("Loading and prepping data...")

# We load a chunk of 50,000 rows to make it fast for learning.
df = pd.read_csv('../Datasets/Expedia_travel.csv', nrows=50000)

# Feature Engineering: We need to turn dates into useful numerical features
df['srch_ci'] = pd.to_datetime(df['srch_ci'], errors='coerce')
df['srch_co'] = pd.to_datetime(df['srch_co'], errors='coerce')
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

# 1. Length of stay (Checkout - Checkin)
df['stay_duration'] = (df['srch_co'] - df['srch_ci']).dt.days

# 2. Days until trip (Checkin - Search Date)
df['days_to_trip'] = (df['srch_ci'] - df['date_time']).dt.days

features = ['is_mobile', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'stay_duration', 'days_to_trip', 'orig_destination_distance']
target = 'is_package'

df_clean = df[features + [target]].copy()
for col in features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
"""
## Stage 2: The Baseline Model - Logistic Regression
**MATH BEHIND LOGISTIC REGRESSION:**
It uses the "Sigmoid Function": `P(y=1) = 1 / (1 + e^-(wX + b))`
It draws a straight line (or plane) through the data. It calculates a weight (w) for every feature. If the output probability is > 0.5, it predicts Class 1 (Package).
"""
# %%

print("\n--- Training Logistic Regression ---")
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# %%
"""
## Stage 3: The Advanced Model - Gradient Boosting (XGBoost)
**MATH BEHIND GRADIENT DESCENT & BOOSTING (For Learning):**
Instead of building one massive tree, XGBoost builds hundreds of simple "weak" trees sequentially. Each new tree focuses *only* on the mistakes of the previous trees.

**A Simple Example:** Imagine predicting a user's probability of buying a package (0.0 or 1.0).
1. **Base Prediction:** Start by guessing the average. Let's say 20% of users buy a package. Our `Pred_0` is `0.2` for everyone.
2. **Calculate Error (Residual):** For User A who *did* buy a package (Actual = 1.0), the Error is `Actual - Pred_0 = 1.0 - 0.2 = 0.8`.
3. **Train a Tree on the Error:** We train Tree #1 to predict this `0.8` error using User A's features (e.g., kids, long stay). Tree #1 looks at User A and predicts, say, `0.6`.
4. **Gradient Descent Step (Update):** Update the overall prediction. 
   `New Pred = Old Pred + (Learning Rate * Tree Prediction)`
   With a Learning Rate of `0.1` (our "step size"), the new prediction for User A is: 
   `Pred_1 = 0.2 + (0.1 * 0.6) = 0.26`
5. **Repeat:** Repeat steps 2-4 hundreds of times. User A's prediction slowly climbs from 0.26, to 0.32, to 0.45... inching closer to 1.0. This slow, steady adjustment is "Gradient Descent".
"""
# %%

print("--- Training XGBoost Classifier ---")
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# %%
"""
## Stage 4: Model Evaluation (Product Metrics)
**PRODUCT METRICS EXPLANATION:**
1. Accuracy: `(True Positives + True Negatives) / Total`
   - "How often were we right overall?" (Can be misleading if data is imbalanced).
2. Precision: `True Positives / (True Positives + False Positives)`
   - "When we predicted a Package, how often were they actually a package?"
   - PM Translation: High Precision = Few False Alarms (We don't annoy users with irrelevant popups).
3. Recall: `True Positives / (True Positives + False Negatives)`
   - "Out of all actual Packages, how many did we successfully find?"
   - PM Translation: High Recall = We didn't miss out on lucrative upsell opportunities.
"""
# %%

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{name} Results:")
    print(f"Accuracy : {acc:.4f} (Correctly classified {acc*100:.2f}% of the time)")
    print(f"Precision: {prec:.4f} (When we predict package, we are right {prec*100:.2f}% of the time)")
    print(f"Recall   : {rec:.4f} (We caught {rec*100:.2f}% of all actual packages)")
    print(f"F1-Score : {f1:.4f} (Harmonic mean of Precision and Recall)")

evaluate_model("Logistic Regression Baseline", y_test, y_pred_log)
evaluate_model("XGBoost Advanced Model", y_test, y_pred_xgb)


# %%
"""
## Stage 5: Hyperparameter Tuning (Finding the Best Settings)
**WHAT IS HYPERPARAMETER TUNING?**
Algorithms like XGBoost have settings (called "hyperparameters") that we have to set *before* training begins.
Think of it like tuning the dials on a radio to get the clearest signal. 

For XGBoost, some important dials are:
1. `learning_rate`: How big of a step we take during Gradient Descent. 
   - *Too large:* We might step over the best solution. 
   - *Too small:* It takes forever to learn.
2. `n_estimators`: How many trees to build.
   - *Too many:* The model memorizes the data (Overfitting).
   - *Too few:* The model doesn't learn enough (Underfitting).

We can test multiple combinations automatically to find what works best!
"""
# %%

from sklearn.model_selection import GridSearchCV

print("\n--- Hyperparameter Tuning (Grid Search) ---")
# 1. Define the parameters we want to test
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}

# 2. Create a basic model
base_xgb = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# 3. Set up the Search (This will try all 3x3 = 9 combinations!)
grid_search = GridSearchCV(estimator=base_xgb, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

# 4. Run the Search
print("Searching for the best parameters...")
grid_search.fit(X_train, y_train)

# 5. Look at the results for ALL combinations tested
print("\n--- Detailed Results for Each Setup ---")
results = grid_search.cv_results_
for i in range(len(results['params'])):
    acc = results['mean_test_score'][i]
    params = results['params'][i]
    print(f"Accuracy: {acc:.4f} | Parameters: {params}")

print(f"\n🏆 Best Settings Found: {grid_search.best_params_}")
best_xgb_model = grid_search.best_estimator_

# 6. Evaluate the Best Model
y_pred_best = best_xgb_model.predict(X_test)
evaluate_model("Tuned XGBoost Model", y_test, y_pred_best)

# %%
"""
## Stage 6: Testing with Real-World Scenarios
Let's act like PMs! We'll create some fake user profiles to see who the model predicts will buy a package and with what probability.

**Features we used:** `is_mobile`, `srch_adults_cnt`, `srch_children_cnt`, `srch_rm_cnt`, `stay_duration`, `days_to_trip`, `orig_destination_distance`
"""
# %%

# Create a DataFrame with 10 distinct user personas
dummy_data = pd.DataFrame([
    # 1. "The Family Planner"
    {'is_mobile': 0, 'srch_adults_cnt': 2, 'srch_children_cnt': 2, 'srch_rm_cnt': 1, 'stay_duration': 7, 'days_to_trip': 90, 'orig_destination_distance': 3000},
    # 2. "The Last-Minute Business Traveler"
    {'is_mobile': 1, 'srch_adults_cnt': 1, 'srch_children_cnt': 0, 'srch_rm_cnt': 1, 'stay_duration': 2, 'days_to_trip': 1, 'orig_destination_distance': 200},
    # 3. "The Couple's Weekend Getaway"
    {'is_mobile': 0, 'srch_adults_cnt': 2, 'srch_children_cnt': 0, 'srch_rm_cnt': 1, 'stay_duration': 3, 'days_to_trip': 30, 'orig_destination_distance': 800},
    # 4. "The Solo Backpacker"
    {'is_mobile': 1, 'srch_adults_cnt': 1, 'srch_children_cnt': 0, 'srch_rm_cnt': 1, 'stay_duration': 14, 'days_to_trip': 60, 'orig_destination_distance': 5000},
    # 5. "The Group Trip Planner"
    {'is_mobile': 0, 'srch_adults_cnt': 6, 'srch_children_cnt': 0, 'srch_rm_cnt': 3, 'stay_duration': 5, 'days_to_trip': 120, 'orig_destination_distance': 1500},
    # 6. "The Staycationer"
    {'is_mobile': 1, 'srch_adults_cnt': 2, 'srch_children_cnt': 2, 'srch_rm_cnt': 1, 'stay_duration': 2, 'days_to_trip': 7, 'orig_destination_distance': 30},
    # 7. "The Honeymooners"
    {'is_mobile': 0, 'srch_adults_cnt': 2, 'srch_children_cnt': 0, 'srch_rm_cnt': 1, 'stay_duration': 10, 'days_to_trip': 180, 'orig_destination_distance': 4000},
    # 8. "The Deal Hunter"
    {'is_mobile': 1, 'srch_adults_cnt': 2, 'srch_children_cnt': 1, 'srch_rm_cnt': 1, 'stay_duration': 4, 'days_to_trip': 14, 'orig_destination_distance': 600},
    # 9. "The Extended Stay Business"
    {'is_mobile': 0, 'srch_adults_cnt': 1, 'srch_children_cnt': 0, 'srch_rm_cnt': 1, 'stay_duration': 21, 'days_to_trip': 10, 'orig_destination_distance': 1200},
    # 10. "The Luxury Spender"
    {'is_mobile': 0, 'srch_adults_cnt': 2, 'srch_children_cnt': 0, 'srch_rm_cnt': 2, 'stay_duration': 5, 'days_to_trip': 45, 'orig_destination_distance': 2500}
])

personas = [
    "The Family Planner", "The Last-Minute Business Traveler", "The Couple's Weekend Getaway",
    "The Solo Backpacker", "The Group Trip Planner", "The Staycationer",
    "The Honeymooners", "The Deal Hunter", "The Extended Stay Business", 
    "The Luxury Spender"
]

# Ensure columns match training order perfectly
dummy_data = dummy_data[features] 

print("\n--- Testing Model on User Personas (A/B Test Logic) ---")
print("Rule: Only show the 'Bundle & Save' popup if Probability > 0.6 (60%)")

# predict_proba returns probabilities for [Class0, Class1]. We only want the probability of Class 1 (Package)
probabilities = best_xgb_model.predict_proba(dummy_data)[:, 1] 

for i in range(len(dummy_data)):
    print(f"\nPersona {i+1}: {personas[i]}")
    
    # Check A/B Test Popup Logic (Probability > 0.6)
    trigger_popup = probabilities[i] > 0.6
    
    print(f"Probability of Upsell: {probabilities[i]*100:.2f}%")
    print(f"Action: {'� SHOW POPUP (Prob > 60%)' if trigger_popup else '🛑 NO POPUP (Skip to reduce friction)'}")

print("\nPM Takeaway: Setting the threshold at 0.6 balances aggressive upselling with user experience.")
