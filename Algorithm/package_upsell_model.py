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
## Stage 1: Building the Data Foundation (Feature Engineering)
**PM Perspective:** Before AI can predict anything, we have to translate raw database rows into behavioral human signals. 
Instead of just looking at raw 'checkout dates', we engineer features like 'Length of Stay' and 'Days Until Trip' because those represent actual user intent.
"""
# %%

print("\n--- Stage 1: Data Preparation ---")
print("Translating raw dates into behavioral signals (stay_duration, days_to_trip)...")

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

# PM Note: 'orig_destination_distance' has many missing values. We are using median imputation here for simplicity,
# but it is important to flag that this could introduce noise into our model's signal.
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
## Stage 2: The Baseline Model (Logistic Regression)
**PM Perspective:** Never jump straight to the most complex AI. Always build a simple, interpretable baseline first. 
If a basic statistical equation (Logistic Regression) achieves our business goals, we don't need to pay for a massive neural network.
"""
# %%

print("\n--- Training Logistic Regression ---")
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# %%
"""
## Stage 3: The Optimized Engine (XGBoost)
**PM Perspective:** To maximize revenue, we need the highest possible accuracy on our structured, proprietary data (our 'Data Moat').
XGBoost is the industry standard for this. Instead of one complex AI brain, it builds hundreds of simple decision trees chronologically. Each new tree focuses *only* on correcting the mistakes of the previous trees.
"""
# %%

print("--- Training XGBoost Classifier ---")
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# %%
"""
## Stage 4: Defining Success (Product Metrics)
**PM Perspective:** Accuracy alone is a vanity metric. If 90% of users never buy a package, a model that guesses "No" every time is 90% accurate, but drives $0 in revenue. We care about balancing Precision and Recall.
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
# PM Note: We use 'roc_auc' (Area Under the Receiver Operating Characteristic Curve) instead of 'accuracy' 
# because 'accuracy' is a vanity metric when our target class (packages) is heavily imbalanced.
grid_search = GridSearchCV(estimator=base_xgb, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)

# 4. Run the Search
print("Searching for the best parameters...")
grid_search.fit(X_train, y_train)

# 5. Look at the results for ALL combinations tested
print("\n--- Detailed Results for Each Setup ---")
results = grid_search.cv_results_
for i in range(len(results['params'])):
    score = results['mean_test_score'][i]
    params = results['params'][i]
    print(f"ROC-AUC Score: {score:.4f} | Parameters: {params}")

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
import time
from tabpfn import TabPFNClassifier

print("\n--- Stage 7: The Foundation Model Prototype (TabPFN) ---")
print("Let's test an emerging 'Tabular Foundation Model' like TabPFN.")
print("It uses a Transformer architecture (like an LLM) pre-trained to understand tabular data without hyperparameter tuning.")

# 1. Initialize TabPFN
# We limit to 1000 rows. While not a fully scaled apples-to-apples comparison with XGBoost,
# it provides a fairer latency benchmark than just 100 rows without completely maxing out memory.
subset_size = min(1000, len(X_train))
X_train_sub = X_train.iloc[:subset_size]
y_train_sub = y_train.iloc[:subset_size]

print(f"\nInitializing TabPFN (Training on {subset_size} samples for demonstration)...")
try:
    # Try older version API if it changed recently
    tabpfn_model = TabPFNClassifier(device='cpu') 
except Exception as e:
    print(f"Init Error: {e}")

# 2. "Train" TabPFN
start_time = time.time()
tabpfn_model.fit(X_train_sub, y_train_sub)
tabpfn_train_time = time.time() - start_time
print(f"TabPFN Fitting Time: {tabpfn_train_time:.4f} seconds")

# 3. Evaluate Accuracy (on a small subset too so it doesn't hang)
# We test on 1000 rows
X_test_sub = X_test.iloc[:1000]
y_test_sub = y_test.iloc[:1000]
y_pred_tabpfn = tabpfn_model.predict(X_test_sub)
evaluate_model("TabPFN Foundation Model (Subset Test)", y_test_sub, y_pred_tabpfn)

# 4. The Critical UX Metric: Inference Latency
print("\n--- The Ultimate PM Metric: Inference Latency (Speed) ---")
print("How fast can these models make a single prediction when a user clicks 'Search'?")

single_user = X_test.iloc[[0]]

start_xgb = time.perf_counter()
best_xgb_model.predict(single_user)
xgb_latency = (time.perf_counter() - start_xgb) * 1000  # in ms

start_tabpfn = time.perf_counter()
tabpfn_model.predict(single_user)
tabpfn_latency = (time.perf_counter() - start_tabpfn) * 1000  # in ms

print(f"XGBoost Single Inference: {xgb_latency:.2f} ms")
print(f"TabPFN Single Inference:  {tabpfn_latency:.2f} ms")

if tabpfn_latency > xgb_latency:
    multiplier = tabpfn_latency / xgb_latency
    print(f"\nConclusion: XGBoost is {multiplier:.0f}x faster for real-time inference.")
    print("While TabPFN is incredibly impressive and requires zero tuning, its latency makes synchronous (real-time) execution difficult at Skyscanner scale compared to optimized tree ensembles.")
