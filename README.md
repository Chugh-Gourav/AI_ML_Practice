# ✈️ AI-Driven Travel Package Upsell: A Product Management Case Study

Welcome to this learning project! This repository contains a complete end-to-end demonstration of how Machine Learning can be applied to solve a common e-commerce challenge: **when and how to upsell.**

Specifically, we use flight search data to predict whether a user is likely to book a "Package" (Flight + Hotel), and we use those predictions to conditionally trigger a UI popup, balancing revenue generation with user friction.

## 🎯 The Core Concept
Instead of aggressively showing a "Bundle & Save" popup to *every* single user (which causes friction and annoyance), what if we could predict *who* actually wants a package? 

By using an **XGBoost Classification Model**, we calculate a probability score for each user. 
- **Probability > 60%:** We show them a subtle, targeted upsell popup to drive revenue.
- **Probability < 60%:** We suppress the popup, letting them book their flight/hotel seamlessly.

## 📁 Repository Structure

*   **/Algorithm/package_upsell_model.py**
    *   The core Machine Learning pipeline. 
    *   Includes Exploratory Data Analysis (EDA), Data Preprocessing, and Model Training (Logistic Regression baseline vs. tuned XGBoost).
    *   Contains in-code, PM-friendly mathematical explanations of Gradient Descent.
    *   Creates 10 diverse "Dummy Personas" to simulate real-world testing.
*   **/Algorithm/skyscanner_upsell_demo.html**
    *   A frontend UI prototype ("Skydemo") built in HTML/CSS/JS.
    *   Replicates a modern hotel search experience.
    *   Demonstrates the final UX of the A/B test: injecting a subtle slide-up popup only when a high-probability simulated persona hits the page.

*(Note: The `Expedia_travel.csv` dataset, sourced via Kaggle, is required in a `/Datasets/` folder to run the `.py` script, but is excluded from version control due to file size).*



## 🚀 How to Run

1.  **To see the ML Model:** Run `python Algorithm/package_upsell_model.py` in your terminal. You will see the persona simulations print to the console.
2.  **To see the UI Prototype:** Open `Algorithm/skyscanner_upsell_demo.html` in any web browser to see the "Skydemo" experience, and watch the subtle AI-driven popup slide in based on the backend model.
