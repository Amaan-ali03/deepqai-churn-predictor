import streamlit as st
import lightgbm as lgb
import pandas as pd
import joblib

# Load trained model
model = lgb.Booster(model_file='lgbm_churn_model.txt')

# Get model feature names
model_features = model.feature_name_

# App title
st.title("Churn Probability Predictor")
st.write("Enter key customer features to predict churn probability.")

# Example input fields (adjust to match your top features)
X1 = st.number_input("X1 (numeric)", value=1600)
X98 = st.number_input("X98 (numeric)", value=0.0)
X19 = st.number_input("X19 (numeric)", value=0.0)
X85 = st.number_input("X85 (numeric)", value=0.0)
lease_duration = st.number_input("Lease duration (X0 numeric)", value=12)

# Predict
if st.button("Predict"):
    # Build input dataframe with provided fields
    input_df = pd.DataFrame({
        'X1': [X1],
        'X98': [X98],
        'X19': [X19],
        'X85': [X85],
        'X0': [lease_duration]
    })

    # Add missing columns expected by model, fill with 0
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match model
    input_df = input_df[model_features]

    # Predict churn probability
    prob = model.predict_proba(input_df)[:, 1][0]
    st.success(f"Predicted churn probability: {prob:.2%}")

    if prob > 0.5:
        st.warning("High risk of churn")
    else:
        st.info("Low risk of churn")
