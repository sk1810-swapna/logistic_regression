import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
# Load model
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please upload model.pkl to the app directory.")
    st.stop()

# Load feature names
feature_names = None
if os.path.exists("feature_names.pkl"):
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
else:
    st.error("Missing feature_names.pkl. Please upload it to the app directory.")
    st.stop()

# Load preprocessing objects
try:
    age_imputer = pickle.load(open("age_imputer.pkl", "rb"))
    fare_imputer = pickle.load(open("fare_imputer.pkl", "rb"))
    le_sex = pickle.load(open("le_sex.pkl", "rb"))
    le_embarked = pickle.load(open("le_embarked.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Missing preprocessing file: {e}")
    st.stop()

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")

st.markdown("Enter passenger details below to predict survival:")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
pclass = st.selectbox("Pclass", [1, 2, 3])
sibsp = st.number_input("SibSp", min_value=0, value=0)
parch = st.number_input("Parch", min_value=0, value=0)

# Create input DataFrame
input_data = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "Fare": fare,
    "Embarked": embarked,
    "SibSp": sibsp,
    "Parch": parch
}
input_df = pd.DataFrame([input_data])

# Run prediction only when button is clicked
if st.button("Predict"):
    try:
        # Impute missing values
        input_df["Age"] = age_imputer.transform(input_df[["Age"]])
        input_df["Fare"] = fare_imputer.transform(input_df[["Fare"]])

        # Encode categorical variables
        input_df["Sex"] = le_sex.transform(input_df["Sex"])
        input_df["Embarked"] = le_embarked.transform(input_df["Embarked"])

        # Add any missing columns with default value
        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0

        # Reorder columns to match training
        input_df = input_df[feature_names]

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
