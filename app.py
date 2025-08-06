import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model and preprocessing objects
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please upload model.pkl to the app directory.")
import os
import streamlit as st

if os.path.exists("le_embarked.pkl"):
    with open("le_embarked.pkl", "rb") as f:
        le_embarked = pickle.load(f)
else:
    st.error("Missing le_embarked.pkl. Please upload it to the app directory.")

# Load other required files
age_imputer = pickle.load(open("age_imputer.pkl", "rb"))
fare_imputer = pickle.load(open("fare_imputer.pkl", "rb"))
le_sex = pickle.load(open("le_sex.pkl", "rb"))
le_embarked = pickle.load(open("le_embarked.pkl", "rb"))  # ✅ correct filename
feature_names = pickle.load(open("feature_names.pkl", "rb"))
combined = pickle.load(open("combined.pkl", "rb"))

# Streamlit UI
st.title("Titanic Survival Prediction")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=100)
fare = st.number_input("Fare", min_value=0.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
pclass = st.selectbox("Pclass", [1, 2, 3])
sibsp = st.number_input("SibSp", min_value=0)
parch = st.number_input("Parch", min_value=0)

# Create input DataFrame
input_df = pd.DataFrame([{
    "Age": age,
    "Fare": fare,
    "Sex": sex,
    "Embarked": embarked,
    "Pclass": pclass,
    "SibSp": sibsp,
    "Parch": parch
}])

# Preprocess and predict
try:
    # Apply same preprocessing as training
    input_df["Age"] = age_imputer.transform(input_df[["Age"]])
    input_df["Fare"] = fare_imputer.transform(input_df[["Fare"]])
    input_df["Embarked"] = input_df["Embarked"].fillna(combined["Embarked"].mode()[0])
    input_df["Sex"] = le_sex.transform(input_df["Sex"])
    input_df["Embarked"] = le_emb.transform(input_df["Embarked"])

    # Add missing columns if needed
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # default value

    # Reorder columns to match training
    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")

except Exception as e:
    st.error(f"Error during prediction: {e}")
