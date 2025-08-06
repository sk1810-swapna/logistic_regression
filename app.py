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

# Load preprocessing objects
if os.path.exists("le_embarked.pkl"):
    with open("le_embarked.pkl", "rb") as f:
        le_embarked = pickle.load(f)
else:
    st.error("Missing le_embarked.pkl. Please upload it to the app directory.")

age_imputer = pickle.load(open("age_imputer.pkl", "rb"))
fare_imputer = pickle.load(open("fare_imputer.pkl", "rb"))
le_sex = pickle.load(open("le_sex.pkl", "rb"))

# Define feature order
# Define feature order (updated)
feature_names = ["Pclass", "Sex", "Age", "Fare", "Embarked", "SibSp", "Parch"]

# Streamlit UI
st.title("Titanic Survival Prediction")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=100)
fare = st.number_input("Fare", min_value=0.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ["C", "Q", "S", ""])
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
    # Fill missing 'Embarked' with default value
    input_df["Embarked"] = input_df["Embarked"].fillna("S")

    # Impute missing values
    input_df["Age"] = age_imputer.transform(input_df[["Age"]])
    input_df["Fare"] = fare_imputer.transform(input_df[["Fare"]])

    # Encode categorical variables
    input_df["Sex"] = le_sex.transform(input_df["Sex"])
    input_df["Embarked"] = le_embarked.transform(input_df["Embarked"])

    # Ensure all required features are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")

except Exception as e:
    st.error(f"Error during prediction: {e}")
