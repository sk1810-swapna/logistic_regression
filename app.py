import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and preprocessing objects
model = pickle.load(open("model.pkl", "rb"))
age_imputer = pickle.load(open("age_imputer.pkl", "rb"))
fare_imputer = pickle.load(open("fare_imputer.pkl", "rb"))
le_sex = pickle.load(open("le_sex.pkl", "rb"))
le_emb = pickle.load(open("le_emb.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))  # saved after training
combined = pickle.load(open("combined.pkl", "rb"))  # used for mode imputation

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

# Preprocess input
try:
    input_df["Age"] = age_imputer.transform(input_df[["Age"]])
    input_df["Fare"] = fare_imputer.transform(input_df[["Fare"]])
    input_df["Embarked"] = input_df["Embarked"].fillna(combined["Embarked"].mode()[0])
    input_df["Sex"] = le_sex.transform(input_df["Sex"])
    input_df["Embarked"] = le_emb.transform(input_df["Embarked"])

    # Add missing columns if needed
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # or appropriate default

    # Reorder columns
    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")

except Exception as e:
    st.error(f"Error during prediction: {e}")
