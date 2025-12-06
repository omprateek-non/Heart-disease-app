import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart stroke Predicter")
st.markdown("provide the following details")
age=st.slider("Age",18,100,40)
sex=st.selectbox("Sex",["Male","Female"])
chest_pain_type=st.selectbox("chest_pain_type",["typical angina","atypical angina","non-anginal pain","asymptomatic"])
resting_blood_pressure=st.slider("resting_blood_pressure",80,200,120)
serum_cholestoral=st.slider("serum_cholestoral",100,600,200)
fasting_blood_sugar=st.selectbox("fasting_blood_sugar",["true","false"])
resting_ecg=st.selectbox("resting_ecg",["normal","ST-T wave abnormality","left ventricular hypertrophy"])
max_heart_rate_achieved=st.slider("max_heart_rate_achieved",60,220,150)
exercise_induced_angina=st.selectbox("exercise_induced_angina",["yes","no"])
oldpeak=st.slider("oldpeak",0.0,6.0,1.0)
st_slope=st.selectbox("st_slope",["upsloping","flat","downsloping"])



st.button("Predict")
input_data = {
    "age": age,
    "sex" : sex,
    "chest_pain_type": chest_pain_type,
    "resting_blood_pressure": resting_blood_pressure,
    "serum_cholestoral": serum_cholestoral,
    "fasting_blood_sugar": fasting_blood_sugar,
    "resting_ecg": resting_ecg,
    "max_heart_rate_achieved": max_heart_rate_achieved,
    "exercise_induced_angina": exercise_induced_angina,
    "oldpeak": oldpeak,
    "st_slope": st_slope
}

input_df = pd.DataFrame([input_data])

for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_columns]
scaled_data = scaler.transform(input_df)
prediction = model.predict(scaled_data)[0]

if prediction == 1:
    st.error("The person is likely to have a heart stroke")
else:
    st.success("The person is unlikely to have a heart stroke")
