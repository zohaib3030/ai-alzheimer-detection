import streamlit as st
import pickle
import numpy as np
st.title("AI for Alzheimer's Early Detection!")
st.write("Enter patient details to predict Alzheimer's risk")

#warning
st.warning(
    "⚠️ Disclaimer: This application is an educational prototype and not a medical diagnosis. "
    "Please consult a healthcare professional for medical advice."
)

#taking input from user
age=st.number_input("Age:",min_value=40,max_value=100,value=65)
mmse=st.number_input("MMSE Score:",min_value=0,max_value=30,value=25)
educ=st.number_input("Years of Education:",min_value=0,max_value=16,value=12)

#loading ML trained model
model = pickle.load(open("alzheimer_model.pkl", "rb"))

# Predict button
if st.button("Predict Alzheimer’s Risk"):
    input_data = np.array([[age,mmse,educ]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk: Demented")
    else:
        st.success("Low Risk: Non-Demented")
