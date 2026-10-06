import streamlit as st
from joblib import load
import pandas as pd

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide"
)

st.title("🩺 Diabetes Progression Prediction")

st.markdown("Enter the patient's values below")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age Feature", value=0.00, format="%.4f")
    sex = st.number_input("Sex Feature", value=0.00, format="%.4f")
    bmi = st.number_input("BMI Feature", value=0.00, format="%.4f")
    bp = st.number_input("Blood Pressure Feature", value=0.00, format="%.4f")
    s1 = st.number_input("S1 (Total Cholesterol)", value=0.00, format="%.4f")

with col2:
    s2 = st.number_input("S2 (LDL)", value=0.00, format="%.4f")
    s3 = st.number_input("S3 (HDL)", value=0.00, format="%.4f")
    s4 = st.number_input("S4 (TCH)", value=0.00, format="%.4f")
    s5 = st.number_input("S5 (LTG)", value=0.00, format="%.4f")
    s6 = st.number_input("S6 (GLU)", value=0.00, format="%.4f")

#model_path = r"C:\KZTWorkingFile\mine\tbc\ML\saved_ML\diabetes_model.pkl"
#pca_path = r"C:\KZTWorkingFile\mine\tbc\ML\saved_ML\diabetes_pca.pkl"

model_path = "diabetes_model.pkl"
pca_path = "diabetes_pca.pkl"


if st.button("🔍 Predict"):

    try:
        model = load(model_path)
        pca = load(pca_path)

        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "bp": [bp],
            "s1": [s1],
            "s2": [s2],
            "s3": [s3],
            "s4": [s4],
            "s5": [s5],
            "s6": [s6]
        })

        input_pca = pca.transform(input_data)

        prediction = model.predict(input_pca)[0]

        st.metric(
            label="Predicted Diabetes Progression Score",
            value=f"{prediction:.2f}"
        )

        if prediction < 100:
            st.success("✅ Low Risk")
        elif prediction < 200:
            st.warning("⚠️ Medium Risk")
        else:
            st.error("🔴 High Risk")

    except Exception as e:
        st.error(f"Error: {e}")

st.info("These are standardised features from sklearn's diabetes dataset and do not represent actual medical measurements.")