import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
model = joblib.load("ml_model/model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")
columns = joblib.load("ml_model/columns.pkl")

# ✅ CORRECT labels (from your model)
labels = ['covid','Dengue', 'Flu', 'Measles']

# -----------------------------
# TITLE
# -----------------------------
st.title("🏥 Disease Prediction System")
st.write("Enter patient symptoms to predict disease or healthy status")

# -----------------------------
# YES/NO FUNCTION
# -----------------------------
def yes_no(val):
    return 1 if val == "Yes" else 0

# -----------------------------
# SYMPTOMS INPUT
# -----------------------------
st.subheader("Symptoms")

fever = st.selectbox("Fever", ["No", "Yes"])
cough = st.selectbox("Cough", ["No", "Yes"])
rash = st.selectbox("Rash", ["No", "Yes"])
headache = st.selectbox("Headache", ["No", "Yes"])
vomiting = st.selectbox("Vomiting", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
sore_throat = st.selectbox("Sore Throat", ["No", "Yes"])
breathing_issue = st.selectbox("Breathing Issue", ["No", "Yes"])

# -----------------------------
# PATIENT DETAILS
# -----------------------------
st.subheader("Patient Details")

age = st.slider("Age", 0, 100, 25)
temperature = st.slider("Temperature (°C)", 30.0, 45.0, 37.0)
humidity = st.slider("Humidity (%)", 0, 100, 50)
days = st.slider("Days of Symptoms", 0, 30, 3)

travel = st.selectbox("Travel History", ["No", "Yes"])
gender = st.selectbox("Gender", ["Male", "Female"])
severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])

# -----------------------------
# CONVERT INPUT
# -----------------------------
input_data = {
    "fever": yes_no(fever),
    "cough": yes_no(cough),
    "rash": yes_no(rash),
    "headache": yes_no(headache),
    "vomiting": yes_no(vomiting),
    "fatigue": yes_no(fatigue),
    "sore_throat": yes_no(sore_throat),
    "breathing_issue": yes_no(breathing_issue),
    "age": age,
    "temperature": temperature,
    "humidity_level": humidity,
    "days_symptoms": days,
    "travel_history": yes_no(travel),
    "gender": gender,
    "severity": severity
}

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Disease"):

    df = pd.DataFrame([input_data])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=columns, fill_value=0)

    # -----------------------------
    # 🟢 HEALTHY CHECK (all symptoms = 0)
    # -----------------------------
    symptoms = [
        input_data["fever"],
        input_data["cough"],
        input_data["rash"],
        input_data["headache"],
        input_data["vomiting"],
        input_data["fatigue"],
        input_data["sore_throat"],
        input_data["breathing_issue"]
    ]

    if sum(symptoms) == 0:
        st.success("🟢 Prediction: Healthy / No Disease")

    else:
        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]

        # Confidence
        proba = model.predict_proba(df_scaled)[0]
        confidence = round(max(proba) * 100, 2)

        # Convert to disease name
        disease = labels[int(prediction)]

        # -----------------------------
        # 🟢 LOW CONFIDENCE CHECK
        # -----------------------------
        if confidence < 60:
            st.success("🟢 Prediction: Healthy / No Disease")
            st.info(f"📊 Confidence: {confidence}% (Low confidence)")
        else:
            st.success(f"🧠 Prediction: {disease}")
            st.info(f"📊 Confidence: {confidence}%")