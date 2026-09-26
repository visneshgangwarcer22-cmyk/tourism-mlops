
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
artifact = joblib.load(BASE / "model.joblib")
threshold = artifact["threshold"]
model = artifact["model"]
metadata = json.loads((BASE / "feature_metadata.json").read_text())

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Wellness Tourism Package Predictor")
st.write(
    "Enter customer and interaction details to estimate the likelihood "
    "of purchasing the Wellness Tourism Package."
)

def numeric_input(col, step=1.0):
    info = metadata["numeric"][col]
    mn, mx, med = info["min"], info["max"], info["median"]
    if col in ["Passport", "OwnCar"]:
        return st.selectbox(col, [0, 1], index=int(round(med)))
    if float(mn).is_integer() and float(mx).is_integer():
        return st.slider(col, int(mn), int(mx), int(round(med)), step=int(step))
    return st.number_input(col, min_value=float(mn), max_value=float(mx),
                           value=float(med), step=float(step))

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = numeric_input("Age")
        city_tier = numeric_input("CityTier")
        duration = numeric_input("DurationOfPitch")
        persons = numeric_input("NumberOfPersonVisiting")
        followups = numeric_input("NumberOfFollowups")
        hotel = numeric_input("PreferredPropertyStar")
        trips = numeric_input("NumberOfTrips")
    with c2:
        passport = numeric_input("Passport")
        pitch_score = numeric_input("PitchSatisfactionScore")
        own_car = numeric_input("OwnCar")
        children = numeric_input("NumberOfChildrenVisiting")
        income = numeric_input("MonthlyIncome", step=500)

    type_contact = st.selectbox("TypeofContact", metadata["categories"]["TypeofContact"])
    occupation = st.selectbox("Occupation", metadata["categories"]["Occupation"])
    gender = st.selectbox("Gender", metadata["categories"]["Gender"])
    product = st.selectbox("ProductPitched", metadata["categories"]["ProductPitched"])
    marital = st.selectbox("MaritalStatus", metadata["categories"]["MaritalStatus"])
    designation = st.selectbox("Designation", metadata["categories"]["Designation"])

    submitted = st.form_submit_button("Predict Purchase Probability")

if submitted:
    row = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": hotel,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income
    }])

    probability = float(model.predict_proba(row)[:, 1][0])
    prediction = int(probability >= threshold)

    st.metric("Purchase probability", f"{probability:.1%}")
    if prediction == 1:
        st.success("Predicted outcome: Potential buyer")
    else:
        st.info("Predicted outcome: Not predicted as a buyer")

    st.caption(
        f"Decision threshold used by the model: {threshold:.2f}. "
        "The threshold was selected on a validation set before final test evaluation."
    )
