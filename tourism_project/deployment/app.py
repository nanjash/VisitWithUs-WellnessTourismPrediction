import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package
based on their personal details and sales interaction history.
Enter the customer information below to get a prediction.
""")

# Collect inputs
age = st.number_input("Age", 18, 100, 30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_visitors = st.number_input("Number of Persons Visiting", 1, 10, 2)
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", 0, 20, 2)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 1000, 100000, 20000)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Premium"])
num_followups = st.number_input("Number of Follow-ups", 0, 10, 2)
pitch_duration = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)

# Prepare input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_visitors,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": pitch_duration
}])

# Prediction
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The customer is {result}.")
    else:
        st.warning(f"The customer is {result}.")
