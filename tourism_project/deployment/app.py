
import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# -> model lives on HF Hub, pulled at runtime so the app always uses latest version
HF_USERNAME = "harshverma27"
model_repo  = f"{HF_USERNAME}/tourism-wellness-model"

@st.cache_resource
def load_model():
    # -> cache the model so it only downloads once per session
    path = hf_hub_download(
        repo_id=model_repo,
        filename="best_model.pkl",
        repo_type="model",
        token=os.environ.get("HF_TOKEN")
    )
    return joblib.load(path)

model = load_model()

st.title("Wellness Tourism Package -- Purchase Predictor")
st.write("Fill in the customer details below and hit Predict.")

col1, col2 = st.columns(2)

with col1:
    age                     = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier               = st.selectbox("City Tier", [1, 2, 3])
    monthly_income          = st.number_input("Monthly Income (Rs)", min_value=0, value=50000, step=1000)
    number_of_trips         = st.slider("Avg Trips Per Year", 0, 20, 3)
    passport                = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car                 = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    number_persons_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
    preferred_star          = st.selectbox("Preferred Hotel Stars", [3, 4, 5])

with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    occupation      = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender          = st.selectbox("Gender", ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation     = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Standard"])
    number_children = st.number_input("Children (under 5) Visiting", min_value=0, max_value=5, value=0)
    pitch_score     = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    followups       = st.number_input("Number of Follow-Ups", min_value=0, max_value=10, value=2)
    pitch_duration  = st.number_input("Pitch Duration (mins)", min_value=0, max_value=120, value=20)

# -> encoding maps match alphabetical order used by LabelEncoder during training
contact_map = {"Company Invited": 0, "Self Enquiry": 1}
occ_map     = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_map  = {"Female": 0, "Male": 1}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
desig_map   = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
product_map = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}

if st.button("Predict"):
    # -> assemble input in the exact column order the model was trained on
    input_data = pd.DataFrame([{
        "Age":                      age,
        "TypeofContact":            contact_map[type_of_contact],
        "CityTier":                 city_tier,
        "DurationOfPitch":          pitch_duration,
        "Occupation":               occ_map[occupation],
        "Gender":                   gender_map[gender],
        "NumberOfPersonVisiting":   number_persons_visiting,
        "NumberOfFollowups":        followups,
        "ProductPitched":           product_map[product_pitched],
        "PreferredPropertyStar":    preferred_star,
        "MaritalStatus":            marital_map[marital_status],
        "NumberOfTrips":            number_of_trips,
        "Passport":                 passport,
        "PitchSatisfactionScore":   pitch_score,
        "OwnCar":                   own_car,
        "NumberOfChildrenVisiting": number_children,
        "Designation":              desig_map[designation],
        "MonthlyIncome":            monthly_income,
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.divider()
    if pred == 1:
        st.success(f"This customer is likely to purchase the Wellness Package ({prob*100:.1f}% probability)")
    else:
        st.error(f"This customer is unlikely to purchase the Wellness Package ({prob*100:.1f}% probability)")
