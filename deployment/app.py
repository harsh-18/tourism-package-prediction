import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Page config 
st.set_page_config(
    page_title="Tourism Wellness Predictor",
    page_icon="🌿",
    layout="centered"
)

# Loading model from HF Hub 
@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="harshverma27/tourism-wellness-model",
        filename="best_model.pkl"
    )
    with open(path, "rb") as f:
        return joblib.load(path)

model = load_model()

#  UI 
st.title(" Tourism Product Predictor")
st.markdown("Fill in the details below to predict if a customer will take the product.")

st.divider()

# Input Fields (matched to train.csv columns) 
age                      = st.slider("Age", 18, 80, 30)
monthly_income           = st.number_input("Monthly Income (₹)", min_value=0, value=20000, step=1000)
duration_of_pitch        = st.slider("Duration of Pitch (mins)", 0, 60, 15)
num_person_visiting      = st.slider("Number of Persons Visiting", 1, 10, 2)
num_children_visiting    = st.slider("Number of Children Visiting", 0, 5, 0)
num_followups            = st.slider("Number of Followups", 1, 6, 3)
num_trips                = st.slider("Number of Trips", 0, 20, 2)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
preferred_property_star  = st.selectbox("Preferred Property Star", [3, 4, 5])
city_tier                = st.selectbox("City Tier", [1, 2, 3])
passport                 = st.selectbox("Passport", [0, 1])
own_car                  = st.selectbox("Own Car", [0, 1])

type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation      = st.selectbox("Occupation", ["Salaried", "Self Employed", "Free Lancer"])
gender          = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation     = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

st.divider()

# Predict
if st.button("🔍 Predict", use_container_width=True):
    from sklearn.preprocessing import LabelEncoder
    
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    # Encode categorical columns same as training
    cat_cols = ["TypeofContact", "Occupation", "Gender", 
                "ProductPitched", "MaritalStatus", "Designation"]
    
    le = LabelEncoder()
    for col in cat_cols:
        input_df[col] = le.fit_transform(input_df[col])

    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f" Customer likely to take the product!  (Confidence: {proba:.1%})")
    else:
        st.error(f" Customer unlikely to take the product  (Confidence: {1 - proba:.1%})")
