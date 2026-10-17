
from pathlib import Path
import sys, json
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/"models"/"tourism_model.joblib"
st.set_page_config(page_title="Visit with Us | Wellness Tourism",page_icon="🌿",layout="wide")
st.title("🌿 Wellness Tourism Purchase Predictor")
st.write("Estimate the likelihood that a customer will purchase the Wellness Tourism Package. Use predictions to prioritize outreach—not to deny service.")
if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python src/train.py` and commit models/tourism_model.joblib.")
    st.stop()
model=joblib.load(MODEL_PATH)
with st.form("customer_form"):
    st.subheader("Customer profile")
    c1,c2,c3=st.columns(3)
    with c1:
        age=st.number_input("Age",18,100,35)
        contact=st.selectbox("Type of contact",["Self Enquiry","Company Invited"])
        city=st.selectbox("City tier",[1,2,3],index=1)
        occupation=st.selectbox("Occupation",["Salaried","Small Business","Large Business","Free Lancer"])
        gender=st.selectbox("Gender",["Male","Female"])
        marital=st.selectbox("Marital status",["Married","Single","Divorced"])
    with c2:
        people=st.number_input("Number of people visiting",1,10,2)
        children=st.number_input("Children below age 5",0,5,0)
        stars=st.selectbox("Preferred property stars",[3,4,5],index=1)
        trips=st.number_input("Trips per year",0,30,3)
        passport=st.selectbox("Has passport?",["Yes","No"])
        car=st.selectbox("Owns car?",["Yes","No"])
    with c3:
        designation=st.selectbox("Designation",["Executive","Manager","Senior Manager","AVP","VP"])
        income=st.number_input("Monthly income",0,1000000,25000,step=1000)
        pitch=st.selectbox("Product pitched",["Basic","Deluxe","Standard","Super Deluxe","King"])
        satisfaction=st.slider("Pitch satisfaction score",1,5,3)
        followups=st.number_input("Number of follow-ups",0,10,3)
        duration=st.number_input("Pitch duration (minutes)",0.0,120.0,10.0,step=1.0)
    submitted=st.form_submit_button("Predict purchase likelihood")
if submitted:
    record=pd.DataFrame([{
        "Age":age,"TypeofContact":contact,"CityTier":city,"Occupation":occupation,
        "Gender":gender,"NumberOfPersonVisiting":people,"PreferredPropertyStar":stars,
        "MaritalStatus":marital,"NumberOfTrips":trips,"Passport":1 if passport=="Yes" else 0,
        "OwnCar":1 if car=="Yes" else 0,"NumberOfChildrenVisiting":children,
        "Designation":designation,"MonthlyIncome":income,"PitchSatisfactionScore":satisfaction,
        "ProductPitched":pitch,"NumberOfFollowups":followups,"DurationOfPitch":duration
    }])
    try:
        prob=float(model.predict_proba(record)[0,1])
        st.metric("Estimated purchase probability",f"{prob:.1%}")
        st.progress(min(max(prob,0.0),1.0))
        if prob>=0.5: st.success("Higher predicted likelihood: consider a relevant, personalized follow-up.")
        else: st.info("Lower predicted likelihood at the default 0.50 threshold. Consider gathering more information or using a low-cost campaign.")
        st.caption("This is a model estimate, not a guarantee. Validate calibration and campaign ROI before operational use.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")