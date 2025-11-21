import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the pre-trained model
MODEL_PATH = "D:\FCDS\DEPI\\final project\Employee-Attrition-Prediction-and-Analysis-DEPI-main\models\model.pkl"

@st.cache_resource
def load_model():
    """Load the pre-trained pickle model."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error(f"Model file not found at {MODEL_PATH}")
        return None

# Helper to calculate derived features
def calculate_age_group(age: int) -> str:
    """Calculates the age group string based on age."""
    if age < 26:
        return "18-25"
    elif age < 36:
        return "26-35"
    elif age < 46:
        return "36-45"
    elif age < 56:
        return "46-55"
    else:
        return "56-65"

def model_page():
    """Prediction Model page with direct pickle model inference."""
    st.title("Prediction Model 💻")
    st.markdown("---")

    st.header("Predict Employee Attrition Risk")
    st.write("1 = Stayed | 0 = Left")

    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the prediction model. Please check the model file path.")
        return

    st.subheader("Predict for a Single Employee")
    st.write("Fill in the employee details below. Auto-calculated: Employee ID, Age Group, Career Start Age.")

    # Define all 21 user-input features + 3 auto-calculated = 24 total
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (Years)", 18, 65, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's Degree", "Associate Degree", "Master's Degree", "PhD"])
        number_of_dependents = st.number_input("Number of Dependents", 0, 10, 2)
    
    with col2:
        monthly_income = st.number_input("Monthly Income ($)", 1000.0, 20000.0, 4500.0, step=100.0)
        years_at_company = st.number_input("Years at Company", 0, 30, 5)
        number_of_promotions = st.number_input("Number of Promotions", 0, 15, 1)
        distance_from_home = st.number_input("Distance from Home (km)", 1, 100, 25)
        job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
    
    with col3:
        job_role = st.selectbox("Job Role", ["Finance", "Media", "Education", "Technology", "Healthcare"])
        job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
        performance_rating = st.selectbox("Performance Rating", ["Low", "Average", "High", "Excellent"])
        work_life_balance = st.selectbox("Work Life Balance", ["Poor", "Fair", "Good", "Excellent"])
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

    col4, col5 = st.columns(2)
    
    with col4:
        overtime = st.radio("Overtime", ["Yes", "No"], index=1, horizontal=True)
        remote_work = st.radio("Remote Work", ["Yes", "No"], index=1, horizontal=True)
        company_reputation = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])
    
    with col5:
        leadership_opportunities = st.radio("Leadership Opportunities", ["Yes", "No"], index=1, horizontal=True)
        innovation_opportunities = st.radio("Innovation Opportunities", ["Yes", "No"], index=1, horizontal=True)
        employee_recognition = st.selectbox("Employee Recognition", ["Low", "Medium", "High", "Very High"])

    # Auto-calculated fields (hidden from user)
    employee_id = np.random.randint(100000, 999999)
    age_groups = calculate_age_group(age)
    age_before_working = age - years_at_company

    # Prediction button
    if st.button("Predict Attrition", type="primary", use_container_width=True):
        try:
            # Build the payload with all 24 features in correct order
            payload = {
                'employee_id': employee_id,
                'age': age,
                'gender': gender,
                'years_at_company': years_at_company,
                'job_role': job_role,
                'monthly_income': monthly_income,
                'work_life_balance': work_life_balance,
                'job_satisfaction': job_satisfaction,
                'performance_rating': performance_rating,
                'number_of_promotions': number_of_promotions,
                'overtime': overtime,
                'distance_from_home': distance_from_home,
                'education_level': education_level,
                'marital_status': marital_status,
                'number_of_dependents': number_of_dependents,
                'job_level': job_level,
                'company_size': company_size,
                'remote_work': remote_work,
                'leadership_opportunities': leadership_opportunities,
                'innovation_opportunities': innovation_opportunities,
                'company_reputation': company_reputation,
                'employee_recognition': employee_recognition,
                'age_groups': age_groups,
                'age_before_working': age_before_working
            }
            
            # Convert to DataFrame (model expects DataFrame input)
            input_df = pd.DataFrame([payload])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Map prediction to label (1 = Stayed, 0 = Left)
            result_label = "Stayed ✅" if prediction == 1 else "Left ⚠️"
            confidence = max(prediction_proba) * 100
            
            st.markdown("#### Prediction Result:")
            
            if prediction == 1:
                st.success(f"**{result_label}** ({confidence:.1f}% confidence)")
                st.write("This employee is predicted to stay with the company.")
            else:
                st.error(f"**{result_label}** ({confidence:.1f}% confidence)")
                st.write("This employee exhibits characteristics associated with attrition risk.")
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please ensure the model file exists and all inputs are valid.")

