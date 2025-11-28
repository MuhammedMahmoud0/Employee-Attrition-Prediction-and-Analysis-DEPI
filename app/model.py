import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os


# FastAPI server URL
API_URL = os.getenv("API_URL", "http://localhost:8000")
# API_URL = "http://127.0.0.1:8000"


@st.cache_resource
def check_api_health():
    """Check if FastAPI server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


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
        return "55+"


def model_page():
    """Prediction Model page with FastAPI inference."""
    st.title("Prediction Model 💻")
    st.markdown("---")

    st.header("Predict Employee Attrition Risk")
    st.write("1 = Stayed | 0 = Left")

    # Check if FastAPI server is running
    if not check_api_health():
        st.error("⚠️ FastAPI server is not running!")
        st.info(
            "Please run the FastAPI server first using the terminal command provided in the instructions."
        )
        return

    st.subheader("Predict for a Single Employee")
    st.write(
        "Fill in the employee details below. Auto-calculated: Employee ID, Age Group, Career Start Age."
    )

    # Define all 21 user-input features + 3 auto-calculated = 24 total
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age (Years)", 18, 65, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )
        education_level = st.selectbox(
            "Education Level",
            [
                "Associate Degree",
                "Master’s Degree",
                "PhD",
                "Bachelor’s Degree",
                "High School",
            ],
        )
        number_of_dependents = st.number_input("Number of Dependents", 0, 10, 2)

    with col2:
        monthly_income = st.number_input(
            "Monthly Income ($)", 1000.0, 20000.0, 4500.0, step=100.0
        )
        years_at_company = st.number_input("Years at Company", 0, 30, 5)
        number_of_promotions = st.number_input("Number of Promotions", 0, 15, 1)
        distance_from_home = st.number_input("Distance from Home (km)", 1, 100, 25)
        job_satisfaction = st.selectbox(
            "Job Satisfaction", ["Low", "Medium", "High", "Very High"]
        )

    with col3:
        job_role = st.selectbox(
            "Job Role", ["Finance", "Media", "Education", "Technology", "Healthcare"]
        )
        job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
        performance_rating = st.selectbox(
            "Performance Rating", ["Low", "Average", "High", "Excellent"]
        )
        work_life_balance = st.selectbox(
            "Work Life Balance", ["Poor", "Fair", "Good", "Excellent"]
        )
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

    col4, col5 = st.columns(2)

    with col4:
        overtime = st.radio("Overtime", ["Yes", "No"], index=1, horizontal=True)
        remote_work = st.radio("Remote Work", ["Yes", "No"], index=1, horizontal=True)
        company_reputation = st.selectbox(
            "Company Reputation", ["Poor", "Fair", "Good", "Excellent"]
        )

    with col5:
        leadership_opportunities = st.radio(
            "Leadership Opportunities", ["Yes", "No"], index=1, horizontal=True
        )
        innovation_opportunities = st.radio(
            "Innovation Opportunities", ["Yes", "No"], index=1, horizontal=True
        )
        employee_recognition = st.selectbox(
            "Employee Recognition", ["Low", "Medium", "High", "Very High"]
        )

    # Auto-calculated fields (hidden from user)
    employee_id = np.random.randint(100000, 999999)
    age_groups = calculate_age_group(age)
    age_before_working = age - years_at_company

    # Prediction button
    if st.button("Predict Attrition", type="primary", use_container_width=True):
        try:
            # Build the payload with all 24 features in correct order
            payload = {
                "employee_id": employee_id,
                "age": age,
                "gender": gender,
                "years_at_company": years_at_company,
                "job_role": job_role,
                "monthly_income": monthly_income,
                "work_life_balance": work_life_balance,
                "job_satisfaction": job_satisfaction,
                "performance_rating": performance_rating,
                "number_of_promotions": number_of_promotions,
                "overtime": overtime,
                "distance_from_home": distance_from_home,
                "education_level": education_level,
                "marital_status": marital_status,
                "number_of_dependents": number_of_dependents,
                "job_level": job_level,
                "company_size": company_size,
                "remote_work": remote_work,
                "leadership_opportunities": leadership_opportunities,
                "innovation_opportunities": innovation_opportunities,
                "company_reputation": company_reputation,
                "employee_recognition": employee_recognition,
                "age_groups": age_groups,
                "age_before_working": age_before_working,
            }

            # Send request to FastAPI server
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = result["confidence"]
                prediction_label = result["prediction_label"]

                # Map prediction to label (1 = Stayed, 0 = Left)
                result_label = "Stayed ✅" if prediction == 1 else "Left ⚠️"

                st.markdown("#### Prediction Result:")

                if prediction == 1:
                    st.success(f"**{result_label}** ({confidence:.1f}% confidence)")
                    st.write("This employee is predicted to stay with the company.")
                else:
                    st.error(f"**{result_label}** ({confidence:.1f}% confidence)")
                    st.write(
                        "This employee exhibits characteristics associated with attrition risk."
                    )

                # Show additional details
                st.markdown("---")
                st.subheader("Prediction Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", prediction_label)
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to FastAPI server")
            st.info("Make sure the FastAPI server is running on http://127.0.0.1:8000")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
