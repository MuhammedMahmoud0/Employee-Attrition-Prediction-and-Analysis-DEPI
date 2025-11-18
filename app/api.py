import streamlit as st
import pandas as pd
import numpy as np
import io
import requests

# Define the URL for the external FastAPI prediction service
API_URL = "http://127.0.0.1:8000/predict"

# The full list of 24 features required by the external API service
REQUIRED_FEATURES = [
    'employee_id', 'age', 'gender', 'years_at_company', 'job_role', 
    'monthly_income', 'work_life_balance', 'job_satisfaction', 
    'performance_rating', 'number_of_promotions', 'overtime', 
    'distance_from_home', 'education_level', 'marital_status', 
    'number_of_dependents', 'job_level', 'company_size', 'remote_work', 
    'leadership_opportunities', 'innovation_opportunities', 
    'company_reputation', 'employee_recognition', 'age_groups', 
    'age_before_working'
]


def call_predict(payload: dict) -> dict:
    """
    Sends a single payload dict to the FastAPI `/predict` endpoint.

    - Validates that all required features are present and not null.
    - Raises RuntimeError if the FastAPI server is unreachable.
    - Uses `st.error(...)` and raises RuntimeError if response.status_code != 200.

    Returns the parsed JSON response as a dict.
    """
    # Validate payload keys
    missing = [k for k in REQUIRED_FEATURES if k not in payload]
    if missing:
        raise ValueError(f"Missing required features in payload: {', '.join(missing)}")

    # Ensure no NaN/None values
    nan_keys = [k for k, v in payload.items() if pd.isna(v)]
    if nan_keys:
        raise ValueError(f"Payload contains null values for: {', '.join(nan_keys)}")

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
    except requests.exceptions.RequestException:
        # Fatal: do not simulate
        raise RuntimeError("FastAPI model server unreachable. Please start the backend.")

    if response.status_code != 200:
        # Show the error then raise to stop execution as requested
        body = response.text or ''
        st.error("Prediction failed: " + body)
        raise RuntimeError(f"Prediction request failed with status {response.status_code}: {body}")

    # Parse JSON and return
    result = response.json()
    return result


@st.cache_data
def generate_mock_data():
    data = {
        'Age': np.random.randint(22, 60, 200),
        'Gender': np.random.choice(['Male', 'Female'], 200, p=[0.6, 0.4]),
        'MonthlyIncome': np.random.normal(6500, 3000, 200).round(2),
        'JobSatisfaction': np.random.randint(1, 5, 200),
        'YearsAtCompany': np.random.randint(1, 15, 200),
        'Attrition': np.random.choice(['Stayed', 'Left'], 200, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    df['MonthlyIncome'] = np.where(df['MonthlyIncome'] < 1500, 1500, df['MonthlyIncome'])

    df['attrition'] = df['Attrition'].map({'Stayed': 0, 'Left': 1})
    df['years_at_company'] = df['YearsAtCompany']
    df['age_groups'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], labels=["18-25", "26-35", "36-45", "46-55", "56-65"], right=False).astype(str)
    df['overtime'] = np.random.choice(['Yes', 'No'], 200, p=[0.3, 0.7])
    df['job_role'] = np.random.choice(["Education", "Technology", "Media", "Healthcare", "Finance"], 200)

    return df


@st.cache_data
def get_sample_csv_for_download():
    """
    Returns a small CSV template (optional) for users who want to see required columns.
    This is separate from the prediction-download feature which exports actual predictions.
    """
    # Provide a minimal template with headers matching REQUIRED_FEATURES
    df = pd.DataFrame(columns=REQUIRED_FEATURES)
    return df.to_csv(index=False).encode('utf-8')


def calculate_age_group(age):
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
