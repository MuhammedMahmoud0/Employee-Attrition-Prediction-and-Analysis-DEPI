import streamlit as st
import pandas as pd
import os

def home_page():
    """Home page content."""
    st.title("Employee Attrition Classification 💼")
    st.markdown("---")
    
    st.header("Objective", divider="gray")
    st.write("Our objective in this project is to make analysis about employee attrition and make predictions based on the data")

    st.header("Data", divider="gray")
    try:
        base = os.path.dirname(__file__)
        path = os.path.join(base, "..", "data", "Faker_Data", "synthetic_hr_dataset.csv")
        path = os.path.abspath(path)
        df = pd.read_csv(path)
        st.dataframe(df)
    except FileNotFoundError:
        st.warning("Data file not found. Please ensure synthetic_hr_dataset.csv exists.")