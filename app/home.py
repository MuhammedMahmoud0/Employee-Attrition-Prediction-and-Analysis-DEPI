import streamlit as st
import pandas as pd


def home_page():
    """Home page content."""
    st.title("Employee Attrition Classification 💼")
    st.markdown("---")
    
    st.header("Objective", divider="gray")
    st.write("Our objective in this project is to make analysis about employee attrition and make predictions based on the data")

    st.header("Data", divider="gray")
    try:
        df = pd.read_csv("../data/Faker_Data/synthetic_hr_dataset.csv")
        st.dataframe(df)
    except FileNotFoundError:
        st.warning("Data file not found. Please ensure synthetic_hr_dataset.csv exists.")