import streamlit as st
from api import generate_mock_data


def home_page():
    """Home page content."""
    st.title("Employee Attrition Classification 💼")
    st.markdown("---")

    st.header("Objective")
    st.info("Our objective in this project is to perform analysis about employee attrition and make predictions based on key HR data, helping management identify risk factors.")

    st.header("Data Sample (Synthetic)", divider="gray")
    df = generate_mock_data()
    st.dataframe(df[['Age', 'Gender', 'MonthlyIncome', 'YearsAtCompany', 'Attrition']].head(10), use_container_width=True)
    st.caption("This table shows a synthetic HR dataset used for analysis and prediction.")
