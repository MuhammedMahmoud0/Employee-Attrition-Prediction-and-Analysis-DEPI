import streamlit as st
from home import home_page
from visual import visualization_page
from dashboard import dashboard_page
from model import model_page


# MUST be the first Streamlit command - call it before importing page functions
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import page functions after set_page_config


# Simple sidebar navigation
pages = {
    "Home": home_page,
    "Visualization": visualization_page,
    "Dashboard (External)": dashboard_page,
    "Prediction Model": model_page,
}

choice = st.sidebar.radio("Navigate", list(pages.keys()), index=0)
# Call the selected page function
pages[choice]()
