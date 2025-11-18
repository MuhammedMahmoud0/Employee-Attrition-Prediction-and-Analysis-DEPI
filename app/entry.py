import streamlit as st
from home import home_page
from visual import visualization_page
from dashboard import dashboard_page
from model import model_page


def run_app():
    st.set_page_config(
        page_title="HR Attrition Predictor",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    pages = [
        st.Page(home_page, icon="🏠", title="Home"),
        st.Page(visualization_page, icon="📈", title="Visualization"),
        st.Page(dashboard_page, icon="📊", title="Dashboard (External)"),
        st.Page(model_page, icon="💻", title="Prediction Model"),
    ]

    pg = st.navigation(pages)
    pg.run()


if __name__ == '__main__':
    run_app()
