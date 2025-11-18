import streamlit as st


def dashboard_page():
    """Dashboard page content with Power BI iframe."""
    st.title("External Dashboard 📊")
    st.markdown("---")

    st.markdown("This page embeds an external dashboard for deeper analysis. If the URL below is not working, it may be due to security restrictions in the environment.")

    POWER_BI_URL = (
        "https://app.powerbi.com/view?r=eyJrIjoiNjFkMTBjMDktZTI2Ni00YTE5LWI2OTktOTRlMWZjZmY4NWI2IiwidCI6ImVhZjYyNGM4LWEwYzQtNDE5NS04N2QyLTQ0M2U1ZDc1MTZjZCIsImMiOjh9"
    )

    import streamlit.components.v1 as components
    st.header("Embedded Power BI Report")
    components.iframe(POWER_BI_URL, height=650)
