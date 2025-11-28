import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def visualization_page():
    """Visualization page content."""
    st.title("Data Visualization 📈")
    st.markdown("---")
    
    try:
        # Read the data
        base = os.path.dirname(__file__)
        path = os.path.join(base, "..", "data", "Faker_Data", "Preprocessed_Data", "preprocessed_data.csv")
        path = os.path.abspath(path)
        df = pd.read_csv(path)
        
        # Attrition Distribution
        st.header("Attrition Distribution")
        attrition_distribution = plt.figure(figsize=(9, 7))
        sns.countplot(x="attrition", data=df)
        st.pyplot(attrition_distribution)
        
        st.header("Attrition Rate by Years at Company")
        attr_rate_by_years = plt.figure(figsize=(10, 5))
        try:
            df_attr_path = os.path.join(base, "df_attr.csv")
            df_attr = pd.read_csv(df_attr_path)
            sns.lineplot(x="years_at_company", y="attrition_rate", data=df_attr, marker='o')
            plt.title("Attrition Rate by Years at Company")
            st.pyplot(attr_rate_by_years)
        except FileNotFoundError:
            st.warning("df_attr.csv not found. Skipping attrition rate chart.")
        
        # Years at company Distribution
        st.header("Years at Company Distribution")
        years_at_company_dist = plt.figure(figsize=(9, 7))
        bins = st.slider("Bins", 1, 50, value=20)
        sns.histplot(df["years_at_company"], bins=bins)
        st.pyplot(years_at_company_dist)
        
        # Overtime donut chart
        st.header("Overtime based on Job Role")
        jobRole_overtime = df.groupby(["overtime", "job_role"])["employee_id"].count().reset_index()
        jobRole_overtime = jobRole_overtime.rename(columns={"employee_id": "count"})
        overtime_donat = px.sunburst(
            jobRole_overtime,
            path=["overtime", "job_role"],
            values="count",
            title="Job Role Distribution by Overtime"
        )
        st.plotly_chart(overtime_donat)
        
        # Age groups Pie plot
        st.header("Age Groups Distribution")
        age_groups_pie = px.pie(df, names="age_groups", title="Age Groups")
        st.plotly_chart(age_groups_pie)
    except Exception as e:
        st.error(f"Error loading visualization data: {e}")
