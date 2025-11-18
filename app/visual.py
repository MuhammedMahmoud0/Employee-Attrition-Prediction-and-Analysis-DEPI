import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from api import generate_mock_data

plt.style.use('ggplot')


def visualization_page():
    """Visualization page content."""
    st.title("Data Visualization 📈")
    st.markdown("---")

    df = generate_mock_data()

    attrition_summary = df.groupby('years_at_company')['attrition'].agg(['count', 'sum']).reset_index()
    attrition_summary.columns = ['years_at_company', 'total_employees', 'attrition_count']
    attrition_summary['attrition_rate'] = attrition_summary['attrition_count'] / attrition_summary['total_employees']
    df_attr = attrition_summary[['years_at_company', 'attrition_rate']]

    st.header("Attrtion Distribution")
    attrition_distribution = plt.figure(figsize=(9, 7))
    sns.countplot(x="attrition", data=df, palette="viridis")
    plt.title("Employee Attrition Count (0=Stayed, 1=Left)")
    plt.xlabel("Attrition Status")
    plt.ylabel("Count")
    st.pyplot(attrition_distribution)

    st.markdown("---")

    st.header("Attrition Rate by Years at Company")
    attr_rate_by_years = plt.figure(figsize=(10,5))
    sns.lineplot(x="years_at_company", y="attrition_rate", data=df_attr, marker='o', color='red')
    plt.title("Attrition Rate by Years at Company")
    plt.xlabel("Years at Company")
    plt.ylabel("Attrition Rate")
    st.pyplot(attr_rate_by_years)

    st.markdown("---")

    st.header("Years at Company Distribution")
    years_at_company_dist = plt.figure(figsize=(9, 7))
    bins = st.slider("Select Number of Bins for Histogram", 1, 50, value=15)
    sns.histplot(df["years_at_company"], bins=bins, kde=True, color="#0099ff")
    plt.title("Distribution of Years at Company")
    plt.xlabel("Years at Company")
    plt.ylabel("Frequency")
    st.pyplot(years_at_company_dist)

    st.markdown("---")

    st.header("Job Role Distribution by Overtime Status")
    jobRole_overtime = df.groupby(["overtime", "job_role"])['Age'].count().reset_index()
    jobRole_overtime = jobRole_overtime.rename(columns={"Age": "count"})

    overtime_donat = px.sunburst(
        jobRole_overtime,
        path=["overtime", "job_role"],
        values="count",
        title="Job Role Distribution by Overtime",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    overtime_donat.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(overtime_donat, use_container_width=True)

    st.markdown("---")

    st.header("Age Group Distribution")
    age_groups_pie = px.pie(
        df,
        names="age_groups",
        title="Employee Count by Age Groups",
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    age_groups_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(age_groups_pie, use_container_width=True)
