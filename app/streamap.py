import streamlit as st
import pandas as pd
import numpy as np
import io
    
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib # Added joblib for loading the ML model pipeline

# --- Configuration ---
# Setting a default plot style for Matplotlib
plt.style.use('ggplot')

# --- Utility Functions (Model Loading and Prediction) ---

# The full list of 24 features required by the updated model pipeline 
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

@st.cache_resource
def load_model_pipeline():
    """
    Attempts to load the ML model pipeline using joblib.
    We use 'models/model.pkl' as a common relative path for resources.
    This function is cached to run only once.
    """
    try:
        # Attempt to load the model file as requested by the user
        model = joblib.load("../models/model.pkl") 
        st.success("Successfully loaded ML model pipeline from 'models/model.pkl'.")
        return model, True
    except FileNotFoundError:
        st.warning("ML Model file not found at 'models/model.pkl'.")
    except Exception as e:
        st.warning(f"Error loading ML model (might be corrupted or incompatible): {e}")
        
    return None, False

def predict_attrition(df: pd.DataFrame) -> np.ndarray:
    """
    Attempts to predict using the loaded ML model. If the model is not loaded,
    it falls back to the robust, self-contained business rules.
    """
    model, is_loaded = load_model_pipeline()
    
    if is_loaded and model is not None:
        st.info("Using loaded **ML Model Pipeline** for prediction.")
        
        # --- ML MODEL PREDICTION ---
        try:
             # The real ML model pipeline is assumed to take the input DataFrame 
             # (with all required 24 features) and return predictions.
             predictions = model.predict(df) 
             return predictions
        except Exception as e:
             st.error(f"Error running prediction with loaded ML model: {e}. Falling back to simulation.")
             # If the ML model fails, we fall through to the simulation logic
    
    # --- FALLBACK: ROBUST SIMULATION LOGIC ---
    st.info("ML model unavailable. Using **robust, self-contained business rules** for prediction results.")

    df = df.copy() 
    df['Prediction_Int'] = 0  # Default to 0 (Stayed)

    # Internal mappings for simulation logic
    js_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    pr_map = {'Low': 1, 'Average': 2, 'High': 3, 'Excellent': 4}

    # Ensure required columns for the enhanced mock logic exist
    required_mock_cols = ['job_satisfaction', 'age', 'monthly_income', 'overtime', 'distance_from_home', 'performance_rating', 'years_at_company']

    # We proceed if ALL required columns are present in the dataframe
    if all(col in df.columns for col in required_mock_cols):
        
        # Map string ratings to numeric values for calculation
        df['JobSatisfaction_Num'] = df['job_satisfaction'].map(js_map).fillna(3)
        df['PerformanceRating_Num'] = df['performance_rating'].map(pr_map).fillna(2)
        
        # --- ENHANCED SIMULATION RULES (Deterministic Prediction) ---

        # Rule 1: Young, Dissatisfied, and Overworked (Very High Risk)
        # Attrition if: Age < 30 AND Job Satisfaction is Low/Medium (<= 2) AND Overtime is Yes
        condition_1 = (df['age'] < 30) & \
                      (df['JobSatisfaction_Num'] <= 2) & \
                      (df['overtime'] == 'Yes')
        df.loc[condition_1, 'Prediction_Int'] = 1

        # Rule 2: Older, Low Income, and Long Commute (High Risk)
        # Attrition if: Age > 50 AND Monthly Income < $3500 AND Distance from Home > 40 miles
        condition_2 = (df['monthly_income'] < 3500) & \
                      (df['age'] > 50) & \
                      (df['distance_from_home'] > 40)
        df.loc[condition_2, 'Prediction_Int'] = 1
        
        # Rule 3: Mid-Career Plateau (High Risk)
        # Attrition if: Age between 30 and 45 AND Low Performance (1) AND High Tenure (> 10 years)
        condition_3 = (df['age'].between(30, 45, inclusive='both')) & \
                      (df['PerformanceRating_Num'] == 1) & \
                      (df['years_at_company'] > 10)
        df.loc[condition_3, 'Prediction_Int'] = 1

    else:
        # Fallback for batches missing data
        df['Prediction_Int'] = 0

    return df['Prediction_Int'].values

@st.cache_data
def generate_mock_data():
    """Generates synthetic HR data for display and visualization."""
    data = {
        'Age': np.random.randint(22, 60, 200),
        'Gender': np.random.choice(['Male', 'Female'], 200, p=[0.6, 0.4]),
        'MonthlyIncome': np.random.normal(6500, 3000, 200).round(2),
        'JobSatisfaction': np.random.randint(1, 5, 200), # 1=Low, 4=Very High (for Home/Viz page)
        'YearsAtCompany': np.random.randint(1, 15, 200),
        'Attrition': np.random.choice(['Stayed', 'Left'], 200, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    df['MonthlyIncome'] = np.where(df['MonthlyIncome'] < 1500, 1500, df['MonthlyIncome'])
    
    # Add fields needed for the new visualizations using mock data logic
    df['attrition'] = df['Attrition'].map({'Stayed': 0, 'Left': 1})
    df['years_at_company'] = df['YearsAtCompany']
    df['age_groups'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], labels=["18-25", "26-35", "36-45", "46-55", "56-65"], right=False).astype(str)
    df['overtime'] = np.random.choice(['Yes', 'No'], 200, p=[0.3, 0.7])
    df['job_role'] = np.random.choice(["Education", "Technology", "Media", "Healthcare", "Finance"], 200)

    return df

# --- Sample Data (Full 24 columns derived from user's input) ---
SAMPLE_CSV_DATA_STRING = """
employee_id,age,gender,years_at_company,job_role,monthly_income,work_life_balance,job_satisfaction,performance_rating,number_of_promotions,overtime,distance_from_home,education_level,marital_status,number_of_dependents,job_level,company_size,remote_work,leadership_opportunities,innovation_opportunities,company_reputation,employee_recognition,age_groups,age_before_working
1,23,Female,1,Education,2794,Excellent,High,Average,0,No,57,High School,Married,2,Entry,Medium,No,No,No,Good,Very High,18-25,22
2,47,Male,2,Technology,4811,Excellent,Very High,High,0,Yes,11,High School,Married,2,Entry,Large,Yes,No,Yes,Excellent,Very High,46-55,45
3,36,Male,6,Media,5725,Poor,Low,Low,2,Yes,46,Master’s Degree,Married,5,Mid,Medium,Yes,No,No,Fair,Low,36-45,30
4,25,Male,3,Technology,5348,Good,Very High,High,1,No,14,High School,Single,0,Entry,Large,No,No,Yes,Excellent,High,18-25,22
5,38,Female,12,Healthcare,4094,Poor,High,High,0,Yes,22,High School,Married,0,Entry,Medium,No,No,No,Good,Medium,36-45,26
"""

@st.cache_data
def get_sample_csv_for_download():
    """
    Creates a downloadable CSV template containing all 24 required features.
    """
    df = pd.read_csv(io.StringIO(SAMPLE_CSV_DATA_STRING))
    # Ensure the column names match the required lowercase format
    df.columns = [col.lower() for col in df.columns]
    
    return df.to_csv(index=False).encode('utf-8')


# --- Page Definitions ---

def home_page():
    """Home page content."""
    st.title("Employee Attrition Classification 💼")
    st.markdown("---")
    
    st.header("Objective")
    st.info("Our objective in this project is to perform analysis about employee attrition and make predictions based on key HR data, helping management identify risk factors.")

    st.header("Data Sample (Synthetic)", divider= "gray")
    df = generate_mock_data()
    # Display only core columns for simplicity on the home page
    st.dataframe(df[['Age', 'Gender', 'MonthlyIncome', 'YearsAtCompany', 'Attrition']].head(10), use_container_width=True)
    st.caption("This table shows a synthetic HR dataset used for analysis and prediction.")

def visualization_page():
    """Visualization page content."""
    st.title("Data Visualization 📈")
    st.markdown("---")
    
    df = generate_mock_data()
    
    # --- Data Preparation for Visualization ---
    
    # 1. Attrition Rate by Years at Company (Generating df_attr mock)
    attrition_summary = df.groupby('years_at_company')['attrition'].agg(['count', 'sum']).reset_index()
    attrition_summary.columns = ['years_at_company', 'total_employees', 'attrition_count']
    attrition_summary['attrition_rate'] = attrition_summary['attrition_count'] / attrition_summary['total_employees']
    df_attr = attrition_summary[['years_at_company', 'attrition_rate']]


    # --- Attrition Distribution ---
    st.header("Attrtion Distribution")
    attrition_distribution = plt.figure(figsize= (9, 7))
    sns.countplot(x="attrition", data=df, palette="viridis")
    plt.title("Employee Attrition Count (0=Stayed, 1=Left)")
    plt.xlabel("Attrition Status")
    plt.ylabel("Count")
    st.pyplot(attrition_distribution)
    
    st.markdown("---")

    # --- Attrition Rate by Years at Company ---
    st.header("Attrition Rate by Years at Company")
    attr_rate_by_years = plt.figure(figsize=(10,5))
    sns.lineplot(x="years_at_company", y="attrition_rate", data=df_attr, marker='o', color='red')
    plt.title("Attrition Rate by Years at Company")
    plt.xlabel("Years at Company")
    plt.ylabel("Attrition Rate")
    st.pyplot(attr_rate_by_years)
    
    st.markdown("---")

    # --- Years at Company Distribution (with Slider) ---
    st.header("Years at Company Distribution")
    years_at_company_dist = plt.figure(figsize= (9, 7))
    bins = st.slider("Select Number of Bins for Histogram", 1, 50, value= 15) # Adjusted default for better distribution
    sns.histplot(df["years_at_company"], bins=bins, kde=True, color="#0099ff")
    plt.title("Distribution of Years at Company")
    plt.xlabel("Years at Company")
    plt.ylabel("Frequency")
    st.pyplot(years_at_company_dist)
    
    st.markdown("---")

    # --- Overtime Donut Chart (Plotly Sunburst/Donut) ---
    st.header("Job Role Distribution by Overtime Status")
    jobRole_overtime = df.groupby(["overtime", "job_role"])["Age"].count().reset_index()
    jobRole_overtime = jobRole_overtime.rename(columns={"Age": "count"})
    
    overtime_donat = px.sunburst(
        jobRole_overtime,
        path=["overtime", "job_role"],  # hierarchy levels
        values="count", 
        title="Job Role Distribution by Overtime",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    overtime_donat.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(overtime_donat, use_container_width=True)

    st.markdown("---")

    # --- Age Groups Pie Plot (Plotly) ---
    st.header("Age Group Distribution")
    age_groups_pie = px.pie(
        df, 
        names="age_groups", 
        title="Employee Count by Age Groups",
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    age_groups_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(age_groups_pie, use_container_width=True)


def dashboard_page():
    """Dashboard page content with Power BI iframe."""
    st.title("External Dashboard 📊")
    st.markdown("---")

    st.markdown("This page embeds an external dashboard for deeper analysis. If the URL below is not working, it may be due to security restrictions in the environment.")

    # Your original Power BI URL
    POWER_BI_URL = (
        "https://app.powerbi.com/view?r=eyJrIjoiNjFkMTBjMDktZTI2Ni00YTE5LWI2OTktOTRlMWZjZmY4NWI2IiwidCI6ImVhZjYyNGM4LWEwYzQtNDE5NS04N2QyLTQ0M2U1ZDc1MTZjZCIsImMiOjh9"
    )

    import streamlit.components.v1 as components
    st.header("Embedded Power BI Report")
    try:
        # Increased height for better viewing
        components.iframe(POWER_BI_URL, height=650)
    except Exception as e:
        st.error(f"Could not load the iframe content. Error: {e}")
        st.code(POWER_BI_URL)


# Helper to calculate age group string
def calculate_age_group(age):
    """Calculates the age group string based on age."""
    if age < 26: return "18-25"
    elif age < 36: return "26-35"
    elif age < 46: return "36-45"
    elif age < 56: return "46-55"
    else: return "56-65"

def model_page():
    """Prediction Model page content with two prediction options."""
    st.title("Prediction Model 💻")
    st.markdown("---")

    st.header("Predict Employee Attrition Risk")
    st.write(f"The model pipeline requires **all {len(REQUIRED_FEATURES)} features** shown below to generate a prediction.")
    st.write(f"*(Note: Employee ID, Age Group, and Career Start Age are handled automatically by the code.)*")


    # Use tabs for clear separation of prediction methods
    tab1, tab2 = st.tabs(["Single Row Prediction", "CSV File Prediction"])

    with tab1:
        st.subheader("Predict for a Single Employee")
        
        # --- HIDDEN FIELDS: Default/Derived values for API Payload ---
        employee_id = np.random.randint(100000, 999999) # Automatically generate a random ID
        age_before_working = 22 # Use a sensible default
        # --- Inputs Grouped by Expander ---

        # 1. Core Demographics and Compensation (Including the original 3 features)
        with st.expander("Demographics and Compensation", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age (Years)", 18, 65, 35)
            with col2:
                gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
            with col3:
                marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"], index=1)
            
            col4, col5, col6 = st.columns(3)
            with col4:
                monthly_income = st.number_input("Monthly Income ($)", 1000.00, 20000.00, 4500.00, step=100.00, format="%.2f")
            with col5:
                number_of_dependents = st.number_input("Number of Dependents", 0, 10, 2)
            with col6:
                education_level = st.selectbox("Education Level", options=["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"], index=2)
            
            # --- Hidden Age Group Calculation ---
            age_groups = calculate_age_group(age)
            # st.markdown(f"**Calculated Age Group (Hidden):** {age_groups}") # Debug: Uncomment to see hidden value

        # 2. Job and Performance Details
        with st.expander("Job, Performance, and Tenure", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                job_role = st.selectbox("Job Role", options=["Education", "Technology", "Media", "Healthcare", "Finance"], index=2)
            with col_b:
                job_level = st.selectbox("Job Level", options=["Entry", "Mid", "Senior"], index=0)
            with col_c:
                years_at_company = st.number_input("Years at Company", 0, 30, 5)
            
            col_d, col_e, col_f = st.columns(3)
            with col_d:
                number_of_promotions = st.number_input("Number of Promotions", 0, 15, 1)
            with col_e:
                performance_rating = st.selectbox("Performance Rating", options=["Low", "Average", "High", "Excellent"], index=1)
            with col_f:
                # Job Satisfaction uses string labels now (Low, Medium, High, Very High)
                job_satisfaction = st.selectbox("Job Satisfaction", options=["Low", "Medium", "High", "Very High"], index=2)

        # 3. Work Environment and Culture
        with st.expander("Work Environment and Culture", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                company_size = st.selectbox("Company Size", options=["Small", "Medium", "Large"], index=1)
            with col_b:
                remote_work = st.radio("Remote Work", options=["Yes", "No"], index=1, horizontal=True)
            with col_c:
                overtime = st.radio("Overtime", options=["Yes", "No"], index=1, horizontal=True)
            
            col_d, col_e, col_f = st.columns(3)
            with col_d:
                distance_from_home = st.number_input("Distance from Home (Miles)", 1, 60, 25)
            with col_e:
                work_life_balance = st.selectbox("Work Life Balance", options=["Poor", "Fair", "Good", "Excellent"], index=2)
            with col_f:
                company_reputation = st.selectbox("Company Reputation", options=["Poor", "Fair", "Good", "Excellent"], index=2)

            col_g, col_h, col_i = st.columns(3)
            with col_g:
                leadership_opportunities = st.radio("Leadership Opportunities", options=["Yes", "No"], index=1, horizontal=True)
            with col_h:
                innovation_opportunities = st.radio("Innovation Opportunities", options=["Yes", "No"], index=1, horizontal=True)
            with col_i:
                employee_recognition = st.selectbox("Employee Recognition", options=["Low", "Medium", "High", "Very High"], index=1)

                
        # --- Prediction Button and Logic ---
        # Collect ALL inputs into the DataFrame
        input_data = pd.DataFrame([{
            'employee_id': employee_id, # Handled automatically
            'age': age,
            'gender': gender,
            'years_at_company': years_at_company,
            'job_role': job_role,
            'monthly_income': monthly_income,
            'work_life_balance': work_life_balance,
            'job_satisfaction': job_satisfaction,
            'performance_rating': performance_rating,
            'number_of_promotions': number_of_promotions,
            'overtime': overtime,
            'distance_from_home': distance_from_home,
            'education_level': education_level,
            'marital_status': marital_status,
            'number_of_dependents': number_of_dependents,
            'job_level': job_level,
            'company_size': company_size,
            'remote_work': remote_work,
            'leadership_opportunities': leadership_opportunities,
            'innovation_opportunities': innovation_opportunities,
            'company_reputation': company_reputation,
            'employee_recognition': employee_recognition,
            'age_groups': age_groups, # Handled automatically
            'age_before_working': age_before_working # Handled automatically
        }])

        if st.button("Predict Attrition", type="primary", use_container_width=True, help="Run the model prediction"):
            # Now calls the function that uses the ML model if available, or falls back to logic
            prediction_int = predict_attrition(input_data)[0]
            
            # Map the integer result to the label
            result_label = "Left (High Risk)" if prediction_int == 1 else "Stayed (Low Risk)"
            
            st.markdown("#### Prediction Result:")
            
            if prediction_int == 1:
                st.error(f"⚠️ Attrition Risk: **{result_label}**")
                st.write("This employee exhibits characteristics associated with a higher likelihood of attrition. Review factors like low job satisfaction and compensation.")
            else:
                st.success(f"✅ Attrition Risk: **{result_label}**")
                st.write("This employee is predicted to stay with the company.")
    
    with tab2:
        st.subheader("Predict for a Batch of Employees (CSV)")
        st.info(f"Your uploaded CSV must contain **all {len(REQUIRED_FEATURES)} columns** for the model to work correctly. If the columns for Employee ID, Age Group, or Age Before Working are missing, the code will use default/derived values.")
        
        # Download button for the sample file template
        sample_data_bytes = get_sample_csv_for_download()
        st.download_button(
            label="Download Full Sample CSV Template",
            data=sample_data_bytes,
            file_name="attrition_prediction_full_template.csv",
            mime="text/csv",
            help="Download a file containing all 24 required columns."
        )
        
        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

        if uploaded_file is not None:
            try:
                # Read the uploaded file
                data = pd.read_csv(uploaded_file)
                st.write(f"Loaded file with {len(data)} rows.")

                # Ensure column names are lowercase to match the model's expectation
                data.columns = [col.lower() for col in data.columns]

                # Validate required columns
                # We check for all features except the three that can be derived/defaulted
                auto_handled_cols = ['employee_id', 'age_groups', 'age_before_working']
                essential_cols = [c for c in REQUIRED_FEATURES if c not in auto_handled_cols]
                missing_cols = [col for col in essential_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"The uploaded CSV is missing the following **essential** columns: {', '.join(missing_cols)}. Please check your file.")
                else:
                    with st.spinner("Running batch prediction..."):
                        # --- Fill Missing Auto-Handled Columns ---
                        if 'employee_id' not in data.columns:
                             data['employee_id'] = [np.random.randint(100000, 999999) for _ in range(len(data))]
                        if 'age_before_working' not in data.columns:
                             data['age_before_working'] = 22
                        if 'age_groups' not in data.columns and 'age' in data.columns:
                             data['age_groups'] = data['age'].apply(calculate_age_group)
                        
                        # 1. Run the prediction pipeline with ALL required features
                        input_df = data[REQUIRED_FEATURES].copy()
                        
                        # Now calls the function that uses the internal, reliable logic
                        predictions_int = predict_attrition(input_df)
                        
                        # 2. Map predictions to human-readable labels
                        predictions_label = pd.Series(predictions_int).map({0: 'Stayed', 1: 'Left'})
                        
                        # 3. Add the prediction column to the original data
                        data["attrition_prediction"] = predictions_label

                        st.success("Batch prediction complete!")
                        st.dataframe(data.head())

                        # Function to convert DataFrame to CSV for download
                        def convert_df_to_csv(df):
                            return df.to_csv(index=False).encode('utf-8')

                        csv = convert_df_to_csv(data)

                        # Download button
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="attrition_predictions_results.csv",
                            mime="text/csv",
                            help="Download the original data with the added 'attrition_prediction' column."
                        )
            except Exception as e:
                st.error(f"An error occurred during file processing. This usually means a column had unexpected non-numeric data or was entirely missing: {e}")
                st.markdown("Please ensure the uploaded file is a valid CSV and all **essential** columns are present.")


# --- Streamlit Page Configuration (Title at first) ---
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Navigation Setup ---
# Pages are defined here 
pages = [
    st.Page(home_page, icon="🏠", title="Home"),
    st.Page(visualization_page, icon="📈", title="Visualization"),
    st.Page(dashboard_page, icon="📊", title="Dashboard (External)"),
    st.Page(model_page, icon="💻", title="Prediction Model"),
]

# Use the st.navigation object for the primary structure
pg = st.navigation(pages)

# Run the app
pg.run()