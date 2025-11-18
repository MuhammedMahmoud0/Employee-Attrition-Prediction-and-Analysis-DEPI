import streamlit as st
import pandas as pd
import numpy as np
from api import REQUIRED_FEATURES, call_predict, get_sample_csv_for_download, calculate_age_group


def model_page():
    """Prediction Model page content with two prediction options."""
    st.title("Prediction Model 💻")
    st.markdown("---")

    st.header("Predict Employee Attrition Risk")
    st.write(f"The model pipeline (via FastAPI) requires **all {len(REQUIRED_FEATURES)} features** shown below to generate a prediction.")
    st.write("*(Note: Employee ID, Age Group, and Career Start Age are handled automatically by the code.)*")

    tab1, tab2 = st.tabs(["Single Row Prediction", "CSV File Prediction"])

    with tab1:
        st.subheader("Predict for a Single Employee")
        employee_id = np.random.randint(100000, 999999)
        age_before_working = 22

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

            age_groups = calculate_age_group(age)

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
                job_satisfaction = st.selectbox("Job Satisfaction", options=["Low", "Medium", "High", "Very High"], index=2)

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

        input_data = pd.DataFrame([{  # collect inputs
            'employee_id': employee_id,
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
            'age_groups': age_groups,
            'age_before_working': age_before_working
        }])

        debug_show_payload = st.checkbox("Show outgoing payload before sending", value=False)

        if st.button("Predict Attrition", type="primary", use_container_width=True, help="Run the model prediction"):
            # Send payload exactly as-is to FastAPI
            payload = input_data[REQUIRED_FEATURES].iloc[0].to_dict()

            if debug_show_payload:
                st.markdown("**Outgoing payload (JSON):**")
                st.json(payload)

            # call_predict will raise RuntimeError on connection errors or non-200 responses
            result = call_predict(payload)

            # Expecting at least a 'prediction' key; optional probability key
            prediction_int = result.get('prediction')
            prob = result.get('prediction_probability') or result.get('probability') or result.get('score')

            result_label = "Left (High Risk)" if int(prediction_int) == 1 else "Stayed (Low Risk)"
            st.markdown("#### Prediction Result:")
            if int(prediction_int) == 1:
                st.error(f"⚠️ Attrition Risk: **{result_label}**")
            else:
                st.success(f"✅ Attrition Risk: **{result_label}**")

            if prob is not None:
                st.write(f"Prediction probability: {prob}")

    with tab2:
        st.subheader("Predict for a Batch of Employees (CSV)")
        st.info(f"Your uploaded CSV must contain **all {len(REQUIRED_FEATURES)} columns** for the model to work correctly. Each row will be processed by calling the API or the fallback simulation individually.")

        sample_data_bytes = get_sample_csv_for_download()
        st.download_button(label="Download Full Sample CSV Template", data=sample_data_bytes, file_name="attrition_prediction_full_template.csv", mime="text/csv")

        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(f"Loaded file with {len(data)} rows.")
            data.columns = [col.lower() for col in data.columns]

            # Fill any auto-handled columns if missing (employee_id/age_groups/age_before_working)
            if 'employee_id' not in data.columns:
                data['employee_id'] = [np.random.randint(100000, 999999) for _ in range(len(data))]
            if 'age_before_working' not in data.columns:
                data['age_before_working'] = 22
            if 'age_groups' not in data.columns and 'age' in data.columns:
                data['age_groups'] = data['age'].apply(calculate_age_group)

            # Ensure all required features are present
            missing_cols = [c for c in REQUIRED_FEATURES if c not in data.columns]
            if missing_cols:
                st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
                raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

            # Iterate rows and call FastAPI per row, collect predictions
            predictions = []
            probabilities = []

            with st.spinner(f"Running batch prediction for {len(data)} rows..."):
                for index, row in data.iterrows():
                    payload = row[REQUIRED_FEATURES].to_dict()
                    # Validate payload contains no NaN
                    nan_keys = [k for k, v in payload.items() if pd.isna(v)]
                    if nan_keys:
                        raise ValueError(f"Row {index} has null values for: {', '.join(nan_keys)}")

                    result = call_predict(payload)
                    pred = result.get('prediction')
                    prob = result.get('prediction_probability') or result.get('probability') or result.get('score')
                    predictions.append(pred)
                    probabilities.append(prob)

            # Attach results to DataFrame
            data['prediction'] = predictions
            data['prediction_probability'] = probabilities

            st.success("Batch prediction completed using FastAPI.")
            st.dataframe(data.head())

            # Download CSV of actual predictions
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(data)
            st.download_button(label="Download Results CSV", data=csv, file_name="attrition_predictions_results.csv", mime="text/csv")
