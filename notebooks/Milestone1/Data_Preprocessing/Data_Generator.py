import pandas as pd
import numpy as np
from faker import Faker
import random
from tqdm import tqdm
import csv
import math

# Initialize Faker
fake = Faker()

# --- Configuration based on the generation plan ---

NUM_EMPLOYEES = 90000

# Define value ranges and probability distributions
JOB_ROLES = ['Technology', 'Healthcare', 'Finance', 'Education', 'Media']
JOB_ROLE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

COMPANY_SIZES = ['Small', 'Medium', 'Large']
COMPANY_SIZE_WEIGHTS = [0.20, 0.45, 0.35]

EDUCATION_LEVELS = ['High School', 'Associate Degree', 'Bachelor’s Degree', 'Master’s Degree', 'PhD']

# --- Helper functions to enforce logic and dependencies ---

def get_job_level(age, years_at_company):
    """Determines job level based on age and tenure."""
    if age <= 25 or years_at_company <= 2:
        return np.random.choice(['Entry', 'Mid'], p=[0.9, 0.1])
    elif 26 <= age <= 38 or 3 <= years_at_company <= 8:
        return np.random.choice(['Entry', 'Mid', 'Senior'], p=[0.15, 0.75, 0.10])
    else:
        return np.random.choice(['Mid', 'Senior'], p=[0.3, 0.7])

def get_monthly_income(job_level, job_role, company_size):
    """Calculates monthly income based on several factors."""
    base = {'Entry': 3500, 'Mid': 6500, 'Senior': 12000}
    
    role_multiplier = {'Technology': 1.25, 'Finance': 1.18, 'Healthcare': 1.1, 'Media': 0.95, 'Education': 0.9}
    size_multiplier = {'Small': 0.9, 'Medium': 1.0, 'Large': 1.15}
    
    base_income = base.get(job_level, 3000)
    
    # Introduce random variation
    variation = random.uniform(0.85, 1.15)
    
    income = base_income * role_multiplier.get(job_role, 1.0) * size_multiplier.get(company_size, 1.0) * variation
    return int(income)

def get_education_level(job_level, job_role):
    """Determines education level based on job level and role."""
    if job_level == 'Entry':
        return np.random.choice(EDUCATION_LEVELS, p=[0.3, 0.3, 0.35, 0.04, 0.01])
    elif job_level == 'Mid':
        return np.random.choice(EDUCATION_LEVELS, p=[0.05, 0.1, 0.5, 0.3, 0.05])
    else: # Senior
        if job_role == 'Technology' or job_role == 'Education':
             return np.random.choice(EDUCATION_LEVELS, p=[0.01, 0.04, 0.4, 0.4, 0.15])
        else:
             return np.random.choice(EDUCATION_LEVELS, p=[0.02, 0.08, 0.5, 0.35, 0.05])

def get_marital_status_and_dependents(age):
    """Determines marital status and number of dependents based on age."""
    if age <= 28:
        status = np.random.choice(['Single', 'Married'], p=[0.85, 0.15])
    elif 29 <= age <= 45:
        status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.20, 0.70, 0.10])
    else:
        status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.10, 0.60, 0.30])
        
    if status == 'Single':
        dependents = np.random.choice([0, 1], p=[0.9, 0.1])
    elif status == 'Married':
        dependents = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.2, 0.35, 0.2, 0.1, 0.05])
    else: # Divorced
        dependents = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])
        
    return status, dependents
    
def get_attrition(record):
    """
    Determines attrition using a log-odds model for stronger correlation.
    A score is calculated based on various factors, then converted to a probability.
    """
    # Base score: A negative value indicates a baseline tendency to stay.
    score = -2.2 

    # --- Factors increasing attrition risk (positive score contribution) ---
    
    # Job Satisfaction (very high impact)
    satisfaction_map = {'Low': 2.5, 'Medium': 1.2, 'High': -0.5, 'Very High': -2.0}
    score += satisfaction_map.get(record['job_satisfaction'], 0)

    # Work-Life Balance
    wlb_map = {'Poor': 1.5, 'Fair': 0.5, 'Good': -0.5, 'Excellent': -1.0}
    score += wlb_map.get(record['work_life_balance'], 0)

    # Overtime (significant impact)
    if record['overtime'] == 'Yes':
        score += 0.8

    # Employee Recognition
    recognition_map = {'Low': 1.3, 'Medium': 0.2, 'High': -0.6, 'Very High': -1.2}
    score += recognition_map.get(record['employee_recognition'], 0)

    # Income Check (being underpaid is a major factor)
    income_thresholds = {'Entry': 3000, 'Mid': 6000, 'Senior': 11000}
    if record['monthly_income'] < income_thresholds.get(record['job_level'], 99999):
        score += 1.8
        
    # Career Stagnation
    if record['years_at_company'] > 4 and record['number_of_promotions'] == 0:
        score += 1.5
        
    # Early Tenure Risk
    if record['years_at_company'] <= 2:
        score += 0.7

    # --- Factors decreasing attrition risk (negative score contribution) ---

    # Company Reputation
    reputation_map = {'Poor': 0.5, 'Fair': 0.1, 'Good': -0.4, 'Excellent': -0.9}
    score += reputation_map.get(record['company_reputation'], 0)

    # Leadership Opportunities
    if record['leadership_opportunities'] == 'Yes':
        score -= 0.5

    # Sigmoid function to convert the score to a probability
    p_attrition = 1 / (1 + math.exp(-score))
    
    # Ensure probability is within a reasonable bound
    p_attrition = max(0.01, min(0.95, p_attrition))
    
    return np.random.choice(['Stayed', 'Left'], p=[1 - p_attrition, p_attrition])

# --- Main data generation loop ---

def generate_dataset(num_employees):
    """Generates the full dataset with all specified columns and relationships."""
    employee_data = []
    
    for i in tqdm(range(num_employees), desc="Generating Employee Data"):
        record = {}
        
        # Foundational Attributes
        record['employee_id'] = i + 1
        age = int(np.random.normal(loc=38, scale=10))  # mean 38, std dev 10
        age = max(18, min(65, age))  # keep within realistic HR bounds
        record['age'] = age
        record['gender'] = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
        record['company_size'] = np.random.choice(COMPANY_SIZES, p=COMPANY_SIZE_WEIGHTS)
        record['job_role'] = np.random.choice(JOB_ROLES, p=JOB_ROLE_WEIGHTS)

        # Dependent Attributes
        max_years = max(0, record['age'] - 18)
        record['years_at_company'] = random.randint(0, max_years) if max_years > 0 else 0
        
        record['job_level'] = get_job_level(record['age'], record['years_at_company'])
        record['monthly_income'] = get_monthly_income(record['job_level'], record['job_role'], record['company_size'])
        record['education_level'] = get_education_level(record['job_level'], record['job_role'])
        record['marital_status'], record['number_of_dependents'] = get_marital_status_and_dependents(record['age'])

        # Inter-correlated Qualitative Ratings (driven by a hidden 'happiness' score)
        happiness_score = random.uniform(0.1, 1.0)
        
        if happiness_score > 0.7:
            record['work_life_balance'] = np.random.choice(['Good', 'Excellent'], p=[0.6, 0.4])
            record['job_satisfaction'] = np.random.choice(['High', 'Very High'], p=[0.5, 0.5])
            record['company_reputation'] = np.random.choice(['Good', 'Excellent'], p=[0.7, 0.3])
            record['employee_recognition'] = np.random.choice(['High', 'Very High'], p=[0.6, 0.4])
        elif happiness_score < 0.3:
            record['work_life_balance'] = np.random.choice(['Poor', 'Fair'], p=[0.7, 0.3])
            record['job_satisfaction'] = np.random.choice(['Low', 'Medium'], p=[0.8, 0.2])
            record['company_reputation'] = np.random.choice(['Poor', 'Fair'], p=[0.6, 0.4])
            record['employee_recognition'] = np.random.choice(['Low', 'Medium'], p=[0.7, 0.3])
        else: # Average employee
            record['work_life_balance'] = np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], p=[0.1, 0.3, 0.5, 0.1])
            record['job_satisfaction'] = np.random.choice(['Low', 'Medium', 'High', 'Very High'], p=[0.1, 0.4, 0.4, 0.1])
            record['company_reputation'] = np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], p=[0.05, 0.25, 0.6, 0.1])
            record['employee_recognition'] = np.random.choice(['Low', 'Medium', 'High', 'Very High'], p=[0.15, 0.5, 0.3, 0.05])
            
        # Other Attributes
        record['performance_rating'] = np.random.choice(['Low', 'Average', 'High', 'Excellent'], p=[0.1, 0.5, 0.3, 0.1])
        
        max_promotions = max(0, record['years_at_company'] // 3)
        if record['performance_rating'] in ['High', 'Excellent']:
            max_promotions += 1
        record['number_of_promotions'] = random.randint(0, max_promotions) if max_promotions > 0 else 0
        
        # Ensure promotions are logical with job level
        if record['job_level'] == 'Entry' and record['number_of_promotions'] > 1:
            record['number_of_promotions'] = np.random.choice([0, 1])

        record['overtime'] = 'Yes' if record['work_life_balance'] == 'Poor' or (record['job_role'] == 'Technology' and random.random() < 0.4) else 'No'
        record['distance_from_home'] = max(1, int(random.gammavariate(2, 10)))
        
        record['remote_work'] = 'Yes' if record['job_role'] in ['Technology', 'Media', 'Finance'] and random.random() < 0.6 else 'No'
        if record['job_role'] == 'Healthcare': record['remote_work'] = 'No' # Override for healthcare

        record['leadership_opportunities'] = 'Yes' if record['job_level'] == 'Senior' or (record['job_level'] == 'Mid' and random.random() < 0.3) else 'No'
        record['innovation_opportunities'] = 'Yes' if record['job_role'] == 'Technology' or (record['company_size'] != 'Medium' and random.random() < 0.4) else 'No'

        # Target Variable - Attrition (generated last based on other factors)
        record['attrition'] = get_attrition(record)

        employee_data.append(record)
        
    return employee_data

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Generate the core data
    data = generate_dataset(NUM_EMPLOYEES)
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(data)
    
    # 3. Calculate derived columns
    print("\nCalculating derived columns...")
    df['age_before_working'] = df['age'] - df['years_at_company']
    
    age_bins = [17, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['age_groups'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
    
    # 4. Define final column order for the CSV
    final_column_order = [
        'employee_id', 'age', 'gender', 'years_at_company', 'job_role',
        'monthly_income', 'work_life_balance', 'job_satisfaction',
        'performance_rating', 'number_of_promotions', 'overtime',
        'distance_from_home', 'education_level', 'marital_status',
        'number_of_dependents', 'job_level', 'company_size', 'remote_work',
        'leadership_opportunities', 'innovation_opportunities',
        'company_reputation', 'employee_recognition', 'attrition',
        'age_groups', 'age_before_working'
    ]
    df = df[final_column_order]

    # 5. Save to CSV
    output_filename = '../../../data/Faker_Data/synthetic_hr_dataset.csv'
    df.to_csv(output_filename, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\nSuccessfully generated and saved {len(df)} records to '{output_filename}'.")
    
    # 6. Display a sample of the data and verify constraints
    print("\n--- Data Sample (First 5 Rows) ---")
    print(df.head())
    print("\n--- Data Verification ---")
    print(f"Number of records: {len(df)}")
    print(f"Years at company > (Age - 18)? : { (df['years_at_company'] >= (df['age'] - 18)).any() }")
    print(f"Min age before working: { df['age_before_working'].min() }")


