from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load your trained pipeline (including preprocessing)
pipeline = joblib.load("../models/model.pkl")

app = FastAPI(title="Employee Attrition Prediction")


class EmployeeData(BaseModel):
    employee_id: int
    age: int
    gender: str
    years_at_company: int
    job_role: str
    monthly_income: float
    work_life_balance: str
    job_satisfaction: str
    performance_rating: str
    number_of_promotions: int
    overtime: str
    distance_from_home: int
    education_level: str
    marital_status: str
    number_of_dependents: int
    job_level: str
    company_size: str
    remote_work: str
    leadership_opportunities: str
    innovation_opportunities: str
    company_reputation: str
    employee_recognition: str
    age_groups: str
    age_before_working: int


@app.post("/predict")
def predict(data: EmployeeData):
    input_df = pd.DataFrame([data.dict()])

    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)[:, 0]
    # Return the prediction and probability
    # the propability is for the positive class (attrition = 1)
    return {"prediction": int(prediction[0]), "probability": float(prediction_proba[0])}


@app.get("/health")
def health():
    return {"status": "ok"}
