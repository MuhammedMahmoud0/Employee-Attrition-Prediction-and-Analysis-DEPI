from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("../models/model.pkl")

app = FastAPI(title="Employee Attrition Prediction")

class EmployeeData(BaseModel):
    Age: int
    MonthlyIncome: float
    JobSatisfaction: int

@app.post("/predict")
def predict(data: EmployeeData):
    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]  # probability for positive class

    return {
        "prediction": int(prediction[0]),
        "probability": float(prediction_proba[0])
    }

@app.get("/health")
def health():
    return {"status": "ok"}
