from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import uvicorn
# uvicorn api:app --reload --host 127.0.0.1 --port 8000
app = FastAPI(title="Attrition Prediction API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global model variable
model = None

class PredictionRequest(BaseModel):
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

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probability_stayed: float
    probability_left: float


def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

# Run this function automatically when the server starts.
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("✓ FastAPI server started")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

#Defines the home route of the API.
@app.get("/")
async def root():
    return {"status": "online", "model_loaded": model is not None}
#Used to check the health of the API & Ensures API only runs when the ML model is ready.
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_df = pd.DataFrame([request.dict()])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        prediction_label = "Stayed" if prediction == 1 else "Left"
        confidence = max(prediction_proba) * 100
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            probability_stayed=float(prediction_proba[1]) * 100 if len(prediction_proba) > 1 else 0,
            probability_left=float(prediction_proba[0]) * 100 if len(prediction_proba) > 0 else 0
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
