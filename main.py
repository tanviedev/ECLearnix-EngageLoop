from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load models and preprocessor
churn_model = joblib.load("models/model_churn_predictor.pkl")
segmentation_model = joblib.load("models/model_user_segmentation_kmeans.pkl")
user_type_model = joblib.load("models/model_user_type_predictor.pkl")
preprocessor = joblib.load("models/model_preprocessor.pkl")

app = FastAPI(title="User Intelligence API")

# Input schema
class UserInput(BaseModel):
    App_Installed: int
    First_Login_Completed: int
    Registered_for_Event: int
    Course_Completed: int
    Newsletter_Subscribed: int
    Time_Spent_Total_Minutes: float
    Days_Since_Last_Activity: float
    User_Type: str
    Region: str
    Department: str
    Platform_Source: str

@app.get("/")
def read_root():
    return {"message": "User Intelligence API running"}

@app.post("/predict-churn")
def predict_churn(data: UserInput):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    proba = churn_model.predict_proba(X)[0][1]
    return {
        "dropoff_risk": proba > 0.5,
        "probability": round(proba, 4)
    }

@app.post("/predict-user-segment")
def predict_segment(data: UserInput):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    cluster = int(segmentation_model.predict(X)[0])
    return {"segment": cluster}

@app.post("/predict-user-type")
def predict_user_type(data: UserInput):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    pred = user_type_model.predict(X)[0]
    return {"user_type": pred}
