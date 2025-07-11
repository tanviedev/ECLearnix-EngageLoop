from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import os

# Load models
churn_model = joblib.load("models/model_churn_predictor.pkl")
segment_model = joblib.load("models/model_user_segmentation_kmeans.pkl")
type_model = joblib.load("models/model_user_type_predictor.pkl")
preprocessor = joblib.load("models/model_preprocessor.pkl")

# Initialize app and templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    User_Type: str = Form(...),
    Region: str = Form(...),
    Department: str = Form(...),
    Platform_Source: str = Form(...),
    App_Installed: int = Form(...),
    First_Login_Completed: int = Form(...),
    Registered_for_Event: int = Form(...),
    Course_Completed: int = Form(...),
    Newsletter_Subscribed: int = Form(...),
    Time_Spent_Total_Minutes: int = Form(...),
    Days_Since_Last_Activity: int = Form(...)
):
    # ✅ Input dictionary
    input_data = {
        'User_Type': User_Type,
        'Region': Region,
        'Department': Department,
        'Platform_Source': Platform_Source,
        'App_Installed': App_Installed,
        'First_Login_Completed': First_Login_Completed,
        'Registered_for_Event': Registered_for_Event,
        'Course_Completed': Course_Completed,
        'Newsletter_Subscribed': Newsletter_Subscribed,
        'Time_Spent_Total_Minutes': Time_Spent_Total_Minutes,
        'Days_Since_Last_Activity': Days_Since_Last_Activity
    }

    print("✅ Received POST Data:", input_data)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # ❌ Don't preprocess manually for pipeline models
    churn_result = churn_model.predict(input_df)[0]
    type_result = type_model.predict(input_df)[0]

    # ✅ Preprocess manually only for KMeans
    input_transformed = preprocessor.transform(input_df)
    segment_result = segment_model.predict(input_transformed)[0]

    # Return rendered page with results
    churn_label = "At Risk of Drop-off" if churn_result else "Engaged"

    segment_labels = {
        0: "🟡 Observers (App Installed, No Login)",
        1: "🔴 Dormant Users (Logged In, No Completion)",
        2: "🔵 Partial Web Users (Mid Engagement)",
        3: "🟢 Loyal but Inactive (Fully Engaged Before)"
    }

    # Inverted ranking dictionary (optional: use real types if stored)
    user_type_rank_labels = {
        0: "Highest Drop Risk Type",
        1: "Moderate Drop Risk Type",
        2: "Low Drop Risk Type",
        3: "Least Likely to Drop"
    }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "churn_result": churn_label,
        "segment_result": segment_labels.get(segment_result, f"Segment {segment_result}"),
        "type_result": user_type_rank_labels.get(type_result, f"Encoded Type Rank {type_result}")
    })
