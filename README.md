# EngageLoop: User Churn Intelligence API

This FastAPI application uses machine learning models to predict:
- User types based on interaction history
- Drop-off likelihood (churn risk)

## Project Features
- Drop-off prediction using LightGBM
- Clustering of users (KMeans, DBSCAN, Agglomerative)
- SHAP explainability (in Jupyter)
- FastAPI backend to serve models

## Getting Started

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload
