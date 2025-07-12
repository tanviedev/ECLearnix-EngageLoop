# 🚀 EngageLoop – User Intelligence & Churn Prediction App

**EngageLoop** is an AI-powered web application built using **FastAPI** that predicts user churn risk, segments users by behavior, and analyzes user types — empowering EdTechs, learning platforms, and organizations to engage smarter.

---

## 🧠 Key Features

- 🔍 **Churn Prediction** – Classifies users as “At Risk” or “Engaged”  
- 🧩 **User Segmentation (KMeans)** – Groups users into data-driven engagement segments  
- 🧑‍💼 **User Type Classifier** – Predicts likelihood of user persistence based on profile  
- 📊 **Sleek Dashboard UI** – Styled using a data-intelligent rain-forest dark theme  
- 🌐 **Web Interface** – Built with FastAPI, HTML + Jinja2 templates  

---

## 🛠️ Tech Stack

- **FastAPI** – Lightning-fast web API framework  
- **Jinja2** – HTML template rendering  
- **scikit-learn / LightGBM / joblib** – ML models & serialization  
- **pandas** – Feature inputs & transformation  

---

## 💡 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/engageloop.git
cd engageloop

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
uvicorn main:app --reload
```
Developed by Tanvi Takle

Inspired by user behavior insights and elegant data design ✨
