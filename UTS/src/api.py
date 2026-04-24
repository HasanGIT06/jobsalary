from feature_engineering.feature_engineering import feature_engineering
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ── Auto-train model if pkl is missing ────────────────────────────────────────
PKL_PATH_CAT = Path("artifacts/model_cat.pkl")
PKL_PATH_NUM = Path("artifacts/model_num.pkl")

if not (PKL_PATH_CAT.exists() and PKL_PATH_NUM.exists()):
    print(f"No model found - training pipeline...")
    from pipeline import run_pipeline
    run_pipeline()

# ── Load model ────────────────────────────────────────────────────────────────
model_cat = joblib.load(PKL_PATH_CAT)
model_num = joblib.load(PKL_PATH_NUM)

app = FastAPI(title="Job Placement Status")

class Passenger(BaseModel):
    student_id                 : str   = "S001"
    gender                     : str   = "Male"
    ssc_percentage             : float = 75.0
    hsc_percentage             : float = 78.0
    degree_percentage          : float = 80.0
    cgpa                       : float = 7.5
    entrance_exam_score        : float = 70.0
    technical_skill_score      : float = 75.0
    soft_skill_score           : float = 70.0
    internship_count           : int   = 1
    live_projects              : int   = 1
    work_experience_months     : int   = 6
    certifications             : int   = 2
    extracurricular_activities : str   = "Yes"
    attendance_percentage      : float = 85.0
    backlogs                   : int   = 0

@app.get("/")
def root():
    return {"message": "Selamat Datang di Halaman Pengecekan Job Status!"}

@app.post("/predict")
def predict(passenger: Passenger):
    try:
        data = passenger.model_dump()
        df = pd.DataFrame([data])
        df = feature_engineering(df)
        features = ['skill_academic_score', 'backlogs', 'engagement_score']
        df = df[features]
        pred_cat = model_cat.predict(df)[0]
        pred_num = model_num.predict(df)[0]
        if pred_cat == 0:
            pred_num = 0
        return {"placement_prediction": int(pred_cat), "salary_prediction": float(pred_num)}

    except Exception as e:
        return {"error": str(e)}