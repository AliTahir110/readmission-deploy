from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
import os, joblib, numpy as np

app = FastAPI(title="Readmission API", version="1.1")

# Static + templates
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

SK_MODEL_PATH = os.getenv("SK_MODEL_PATH", "models/sklearn_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

def _load_model():
    if not os.path.exists(SK_MODEL_PATH):
        return None
    try:
        return joblib.load(SK_MODEL_PATH)
    except Exception as e:
        print("Model load error:", e)
        return None

_model = _load_model()

# Match training feature order:
# ["time_in_hospital", "n_medications", "n_procedures", "diag_group_id"]
DIAG_GROUPS = {
    "other": 0,
    "circulatory": 1,
    "respiratory": 2,
    "digestive": 3,
    "diabetes": 4,
    "injury": 5,
    "musculoskeletal": 6,
    "genitourinary": 7,
    "neoplasms": 8,
}

class Patient(BaseModel):
    time_in_hospital: int = Field(ge=0, le=60)
    n_medications:   int = Field(ge=0, le=400)
    n_procedures:    int = Field(ge=0, le=40)
    diag_group: Literal[
        "other","circulatory","respiratory","digestive","diabetes",
        "injury","musculoskeletal","genitourinary","neoplasms"
    ] = "other"

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_version": MODEL_VERSION})

@app.get("/health")
def health():
    return {"status": "ok" if _model else "model_missing", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(p: Patient):
    if _model is None:
        raise HTTPException(503, "Model not available")

    diag_id = DIAG_GROUPS.get(p.diag_group, 0)
    X = np.array([[p.time_in_hospital, p.n_medications, p.n_procedures, diag_id]])

    try:
        prob = float(_model.predict_proba(X)[0, 1]) if hasattr(_model, "predict_proba") \
              else float(_model.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, f"inference_error: {e}")

    return {"risk": "high" if prob >= 0.5 else "low", "prob": round(prob, 4), "model_version": MODEL_VERSION}
