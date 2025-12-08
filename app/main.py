from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 🔥 ADD CSV DEBUG LOADER (HERE)
import pandas as pd

try:
    df = pd.read_csv("training_cases.csv")
    print("======================================")
    print("CSV LOADED:", len(df), "rows")
    print("FIRST ROW:", df.iloc[0].to_dict())
    print("======================================")
except Exception as e:
    print("ERROR LOADING CSV:", e)
# 🔥 END CSV DEBUG LOADER

from .schemas import AnalyzeRequest, AnalyzeResponse
from .normalization import normalize_list
from .ml_model import symptom_model
from .red_flags import evaluate_red_flags

app = FastAPI(
    title="Symptom Checker API",
    description="Basic demo API with simple ML model. Not for medical use.",
    version="0.2.0"
)

# Updated CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://aihealth-30f3c.web.app",
        "https://aihealth-30f3c.firebaseapp.com",
    ],
    allow_origin_regex=r"https://.*\.web\.app|https://.*\.firebaseapp\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONDITION_SEVERITY = {
    "flu": "medium",
    "meningitis": "high",
    "migraine": "medium",
    "asthma": "medium",
    "cardiac_issue": "high",
    "food_poisoning": "medium",
    "dehydration": "medium"
}

DEFAULT_ADVICE = {
    "low": "Rest, hydrate, and monitor for 24–48 hours.",
    "medium": "Monitor and consult a healthcare professional if symptoms persist or worsen.",
    "high": "Seek urgent medical attention immediately."
}

@app.on_event("startup")
def load_model():
    try:
        symptom_model.load()
    except FileNotFoundError as exc:
        raise RuntimeError("Model artifacts not found. Train the model first.") from exc
    except Exception as exc:
        raise RuntimeError("Unable to load model artifacts.") from exc

# Optional helper for GET in browser
@app.get("/analyze")
def analyze_usage():
    return {
        "message": "Use POST /analyze with JSON body.",
        "example": { "symptoms": ["fever", "cough"], "age": 18, "gender": "male" }
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom is required.")

    symptoms_norm = normalize_list(req.symptoms)
    if not symptoms_norm:
        raise HTTPException(status_code=400, detail="Supplied symptoms are not recognized.")

    red_flags, red_flag_severity = evaluate_red_flags(symptoms_norm)

    try:
        predictions = symptom_model.predict(symptoms_norm)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    insights = [c for c, _ in predictions]
    probabilities = {c: float(p) for c, p in predictions} or None

    severity_rank = {"low": 0, "medium": 1, "high": 2}
    computed_severity = "low"
    for cond in insights:
        sev = CONDITION_SEVERITY.get(cond)
        if sev and severity_rank[sev] > severity_rank[computed_severity]:
            computed_severity = sev
    if red_flag_severity and severity_rank[red_flag_severity] > severity_rank[computed_severity]:
        computed_severity = red_flag_severity

    advice = DEFAULT_ADVICE[computed_severity]

    top_condition = insights[0] if insights else None
    top_probability = probabilities[top_condition] if top_condition and probabilities else 0.0
    if top_probability >= 0.5:
        accuracy_level = "High"
    elif top_probability >= 0.2:
        accuracy_level = "Moderate"
    else:
        accuracy_level = "Low"

    condition_details = treatment = None
    if top_condition:
        info = symptom_model.get_condition_info(top_condition)
        condition_details = info.get("details") or None
        treatment = info.get("treatment") or None

    return AnalyzeResponse(
        severity=computed_severity,
        insights=insights,
        advice=advice,
        redFlags=red_flags,
        probabilities=probabilities,
        topCondition=top_condition,
        conditionDetails=condition_details,
        treatment=treatment,
        accuracyLevel=accuracy_level,
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": symptom_model.loaded}

@app.get("/version")
def version():
    return {"app": "Symptom Checker API", "version": "0.2.0"}

@app.get("/")
def root():
    return {"message": "Symptom Checker API running. Not for diagnostic use."}
