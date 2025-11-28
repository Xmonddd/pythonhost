from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Tuple, Optional
import os

from .schemas import AnalyzeRequest, AnalyzeResponse
from .normalization import normalize_list
from .red_flags import evaluate_red_flags
from .ml_model import symptom_model

app = FastAPI(title="AiHealth API", version="0.3.0")

# CORS (Firebase + local)
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

# Severity advice
DEFAULT_ADVICE: Dict[str, str] = {
    "low": "Maintain hydration, rest, and reassess if new symptoms appear.",
    "medium": "Rest, hydrate, consider OTC relief, and consult a clinician if symptoms persist.",
    "high": "Urgent concern. Seek immediate medical attention.",
}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

def ensure_model_loaded() -> None:
    try:
        symptom_model.load()
    except FileNotFoundError:
        # Train on first boot if artifacts are missing
        from train_model import main as train
        train()  # creates model.joblib + meta.joblib in working dir
        symptom_model.load()

@app.on_event("startup")
def _startup():
    ensure_model_loaded()

@app.get("/")
def root():
    return {"message": "AiHealth API running", "model_loaded": symptom_model.loaded}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": symptom_model.loaded}

@app.get("/symptoms")
def list_symptoms():
    ensure_model_loaded()
    return {"symptoms": symptom_model.symptoms_space}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom is required.")
    ensure_model_loaded()

    # Normalize inputs
    sym_norm: List[str] = normalize_list(req.symptoms)
    if not sym_norm:
        raise HTTPException(status_code=400, detail="Supplied symptoms are not recognized.")

    # Rule-based red flags
    red_flags, red_sev = evaluate_red_flags(sym_norm)

    # ML predictions
    try:
        preds: List[Tuple[str, float]] = symptom_model.predict(sym_norm, top_k=3, prob_threshold=0.15)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    insights = [c for c, _ in preds]
    probabilities = {c: float(p) for c, p in preds} or None

    # Compute severity (combine rules + simple mapping by top prob)
    computed = "low"
    if red_sev and SEVERITY_RANK[red_sev] > SEVERITY_RANK[computed]:
        computed = red_sev
    if preds:
        top_prob = preds[0][1]
        if top_prob >= 0.6:
            computed = max(computed, "medium", key=lambda s: SEVERITY_RANK[s])

    advice = DEFAULT_ADVICE[computed]

    top_condition: Optional[str] = insights[0] if insights else None
    top_probability: float = probabilities[top_condition] if top_condition and probabilities else 0.0
    accuracy_level = "High" if top_probability >= 0.5 else ("Moderate" if top_probability >= 0.2 else "Low")

    # Details/treatment from training metadata
    condition_details = treatment = None
    if top_condition:
        info = symptom_model.get_condition_info(top_condition)
        condition_details = info.get("details") or None
        treatment = info.get("treatment") or None

    return AnalyzeResponse(
        severity=computed,
        insights=insights,
        advice=advice,
        redFlags=red_flags,
        probabilities=probabilities,
        topCondition=top_condition,
        conditionDetails=condition_details,
        treatment=treatment,
        accuracyLevel=accuracy_level,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))