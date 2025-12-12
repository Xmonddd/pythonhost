from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Add missing imports
import os
import ssl
import redis

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

def create_redis_client() -> redis.Redis | None:
    url = os.getenv("redis://default:oaFpmaEqIKDTZEMlRsRLPuJYghemlmSp@redis.railway.internal:6379")
    # Mask password when logging
    masked = None
    if url:
        try:
            # mask password in logs
            masked = url
            if "@" in url and "://" in url:
                scheme, rest = url.split("://", 1)
                if "@" in rest:
                    creds, hostp = rest.split("@", 1)
                    masked = f"{scheme}://***@{hostp}"
        except Exception:
            masked = "<invalid>"
    print(f"[redis] REDIS_URL={masked}")
    try:
        if url:
            # Support rediss:// (TLS) and redis://
            use_ssl = url.startswith("rediss://")
            ssl_opts = {"ssl": True, "ssl_cert_reqs": ssl.CERT_NONE} if use_ssl else {}
            return redis.from_url(url, decode_responses=True, **ssl_opts)

        # Fallback if REDIS_URL not set (defaults)
        host = os.getenv("redis.railway.internal", "localhost")
        port = int(os.getenv("6379", "6379"))
        password = os.getenv("oaFpmaEqIKDTZEMlRsRLPuJYghemlmSp") or None
        db = int(os.getenv("REDIS_DB", "0"))
        return redis.Redis(host=host, port=port, password=password, db=db, decode_responses=True)
    except Exception as e:
        print(f"[redis] client init error: {e}")
        return None

@app.on_event("startup")
def load_model():
    try:
        symptom_model.load()
        print("[model] loaded")
    except FileNotFoundError as exc:
        print(f"[model] not found: {exc}")
    except Exception as exc:
        print(f"[model] load error: {exc}")

    # Initialize Redis
    app.state.redis = create_redis_client()
    if app.state.redis:
        try:
            app.state.redis.ping()
            print("[redis] connected")
        except Exception as e:
            print(f"[redis] ping failed: {e}")
            app.state.redis = None

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

    # Cache lookup
    r: redis.Redis | None = getattr(app.state, "redis", None)
    cache_key = f"analyze:{'|'.join(sorted(symptoms_norm))}:{req.age}:{req.gender}"
    if r:
        try:
            cached = r.get(cache_key)
            if cached:
                print(f"[cache] hit: {cache_key}")
                payload = json.loads(cached)
                return AnalyzeResponse(**payload)
        except Exception:
            pass

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

    response = AnalyzeResponse(
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

    # Cache store
    if r:
        try:
            print(f"[cache] store: {cache_key}")
            r.setex(cache_key, 3600, json.dumps(response.dict()))
        except Exception:
            pass

    return response

@app.get("/health")
def health():
    redis_connected = False
    r = getattr(app.state, "redis", None)
    if r:
        try:
            redis_connected = r.ping()
        except Exception:
            redis_connected = False
    return {"status": "ok", "model_loaded": symptom_model.loaded, "redis": {"connected": redis_connected}}

@app.get("/version")
def version():
    return {"app": "Symptom Checker API", "version": "0.2.0"}

@app.get("/")
def root():
    return {"message": "Symptom Checker API running. Not for diagnostic use."}

@app.get("/redis")
def redis_status():
    url = os.getenv("redis://default:oaFpmaEqIKDTZEMlRsRLPuJYghemlmSp@redis.railway.internal:6379")
    r = getattr(app.state, "redis", None)
    connected = False
    err = None
    if r:
        try:
            connected = r.ping()
        except Exception as e:
            err = str(e)
    masked = None
    if url and "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, hostp = rest.split("@", 1)
        masked = f"{scheme}://***@{hostp}"
    return {"connected": connected, "url": masked, "error": err}