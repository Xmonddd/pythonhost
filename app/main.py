from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="AiHealth API")

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

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/analyze")
def analyze(body: dict):
    if not body.get("symptoms"):
        raise HTTPException(status_code=400, detail="symptoms required")
    return {"severity":"low","insights":["demo"],"advice":"hydrate"}

@app.get("/")
def root(): return {"message":"AiHealth API running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")))