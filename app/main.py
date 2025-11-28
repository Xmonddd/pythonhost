from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="AiHealth API")

origins = [
  "http://localhost:4200",
  "https://aihealth-30f3c.web.app",
  "https://aihealth-30f3c.firebaseapp.com",
]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,   # use ["*"] during dev if needed
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/health")
def health():
  return {"status": "ok"}

@app.post("/analyze")
def analyze(body: dict):
  if not body.get("symptoms"):
    raise HTTPException(status_code=400, detail="symptoms required")
  return {"severity": "low", "insights": ["demo"], "advice": "hydrate"}

if __name__ == "__main__":
  import uvicorn
  port = int(os.getenv("PORT", "8080"))   # Railway provides $PORT
  uvicorn.run("app.main:app", host="0.0.0.0", port=port)