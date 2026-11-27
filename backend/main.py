
from fastapi import FastAPI

app = FastAPI(
    title="CosmoTrack API",
    description="AI-Powered Space Domain Awareness Platform",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "project": "CosmoTrack",
        "status": "online",
        "mode": "prototype"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}
