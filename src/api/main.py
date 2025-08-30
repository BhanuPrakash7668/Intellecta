from fastapi import FastAPI
from .routes import router as api_router

app = FastAPI(title="Research Intelligence API", version="0.1.0")

app.include_router(api_router, prefix="/api")

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
