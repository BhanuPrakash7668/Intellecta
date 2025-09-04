from fastapi import FastAPI
from .routes import router as api_router
import os

"""
LangSmith configuration for tracing and metrics
"""

  # Instruments supported libraries


app = FastAPI(title="Research Intelligence API", version="0.2.0")


app.include_router(api_router, prefix="/api")

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
