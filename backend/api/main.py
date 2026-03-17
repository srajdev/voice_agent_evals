"""
FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd or any parent directory

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.evaluate import router as evaluate_router
from api.routes.reports import router as reports_router

app = FastAPI(
    title="Voice Evals API",
    description="Open-source framework for evaluating voice AI agents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
