"""
Main FastAPI Application Entrypoint (Track C - Ali).

Runs the REST API server and serves the Interactive Web Dashboard.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is in python search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.routes import stocks, predictions, analytics

app = FastAPI(
    title="IDX AI Stock Prediction & Trading Strategy API",
    description=(
        "Fullstack REST API for IDX Bluechip Stock Prediction, "
        "Ensemble Trading Signals (T+1), Real-time Scoring, and Strategy Performance Leaderboard. "
        "Track C (Ali)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local and web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(stocks.router)
app.include_router(predictions.router)
app.include_router(analytics.router)

# Mount Frontend static assets and HTML UI
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found, please check frontend/ directory."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
