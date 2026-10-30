"""
main.py — Entry point for the Machine Translation System
NLPA Project 2026

Usage:
    python main.py
    # or
    uvicorn module3_engine.api:app --reload --port 8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "module3_engine.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
