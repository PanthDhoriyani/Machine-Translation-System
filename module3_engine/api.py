"""
FastAPI REST API — Machine Translation System
Module 3 — Translation Engine + API

Endpoints:
  POST /translate     (FR-3.4)
  GET  /model/info    (FR-2.5)
  POST /evaluate      (Appendix A)
  GET  /              Serves the Web UI

Run with:
  uvicorn module3_engine.api:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import asyncio

# Local imports
from module2_model.model_manager import ModelManager
from module3_engine.translation_engine import TranslationEngine
from module4_eval_ui.evaluator import Evaluator
from module4_eval_ui.report_exporter import ReportExporter

# ---------------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global engine singleton and evaluation cache
# ---------------------------------------------------------------------------
engine: Optional[TranslationEngine] = None
_lock = asyncio.Lock()

latest_eval_report: dict = {"bleu": 0.0, "meteor": 0.0, "chrf": 0.0, "num_sentences": 0}
latest_hypotheses: list[str] = []
latest_references: list[str] = []

UI_DIR = Path(__file__).parent.parent / "module4_eval_ui" / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the default model on startup."""
    global engine
    logger.info("Loading default model on startup …")
    mm = ModelManager("Helsinki-NLP/opus-mt-en-fr")
    engine = TranslationEngine(mm)
    logger.info("Default model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Machine Translation API",
    description="NLPA Project 2026 — REST API for the MT system using HuggingFace transformers.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the Web UI as static files
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    """Request body for POST /translate."""
    text: str = Field(..., min_length=1, description="Source text to translate.")
    src_lang: str = Field(..., min_length=2, max_length=10, description="Source language code, e.g. 'en'.")
    tgt_lang: str = Field(..., min_length=2, max_length=10, description="Target language code, e.g. 'fr'.")
    beam_size: int = Field(default=4, ge=1, le=10, description="Beam width for beam search.")
    decoding: Literal["beam", "greedy", "top_k"] = Field(default="beam")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text field cannot be blank.")
        return v


class TranslateResponse(BaseModel):
    """Response body for POST /translate."""
    translation: str
    src_lang: str
    tgt_lang: str
    model_used: str
    latency_ms: float


class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""
    hypotheses: list[str] = Field(..., min_length=1)
    references: list[str] = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    """Response body for POST /evaluate."""
    bleu: float
    meteor: float
    chrf: float
    num_sentences: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the Web UI index page."""
    index = UI_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "MT API is running. UI not found — see /docs for API."})


@app.post("/translate", response_model=TranslateResponse, summary="Translate text")
async def translate(request: TranslateRequest):
    """Translate source text to the target language. (FR-3.4)

    - Single sentence latency ≤ 3 s on CPU (NFR-1)
    - Returns 400 for unsupported language pairs (FR-3.5)
    - Returns 422 automatically for empty / invalid inputs (TC-3.8)
    """
    global engine
    async with _lock:  # thread-safe model access (TC-3.9)
        try:
            result = engine.translate(
                text=request.text,
                src_lang=request.src_lang,
                tgt_lang=request.tgt_lang,
                decoding=request.decoding,
                beam_size=request.beam_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except OSError as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    return TranslateResponse(**result)


@app.get("/model/info", summary="Get current model metadata")
async def model_info():
    """Return metadata about the currently loaded model. (FR-2.5, TC-2.5)"""
    info = engine.model_manager.get_model_info()
    return JSONResponse(content=info)


@app.post("/evaluate", response_model=EvaluateResponse, summary="Evaluate translation quality")
async def evaluate(request: EvaluateRequest):
    """Compute BLEU, METEOR, and chrF for hypothesis/reference pairs. (Appendix A)"""
    global latest_eval_report, latest_hypotheses, latest_references
    try:
        ev = Evaluator(request.hypotheses, request.references)
        report = ev.generate_report()
        latest_eval_report = report
        latest_hypotheses = request.hypotheses
        latest_references = request.references
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return EvaluateResponse(**report)


@app.post("/evaluate/upload", summary="Batch evaluation via CSV file upload")
async def evaluate_upload(file: UploadFile = File(...)):
    """Accept a CSV file with 'src' and 'tgt' columns, translate and evaluate. (FR-4.5)"""
    global latest_eval_report, latest_hypotheses, latest_references
    import io
    import pandas as pd

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if "src" not in df.columns or "tgt" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have 'src' and 'tgt' columns.")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    hypotheses = df["src"].dropna().tolist()
    references = df["tgt"].dropna().tolist()

    ev = Evaluator(hypotheses, references)
    report = ev.generate_report()
    latest_eval_report = report
    latest_hypotheses = hypotheses
    latest_references = references
    return JSONResponse(content=report)


@app.get("/evaluate/export/csv", summary="Download last evaluation as CSV")
async def export_csv():
    """Export the most recent evaluation report as a downloadable CSV. (FR-4.6, TC-4.7)"""
    exporter = ReportExporter(
        report=latest_eval_report,
        hypotheses=latest_hypotheses,
        references=latest_references,
    )
    csv_bytes = exporter.to_csv_bytes()
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluation_report.csv"},
    )


@app.get("/evaluate/export/pdf", summary="Download last evaluation as PDF")
async def export_pdf():
    """Export the most recent evaluation report as a downloadable PDF. (FR-4.6)"""
    exporter = ReportExporter(
        report=latest_eval_report,
        hypotheses=latest_hypotheses,
        references=latest_references,
    )
    try:
        pdf_bytes = exporter.to_pdf_bytes()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=evaluation_report.pdf"},
    )
