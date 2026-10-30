# Machine Translation System — NLPA Project 2026

A machine translation system using pre-trained Transformer architectures (MarianMT, mBART-50, M2M-100) with evaluation via BLEU, METEOR, and chrF metrics.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4   # METEOR dependency
```

## Running

```bash
# Start the API server + serve UI
uvicorn module3_engine.api:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/translate` | Translate text |
| `GET` | `/model/info` | Current model metadata |
| `POST` | `/evaluate` | Compute BLEU/METEOR/chrF |

## Module Structure

| Module | Member | Responsibility |
|--------|--------|---------------|
| `module1_preprocessing` | Member 1 | Text input, normalization, tokenization, dataset loading |
| `module2_model` | Member 2 | Model loading, caching, switching (HuggingFace) |
| `module3_engine` | Member 3 | Translation engine + FastAPI REST backend |
| `module4_eval_ui` | Member 4 | Evaluation metrics + Web UI + CSV/PDF export |

## Running Tests

```bash
pytest tests/ -v
```
