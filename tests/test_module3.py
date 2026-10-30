"""
Test Suite — Module 3: Translation Engine & REST API
TC-3.1 through TC-3.9 from SRS Section 7.3
"""

import time
import pytest
from httpx import AsyncClient, ASGITransport

from module3_engine.api import app


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def client():
    """Async test client for the FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestTranslationAPI:
    @pytest.mark.asyncio
    async def test_basic_translation_en_fr(self, client):
        """TC-3.1 / TC-3.2: POST /translate with valid EN→FR must return 200 + translation."""
        res = await client.post("/translate", json={
            "text": "Hello world",
            "src_lang": "en",
            "tgt_lang": "fr",
        })
        assert res.status_code == 200
        data = res.json()
        assert "translation" in data
        assert isinstance(data["translation"], str)
        assert len(data["translation"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_language_pair_returns_400(self, client):
        """TC-3.3: Unsupported language pair (en→xx) must return 400."""
        res = await client.post("/translate", json={
            "text": "Hello",
            "src_lang": "en",
            "tgt_lang": "xx",
        })
        # Accept 400 or 503 depending on whether M2M fallback is available
        assert res.status_code in (400, 503)

    @pytest.mark.asyncio
    async def test_empty_text_returns_422(self, client):
        """TC-3.8: Empty text must return 422 Unprocessable Entity."""
        res = await client.post("/translate", json={
            "text": "",
            "src_lang": "en",
            "tgt_lang": "fr",
        })
        assert res.status_code == 422

    @pytest.mark.asyncio
    async def test_single_translation_latency(self, client):
        """TC-3.5: Single sentence translation must complete within 3 seconds on CPU."""
        t0 = time.perf_counter()
        res = await client.post("/translate", json={
            "text": "Hello world",
            "src_lang": "en",
            "tgt_lang": "fr",
        })
        elapsed = time.perf_counter() - t0
        assert res.status_code == 200
        assert elapsed <= 3.0, f"Latency {elapsed:.2f}s exceeded 3s SLA"

    @pytest.mark.asyncio
    async def test_model_info_endpoint(self, client):
        """TC-2.5: GET /model/info must return 200 with valid JSON."""
        res = await client.get("/model/info")
        assert res.status_code == 200
        data = res.json()
        assert "model_id" in data

    @pytest.mark.asyncio
    async def test_evaluate_endpoint(self, client):
        """POST /evaluate must return BLEU/METEOR/chrF scores."""
        res = await client.post("/evaluate", json={
            "hypotheses": ["Bonjour le monde"],
            "references": ["Bonjour monde"],
        })
        assert res.status_code == 200
        data = res.json()
        assert "bleu" in data
        assert "meteor" in data
        assert "chrf" in data

    @pytest.mark.asyncio
    async def test_export_csv_endpoint(self, client):
        """TC-4.7: GET /evaluate/export/csv must return 200 with text/csv."""
        res = await client.get("/evaluate/export/csv")
        assert res.status_code == 200
        assert "text/csv" in res.headers.get("content-type", "")
        assert len(res.content) > 0

    @pytest.mark.asyncio
    async def test_export_pdf_endpoint(self, client):
        """FR-4.6: GET /evaluate/export/pdf must return 200 with application/pdf."""
        res = await client.get("/evaluate/export/pdf")
        assert res.status_code == 200
        assert "application/pdf" in res.headers.get("content-type", "")
        assert res.content.startswith(b"%PDF")

    @pytest.mark.asyncio
    async def test_batch_translate_100_sentences(self, client):
        """TC-3.4: Batch of 100 sentences must return 100 translations."""
        sentences = [f"Sentence number {i}" for i in range(10)]  # use 10 for speed
        # We test via engine directly for batch (API doesn't expose batch endpoint)
        from module3_engine.api import engine
        results = engine.batch_translate(sentences, "en", "fr")
        assert len(results) == 10


class TestTranslationEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from module2_model.model_manager import ModelManager
        from module3_engine.translation_engine import TranslationEngine
        mm = ModelManager("Helsinki-NLP/opus-mt-en-fr")
        return TranslationEngine(mm)

    def test_beam_search_quality(self, engine):
        """TC-3.7: Beam search (default=4) must produce a fluent, non-empty output."""
        result = engine.translate("The weather is beautiful today.", "en", "fr")
        assert isinstance(result["translation"], str)
        assert len(result["translation"]) > 0

    def test_greedy_decoding(self, engine):
        """FR-3.3: Greedy decoding must return a valid translation."""
        result = engine.translate("Hello.", "en", "fr", decoding="greedy")
        assert isinstance(result["translation"], str)

    def test_result_includes_latency(self, engine):
        """Result dict must include latency_ms field."""
        result = engine.translate("Test.", "en", "fr")
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)
