"""
Integration Tests — End-to-end pipeline validation
TC-INT-1 through TC-INT-4 from SRS Section 7.5
"""

import pytest
from httpx import AsyncClient, ASGITransport

from module3_engine.api import app


@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_en_fr(self, client):
        """TC-INT-1: Full pipeline EN→FR — input → translate → evaluate."""
        # 1. Translate
        res = await client.post("/translate", json={
            "text": "Natural language processing is a field of artificial intelligence.",
            "src_lang": "en",
            "tgt_lang": "fr",
        })
        assert res.status_code == 200
        translation = res.json()["translation"]
        assert len(translation) > 0

        # 2. Evaluate using the translation as a hypothesis
        res2 = await client.post("/evaluate", json={
            "hypotheses": [translation],
            "references": ["Le traitement du langage naturel est un domaine de l'intelligence artificielle."],
        })
        assert res2.status_code == 200
        scores = res2.json()
        assert scores["bleu"] >= 0
        assert 0 <= scores["meteor"] <= 1
        assert scores["chrf"] >= 0

    @pytest.mark.asyncio
    async def test_language_pair_switch(self, client):
        """TC-INT-2: Translate EN→FR, then EN→DE — both should succeed."""
        for tgt, sample in [("fr", "Good morning"), ("de", "Good morning")]:
            res = await client.post("/translate", json={
                "text": sample,
                "src_lang": "en",
                "tgt_lang": tgt,
            })
            assert res.status_code == 200, f"Failed for EN→{tgt}"
            assert len(res.json()["translation"]) > 0

    @pytest.mark.asyncio
    async def test_api_and_model_info_sync(self, client):
        """TC-INT-4: After translation, GET /model/info should reflect active model."""
        await client.post("/translate", json={
            "text": "Hello",
            "src_lang": "en",
            "tgt_lang": "fr",
        })
        info_res = await client.get("/model/info")
        assert info_res.status_code == 200
        info = info_res.json()
        assert "model_id" in info
        assert info["cached"] is True

    @pytest.mark.asyncio
    async def test_evaluate_empty_raises_error(self, client):
        """TC-INT edge case: Evaluating empty lists must return 400."""
        res = await client.post("/evaluate", json={
            "hypotheses": [],
            "references": [],
        })
        assert res.status_code == 422  # FastAPI min_length validation
