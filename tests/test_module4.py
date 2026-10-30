"""
Test Suite — Module 4: Evaluation & Report Export
TC-4.1 through TC-4.9 from SRS Section 7.4
"""

import pytest
from module4_eval_ui.evaluator import Evaluator
from module4_eval_ui.report_exporter import ReportExporter


class TestEvaluator:
    def test_bleu_perfect_match(self):
        """TC-4.1: Hypothesis == Reference must yield BLEU = 100."""
        ev = Evaluator(["Bonjour le monde"], ["Bonjour le monde"])
        score = ev.compute_bleu()
        assert score == pytest.approx(100.0, abs=0.01)

    def test_bleu_no_match(self):
        """TC-4.2: Completely different hypothesis must yield BLEU ≈ 0."""
        ev = Evaluator(["xyz abc def ghi jkl"], ["Bonjour le monde entier"])
        score = ev.compute_bleu()
        assert score < 1.0

    def test_meteor_score_range(self):
        """TC-4.3: METEOR score must be in [0, 1]."""
        ev = Evaluator(["The cat sat on the mat"], ["The cat is sitting on the mat"])
        score = ev.compute_meteor()
        assert 0.0 <= score <= 1.0

    def test_chrf_score_range(self):
        """TC-4.4: chrF score must be in [0, 100]."""
        ev = Evaluator(["Bonjour le monde"], ["Bonjour monde"])
        score = ev.compute_chrf()
        assert 0.0 <= score <= 100.0

    def test_empty_hypotheses_raises(self):
        """TC-4.9: Empty hypotheses list must raise ValueError."""
        with pytest.raises(ValueError):
            Evaluator([], ["reference"])

    def test_empty_references_raises(self):
        """TC-4.9: Empty references list must raise ValueError."""
        with pytest.raises(ValueError):
            Evaluator(["hypothesis"], [])

    def test_length_mismatch_raises(self):
        """Length mismatch between hyp and ref must raise ValueError."""
        with pytest.raises(ValueError):
            Evaluator(["a", "b"], ["c"])

    def test_generate_report_keys(self):
        """generate_report() must return dict with bleu, meteor, chrf, num_sentences."""
        ev = Evaluator(["Hello world"], ["Hello world"])
        report = ev.generate_report()
        assert "bleu" in report
        assert "meteor" in report
        assert "chrf" in report
        assert "num_sentences" in report
        assert report["num_sentences"] == 1


class TestReportExporter:
    @pytest.fixture
    def sample_report(self):
        ev = Evaluator(["Bonjour le monde"], ["Bonjour monde"])
        return ev.generate_report()

    @pytest.fixture
    def exporter(self, sample_report):
        return ReportExporter(
            report=sample_report,
            hypotheses=["Bonjour le monde"],
            references=["Bonjour monde"],
        )

    def test_to_csv_bytes_returns_bytes(self, exporter):
        """TC-4.7: to_csv_bytes() must return bytes that decode as valid UTF-8."""
        data = exporter.to_csv_bytes()
        assert isinstance(data, bytes)
        text = data.decode("utf-8")
        assert "BLEU" in text

    def test_csv_contains_metrics(self, exporter):
        """TC-4.7: CSV content must include all metric names."""
        text = exporter.to_csv_bytes().decode("utf-8")
        assert "METEOR" in text
        assert "chrF" in text

    def test_save_csv_creates_file(self, exporter, tmp_path):
        """TC-4.7: save_csv() must create a file at the target path."""
        path = exporter.save_csv(str(tmp_path / "report.csv"))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_to_pdf_bytes_returns_pdf(self, exporter):
        """TC-4.7 / FR-4.6: to_pdf_bytes() must produce a valid PDF byte stream."""
        pdf_data = exporter.to_pdf_bytes()
        assert isinstance(pdf_data, bytes)
        assert pdf_data.startswith(b"%PDF")

    def test_save_pdf_creates_file(self, exporter, tmp_path):
        """FR-4.6: save_pdf() must create a PDF file at the target path."""
        path = exporter.save_pdf(str(tmp_path / "report.pdf"))
        assert path.exists()
        assert path.stat().st_size > 0
