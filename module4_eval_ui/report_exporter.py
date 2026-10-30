"""
ReportExporter — Exports evaluation results as CSV or PDF.

FR-4.6: Export evaluation results as CSV/PDF report
"""

import csv
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ReportExporter:
    """Serialises evaluation reports to CSV and PDF formats.

    Attributes:
        report (dict): Evaluation metrics dict from Evaluator.generate_report().
        hypotheses (list[str]): Model-generated translations.
        references (list[str]): Reference translations.
    """

    def __init__(
        self,
        report: dict,
        hypotheses: Optional[list[str]] = None,
        references: Optional[list[str]] = None,
    ):
        """Initialise the exporter.

        Args:
            report (dict): Dict containing bleu, meteor, chrf keys.
            hypotheses (list[str]): Optional per-sentence hypothesis list.
            references (list[str]): Optional per-sentence reference list.
        """
        self.report = report
        self.hypotheses = hypotheses or []
        self.references = references or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_csv_bytes(self) -> bytes:
        """Serialise the report to CSV bytes. (FR-4.6, TC-4.7)

        Produces a two-section CSV:
          1. Summary metrics (BLEU, METEOR, chrF)
          2. Per-sentence hypothesis / reference pairs (if available)

        Returns:
            UTF-8 encoded CSV content as bytes.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # -- Summary section
        writer.writerow(["Machine Translation Evaluation Report"])
        writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])
        writer.writerow(["Metric", "Score"])
        writer.writerow(["BLEU", self.report.get("bleu", "")])
        writer.writerow(["METEOR", self.report.get("meteor", "")])
        writer.writerow(["chrF", self.report.get("chrf", "")])
        writer.writerow(["Sentences", self.report.get("num_sentences", "")])

        # -- Per-sentence section
        if self.hypotheses and self.references:
            writer.writerow([])
            writer.writerow(["#", "Hypothesis", "Reference"])
            for i, (hyp, ref) in enumerate(zip(self.hypotheses, self.references), start=1):
                writer.writerow([i, hyp, ref])

        return output.getvalue().encode("utf-8")

    def save_csv(self, path: str) -> Path:
        """Save the report as a CSV file on disk.

        Args:
            path (str): Destination file path.

        Returns:
            Path object pointing to the saved file.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(self.to_csv_bytes())
        logger.info("CSV report saved to '%s'.", file_path)
        return file_path

    def to_pdf_bytes(self) -> bytes:
        """Serialise the report to PDF bytes. (FR-4.6)

        Uses reportlab for PDF generation.

        Returns:
            PDF content as bytes.

        Raises:
            ImportError: If reportlab is not installed.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
        except ImportError as exc:
            raise ImportError(
                "reportlab is required for PDF export. Run: pip install reportlab"
            ) from exc

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Machine Translation Evaluation Report", styles["Title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 20))

        # Metrics table
        metrics_data = [
            ["Metric", "Score"],
            ["BLEU", f"{self.report.get('bleu', 'N/A')}"],
            ["METEOR", f"{self.report.get('meteor', 'N/A')}"],
            ["chrF", f"{self.report.get('chrf', 'N/A')}"],
            ["Sentences", f"{self.report.get('num_sentences', 'N/A')}"],
        ]
        t = Table(metrics_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)

        doc.build(story)
        return buf.getvalue()

    def save_pdf(self, path: str) -> Path:
        """Save the report as a PDF file on disk.

        Args:
            path (str): Destination file path.

        Returns:
            Path object pointing to the saved file.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(self.to_pdf_bytes())
        logger.info("PDF report saved to '%s'.", file_path)
        return file_path
