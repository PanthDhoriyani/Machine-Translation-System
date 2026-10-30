"""
generate_pdf_report.py — Converts TRANSLATE_CLI_PROJECT_REPORT.md into a professional PDF
using ReportLab with custom styling, running headers/footers, tables, and screenshot segments.
"""

import os
import sys
import re
from datetime import datetime

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54,
                842 - 36,
                "NLPA Project 2026 | Standalone Neural Machine Translation System (translate_cli.py)",
            )
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 842 - 42, 595 - 54, 842 - 42)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 595 - 54, 45)

        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(595 - 54, 32, footer_text)
        self.drawString(54, 32, "Machine Translation System — Comprehensive Technical Report")
        self.restoreState()


def build_pdf(md_path: str, pdf_path: str):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#1E293B"),
        alignment=0,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#312E81"),  # deep indigo
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        alignment=1,  # Center
        spaceAfter=8,
    )

    box_text_style = ParagraphStyle(
        "BoxTextStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4338CA"),
        alignment=1,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Standalone Terminal-Based Neural Machine Translation System", title_style))
    story.append(
        Paragraph(
            "<b>Course:</b> Natural Language Processing & Applications (NLPA 2026) &nbsp;|&nbsp; "
            "<b>Module:</b> translate_cli.py &nbsp;|&nbsp; "
            "<b>Repo:</b> github.com/PanthDhoriyani/Machine-Translation-System",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4338CA"), spaceAfter=10))

    # Student info table
    info_data = [
        [
            Paragraph("<b>Author / Student Name:</b>", table_cell_style),
            Paragraph("<i>[Insert Name / Roll Number]</i>", table_cell_style),
            Paragraph("<b>Framework:</b>", table_cell_style),
            Paragraph("PyTorch 2.6.0+cu124, Transformers", table_cell_style),
        ],
        [
            Paragraph("<b>Primary Script:</b>", table_cell_style),
            Paragraph("<code>translate_cli.py</code> (Standalone)", table_cell_style),
            Paragraph("<b>Hardware Acceleration:</b>", table_cell_style),
            Paragraph("NVIDIA CUDA GPU / CPU Auto-detect", table_cell_style),
        ],
    ]
    t_info = Table(info_data, colWidths=[120, 130, 110, 135])
    t_info.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_info)
    story.append(Spacer(1, 10))

    # Abstract
    story.append(Paragraph("1. Executive Summary & Objectives", h1_style))
    story.append(
        Paragraph(
            "Machine Translation (MT) automates the conversion of text between natural languages while preserving semantic nuance. "
            "While enterprise MT platforms incorporate distributed microservices and front-end web layers, lightweight local edge inference "
            "requires a self-contained, dependency-minimal interface. This project develops <b><code>translate_cli.py</code></b>, a standalone "
            "single-file CLI application leveraging pre-trained MarianMT sequence-to-sequence Transformer models. "
            "Key objectives achieved include: (1) zero-server operational overhead, (2) sub-second translation on CUDA GPU, "
            "(3) interactive multi-turn REPL mode with live pair switching, (4) one-shot CLI execution, and (5) automated BLEU metric verification.",
            body_style,
        )
    )

    # Architecture
    story.append(Paragraph("2. System Architecture & Processing Pipeline", h1_style))
    story.append(
        Paragraph(
            "The system follows an encapsulated sequential pipeline:",
            body_style,
        )
    )

    pipeline_items = [
        "<b>Input Normalization:</b> HTML sanitization, space collapsing, and unicode NFC canonical decomposition.",
        "<b>Model Manager & Cache:</b> Dynamic mapping of language pairs to HuggingFace repository IDs with in-memory caching to eliminate redundant loads.",
        "<b>Subword Tokenization:</b> SentencePiece BPE encoding with truncation capped at 512 tokens.",
        "<b>Seq2Seq Generation:</b> Transformer Encoder-Decoder inference supporting configurable Beam Search (default=4) and Greedy search.",
        "<b>Detokenization & Reporting:</b> Output string decoding, latency timing in milliseconds, and optional sentence BLEU score calculation.",
    ]
    for item in pipeline_items:
        story.append(Paragraph(f"• {item}", bullet_style))

    story.append(Spacer(1, 6))

    # Supported Models Table
    story.append(Paragraph("3. Model Catalogue & Supported Language Pairs", h1_style))
    model_rows = [
        [
            Paragraph("Source", table_header_style),
            Paragraph("Target", table_header_style),
            Paragraph("HuggingFace Model ID", table_header_style),
            Paragraph("Architecture", table_header_style),
            Paragraph("Params", table_header_style),
        ],
        [Paragraph("English (en)", table_cell_style), Paragraph("French (fr)", table_cell_style), Paragraph("Helsinki-NLP/opus-mt-en-fr", table_cell_style), Paragraph("6L-6L Transformer", table_cell_style), Paragraph("74.4 M", table_cell_style)],
        [Paragraph("English (en)", table_cell_style), Paragraph("German (de)", table_cell_style), Paragraph("Helsinki-NLP/opus-mt-en-de", table_cell_style), Paragraph("6L-6L Transformer", table_cell_style), Paragraph("74.4 M", table_cell_style)],
        [Paragraph("English (en)", table_cell_style), Paragraph("Spanish (es)", table_cell_style), Paragraph("Helsinki-NLP/opus-mt-en-es", table_cell_style), Paragraph("6L-6L Transformer", table_cell_style), Paragraph("74.4 M", table_cell_style)],
        [Paragraph("English (en)", table_cell_style), Paragraph("Hindi (hi)", table_cell_style), Paragraph("Helsinki-NLP/opus-mt-en-hi", table_cell_style), Paragraph("6L-6L Transformer", table_cell_style), Paragraph("74.4 M", table_cell_style)],
        [Paragraph("French (fr)", table_cell_style), Paragraph("English (en)", table_cell_style), Paragraph("Helsinki-NLP/opus-mt-fr-en", table_cell_style), Paragraph("6L-6L Transformer", table_cell_style), Paragraph("74.4 M", table_cell_style)],
        [Paragraph("German (de)", table_cell_style), Paragraph("English (en)", table_cell_style), Paragraph("Helsinki-NLP/opus-mt-de-en", table_cell_style), Paragraph("6L-6L Transformer", table_cell_style), Paragraph("74.4 M", table_cell_style)],
    ]
    t_models = Table(model_rows, colWidths=[65, 65, 185, 115, 65])
    t_models.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(t_models)
    story.append(Spacer(1, 10))

    # Benchmark results
    story.append(Paragraph("4. Experimental Performance & Verification", h1_style))
    bench_rows = [
        [
            Paragraph("Input Sentence", table_header_style),
            Paragraph("Target", table_header_style),
            Paragraph("Output Translation", table_header_style),
            Paragraph("Latency", table_header_style),
            Paragraph("Hardware", table_header_style),
        ],
        [
            Paragraph("Hello, welcome to our machine translation system.", table_cell_style),
            Paragraph("French (fr)", table_cell_style),
            Paragraph("Bonjour, bienvenue dans notre système de traduction automatique.", table_cell_style),
            Paragraph("896.2 ms", table_cell_style),
            Paragraph("CUDA GPU", table_cell_style),
        ],
        [
            Paragraph("Good morning! How are you doing today?", table_cell_style),
            Paragraph("German (de)", table_cell_style),
            Paragraph("Guten Morgen, wie geht es dir heute?", table_cell_style),
            Paragraph("891.1 ms", table_cell_style),
            Paragraph("CUDA GPU", table_cell_style),
        ],
        [
            Paragraph("Subsequent session translations (cached)", table_cell_style),
            Paragraph("French (fr)", table_cell_style),
            Paragraph("Fast in-memory cache lookup", table_cell_style),
            Paragraph("&lt; 50 ms", table_cell_style),
            Paragraph("CUDA GPU", table_cell_style),
        ],
    ]
    t_bench = Table(bench_rows, colWidths=[130, 55, 195, 55, 60])
    t_bench.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(t_bench)
    story.append(Spacer(1, 12))

    # Helper function to generate standardized Screenshot submission segments
    def create_screenshot_segment(segment_num, title, command, caption):
        elements = []
        elements.append(Paragraph(f"<b>Segment {segment_num}: {title}</b>", h2_style))

        # Command banner
        cmd_p = Paragraph(f"<code>{command}</code>", code_style)
        t_cmd = Table([[cmd_p]], colWidths=[495])
        t_cmd.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        elements.append(t_cmd)
        elements.append(Spacer(1, 4))

        # Screenshot submission box
        box_content = [
            Spacer(1, 14),
            Paragraph(f"📷 <b>[ SUBMIT SCREENSHOT HERE: Segment {segment_num} ]</b>", box_text_style),
            Spacer(1, 3),
            Paragraph("<i>Paste terminal screenshot here before final submission</i>", caption_style),
            Spacer(1, 14),
        ]
        t_box = Table([[box_content]], colWidths=[495])
        t_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
                ("LINEBEFORE", (0, 0), (0, 0), 2.5, colors.HexColor("#4338CA")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        elements.append(t_box)
        elements.append(Paragraph(caption, caption_style))
        elements.append(Spacer(1, 6))
        return KeepTogether(elements)

    # Screenshot Segments Section
    story.append(Paragraph("5. Verification Screenshots & Demonstration Segments", h1_style))
    story.append(
        Paragraph(
            "Attach terminal screenshots into the respective segment frames below for project validation and grading:",
            body_style,
        )
    )
    story.append(Spacer(1, 4))

    # Segment 1
    story.append(
        create_screenshot_segment(
            1,
            "CLI Command-Line Help Output (--help)",
            "python translate_cli.py --help",
            "Figure 1: Terminal display of command line argument parser with --src, --tgt, --beam, and --ref options.",
        )
    )

    # Segment 2
    story.append(
        create_screenshot_segment(
            2,
            "One-Shot English to French Translation",
            'python translate_cli.py "Hello, welcome to our machine translation system." --tgt fr',
            "Figure 2: Execution output for English-to-French translation demonstrating model loading, generation, and 896ms latency.",
        )
    )

    # Segment 3
    story.append(
        create_screenshot_segment(
            3,
            "One-Shot English to German Translation",
            'python translate_cli.py "Good morning! How are you doing today?" --tgt de',
            "Figure 3: Execution output for English-to-German translation utilizing Helsinki-NLP/opus-mt-en-de model.",
        )
    )

    # Segment 4
    story.append(
        create_screenshot_segment(
            4,
            "Interactive REPL Session & Dynamic Pair Switching",
            "python translate_cli.py --interactive",
            "Figure 4: Multi-turn interactive terminal mode showing real-time translation, pair switching (:pair en de), and exit.",
        )
    )

    # Segment 5
    story.append(
        create_screenshot_segment(
            5,
            "Reference-Based Quality Evaluation (BLEU Score)",
            'python translate_cli.py "Hello world" --tgt fr --ref "Bonjour le monde"',
            "Figure 5: Translation accuracy benchmark verifying hypothesis vs reference with BLEU = 100.0 / 100.",
        )
    )

    # Conclusion
    story.append(Paragraph("6. Conclusion & Summary", h1_style))
    story.append(
        Paragraph(
            "The standalone <code>translate_cli.py</code> implementation fulfills all functional translation and performance requirements "
            "specified in the NLPA 2026 project curriculum. By packaging state-of-the-art Transformer models into a single script with "
            "native GPU acceleration, memory caching, and dual-mode execution (CLI and REPL), it bridges the gap between complex research "
            "architectures and practical edge utility.",
            body_style,
        )
    )

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built PDF: {pdf_path}")


if __name__ == "__main__":
    md_file = "d:/anything/nlpa project/TRANSLATE_CLI_PROJECT_REPORT.md"
    pdf_file = "d:/anything/nlpa project/TRANSLATE_CLI_PROJECT_REPORT.pdf"
    build_pdf(md_file, pdf_file)
