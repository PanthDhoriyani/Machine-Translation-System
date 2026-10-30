"""
generate_docx_report.py — Generates a professional Microsoft Word (.docx) project report
from TRANSLATE_CLI_PROJECT_REPORT.md with designated screenshot insertion boxes.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Set individual borders on a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
    borders = {"top": top, "bottom": bottom, "left": left, "right": right}
    for side, border in borders.items():
        if border:
            el = parse_xml(
                f'<w:{side} {nsdecls("w")} w:val="{border.get("val", "single")}" '
                f'w:sz="{border.get("sz", "4")}" w:space="0" w:color="{border.get("color", "auto")}"/>'
            )
            tcBorders.append(el)
        else:
            el = parse_xml(f'<w:{side} {nsdecls("w")} w:val="none"/>')
            tcBorders.append(el)
    tcPr.append(tcBorders)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner padding for a table cell (in twips, 1 pt = 20 twips)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def create_docx_report(output_path: str):
    doc = Document()

    # Page Margins: 0.75 in (54 pt)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Title
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("Standalone Terminal-Based Neural Machine Translation System")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    p_title.paragraph_format.space_after = Pt(4)

    # Subtitle
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run(
        "Natural Language Processing & Applications (NLPA 2026)  |  "
        "Project Report  |  Script: translate_cli.py"
    )
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    p_sub.paragraph_format.space_after = Pt(12)

    # Student & Project Information Table
    t_info = doc.add_table(rows=3, cols=4)
    t_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        [("Author / Student Name:", True), ("[Insert Name / Roll Number]", False), ("Framework:", True), ("PyTorch 2.6.0+cu124, Transformers", False)],
        [("Course & Branch:", True), ("NLPA 2026 — Computer Science & Eng.", False), ("Hardware:", True), ("NVIDIA CUDA GPU / CPU Auto-detect", False)],
        [("Primary Script:", True), ("translate_cli.py (Single-File Standalone)", False), ("GitHub Repo:", True), ("github.com/PanthDhoriyani/Machine-Translation-System", False)],
    ]
    col_widths = [Inches(1.8), Inches(2.2), Inches(1.2), Inches(1.8)]

    for row_idx, row in enumerate(info_data):
        for col_idx, (text, is_bold) in enumerate(row):
            cell = t_info.cell(row_idx, col_idx)
            cell.width = col_widths[col_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(text)
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            if is_bold:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                set_cell_background(cell, "F1F5F9")
            else:
                run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
                set_cell_background(cell, "F8FAFC")
            set_cell_borders(cell,
                top={"val": "single", "sz": "4", "color": "CBD5E1"},
                bottom={"val": "single", "sz": "4", "color": "CBD5E1"},
                left={"val": "single", "sz": "4", "color": "CBD5E1"},
                right={"val": "single", "sz": "4", "color": "CBD5E1"},
            )
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Helper for Headings
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)  # Deep Indigo
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = "Calibri"
        r_bold.font.size = Pt(10)
        r_bold.font.bold = True
        r_bold.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

        r_text = p.add_run(" " + text)
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(10)
        r_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        return p

    # 1. Executive Summary
    add_h1("1. Executive Summary & Project Objectives")
    add_body(
        "Machine Translation (MT) automates the conversion of natural language text while preserving contextual semantics "
        "and fluency. While full-stack MT platforms introduce substantial microservice and web-server overheads, rapid edge "
        "prototyping and local offline inference demand a self-contained, zero-dependency interface. "
        "This project delivers translate_cli.py: a standalone single-file CLI application leveraging pre-trained MarianMT "
        "sequence-to-sequence Transformer models."
    )
    add_body(
        "Key capabilities implemented: (1) Zero-server overhead requiring no background daemon, (2) Sub-second translation "
        "leveraging NVIDIA CUDA GPU acceleration with automatic CPU fallback, (3) Interactive multi-turn REPL mode with live "
        "language pair switching, (4) One-shot command-line invocation for scripts, and (5) Integrated BLEU metric evaluation."
    )

    # 2. System Architecture
    add_h1("2. System Architecture & Processing Pipeline")
    add_body("The translation pipeline is organized into five core stages:")
    add_bullet("Input Normalization:", "Sanitizes HTML tags, standardizes whitespace, and enforces NFC Unicode normalization.")
    add_bullet("Language Routing & Caching:", "Maps language pairs to HuggingFace model IDs; caches loaded models in memory for instant reuse.")
    add_bullet("Subword Tokenization:", "SentencePiece BPE tokenization capped at 512 tokens with attention masks.")
    add_bullet("Seq2Seq Inference:", "Transformer Encoder-Decoder inference supporting configurable Beam Search (default=4) and Greedy decoding.")
    add_bullet("Detokenization & Metrics:", "Converts token IDs to clean natural text, records millisecond latency, and computes sentence BLEU.")

    # 3. Model Catalogue
    add_h1("3. Model Catalogue & Supported Language Pairs")
    t_cat = doc.add_table(rows=7, cols=5)
    t_cat.alignment = WD_TABLE_ALIGNMENT.CENTER
    cat_headers = ["Source", "Target", "HuggingFace Model ID", "Architecture", "Params"]
    cat_data = [
        ["English (en)", "French (fr)", "Helsinki-NLP/opus-mt-en-fr", "6L-6L Transformer", "74.4 M"],
        ["English (en)", "German (de)", "Helsinki-NLP/opus-mt-en-de", "6L-6L Transformer", "74.4 M"],
        ["English (en)", "Spanish (es)", "Helsinki-NLP/opus-mt-en-es", "6L-6L Transformer", "74.4 M"],
        ["English (en)", "Hindi (hi)", "Helsinki-NLP/opus-mt-en-hi", "6L-6L Transformer", "74.4 M"],
        ["French (fr)", "English (en)", "Helsinki-NLP/opus-mt-fr-en", "6L-6L Transformer", "74.4 M"],
        ["German (de)", "English (en)", "Helsinki-NLP/opus-mt-de-en", "6L-6L Transformer", "74.4 M"],
    ]
    cat_widths = [Inches(1.2), Inches(1.2), Inches(2.2), Inches(1.6), Inches(0.8)]

    # Header row
    for col_idx, h in enumerate(cat_headers):
        cell = t_cat.cell(0, col_idx)
        cell.width = cat_widths[col_idx]
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(cell, "4338CA")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

    for r_idx, row in enumerate(cat_data, start=1):
        bg = "FFFFFF" if r_idx % 2 == 1 else "F8FAFC"
        for c_idx, val in enumerate(row):
            cell = t_cat.cell(r_idx, c_idx)
            cell.width = cat_widths[c_idx]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            set_cell_background(cell, bg)
            set_cell_borders(cell,
                top={"val": "single", "sz": "4", "color": "E2E8F0"},
                bottom={"val": "single", "sz": "4", "color": "E2E8F0"},
                left={"val": "single", "sz": "4", "color": "E2E8F0"},
                right={"val": "single", "sz": "4", "color": "E2E8F0"},
            )
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 4. Experimental Results Table
    add_h1("4. Experimental Performance & Latency Benchmarks")
    t_bench = doc.add_table(rows=4, cols=5)
    t_bench.alignment = WD_TABLE_ALIGNMENT.CENTER
    bench_headers = ["Input Sentence", "Target", "Translation Output", "Latency", "Device"]
    bench_data = [
        ["Hello, welcome to our machine translation system.", "French (fr)", "Bonjour, bienvenue dans notre système de traduction automatique.", "896.2 ms (warmup)", "CUDA GPU"],
        ["Good morning! How are you doing today?", "German (de)", "Guten Morgen, wie geht es dir heute?", "891.1 ms (warmup)", "CUDA GPU"],
        ["Subsequent session queries (in-memory cached)", "French (fr)", "Instant cached model inference", "< 50 ms (cached)", "CUDA GPU"],
    ]
    bench_widths = [Inches(1.8), Inches(1.0), Inches(2.2), Inches(1.2), Inches(0.8)]

    for col_idx, h in enumerate(bench_headers):
        cell = t_bench.cell(0, col_idx)
        cell.width = bench_widths[col_idx]
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(cell, "1E293B")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

    for r_idx, row in enumerate(bench_data, start=1):
        bg = "FFFFFF" if r_idx % 2 == 1 else "F8FAFC"
        for c_idx, val in enumerate(row):
            cell = t_bench.cell(r_idx, c_idx)
            cell.width = bench_widths[c_idx]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            set_cell_background(cell, bg)
            set_cell_borders(cell,
                top={"val": "single", "sz": "4", "color": "E2E8F0"},
                bottom={"val": "single", "sz": "4", "color": "E2E8F0"},
                left={"val": "single", "sz": "4", "color": "E2E8F0"},
                right={"val": "single", "sz": "4", "color": "E2E8F0"},
            )
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 5. Screenshot Submission Segments
    add_h1("5. Screenshot Submission Segments")
    add_body(
        "Paste or insert your terminal screenshots into each designated box below. "
        "You can directly drag-and-drop or paste (Ctrl+V) images directly into the designated boxes in Microsoft Word."
    )

    def add_screenshot_box(segment_num, title, command, caption):
        add_h2(f"Segment {segment_num}: {title}")

        # Command Box
        t_cmd = doc.add_table(rows=1, cols=1)
        t_cmd.alignment = WD_TABLE_ALIGNMENT.CENTER
        c_cmd = t_cmd.cell(0, 0)
        c_cmd.width = Inches(7.0)
        p_cmd = c_cmd.paragraphs[0]
        p_cmd.paragraph_format.space_before = Pt(3)
        p_cmd.paragraph_format.space_after = Pt(3)
        r_cmd = p_cmd.add_run(f"Command:  {command}")
        r_cmd.font.name = "Consolas"
        r_cmd.font.size = Pt(9.5)
        r_cmd.font.bold = True
        r_cmd.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        set_cell_background(c_cmd, "F1F5F9")
        set_cell_borders(c_cmd,
            top={"val": "single", "sz": "4", "color": "CBD5E1"},
            bottom={"val": "single", "sz": "4", "color": "CBD5E1"},
            left={"val": "single", "sz": "18", "color": "4338CA"},  # thick accent left border
            right={"val": "single", "sz": "4", "color": "CBD5E1"},
        )
        set_cell_margins(c_cmd, top=80, bottom=80, left=140, right=140)

        # Large Screenshot Insertion Box
        t_ss = doc.add_table(rows=1, cols=1)
        t_ss.alignment = WD_TABLE_ALIGNMENT.CENTER
        c_ss = t_ss.cell(0, 0)
        c_ss.width = Inches(7.0)
        p_ss = c_ss.paragraphs[0]
        p_ss.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ss.paragraph_format.space_before = Pt(35)
        p_ss.paragraph_format.space_after = Pt(6)

        r_icon = p_ss.add_run("📸 [ INSERT SCREENSHOT HERE: Segment " + str(segment_num) + " ]\n")
        r_icon.font.name = "Calibri"
        r_icon.font.size = Pt(12)
        r_icon.font.bold = True
        r_icon.font.color.rgb = RGBColor(0x43, 0x38, 0xCA)

        r_hint = p_ss.add_run("(Click inside this box and paste [Ctrl + V] your screenshot)")
        r_hint.font.name = "Calibri"
        r_hint.font.size = Pt(9.5)
        r_hint.font.italic = True
        r_hint.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

        p_space = c_ss.add_paragraph()
        p_space.paragraph_format.space_after = Pt(35)

        set_cell_background(c_ss, "F8FAFC")
        set_cell_borders(c_ss,
            top={"val": "dashed", "sz": "12", "color": "94A3B8"},
            bottom={"val": "dashed", "sz": "12", "color": "94A3B8"},
            left={"val": "dashed", "sz": "12", "color": "94A3B8"},
            right={"val": "dashed", "sz": "12", "color": "94A3B8"},
        )
        set_cell_margins(c_ss, top=140, bottom=140, left=140, right=140)

        # Caption
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(4)
        p_cap.paragraph_format.space_after = Pt(12)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = "Calibri"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # 5 Segments
    add_screenshot_box(
        1,
        "CLI Command-Line Help Output (--help)",
        "python translate_cli.py --help",
        "Figure 1: Terminal display of command line argument parser showing --src, --tgt, --beam, and --interactive flags.",
    )

    add_screenshot_box(
        2,
        "One-Shot English to French Translation",
        'python translate_cli.py "Hello, welcome to our machine translation system." --tgt fr',
        "Figure 2: Execution output for English-to-French translation demonstrating model loading, generation, and 896ms latency.",
    )

    add_screenshot_box(
        3,
        "One-Shot English to German Translation",
        'python translate_cli.py "Good morning! How are you doing today?" --tgt de',
        "Figure 3: Execution output for English-to-German translation utilizing Helsinki-NLP/opus-mt-en-de model.",
    )

    add_screenshot_box(
        4,
        "Interactive REPL Session & Dynamic Pair Switching",
        "python translate_cli.py --interactive",
        "Figure 4: Multi-turn interactive terminal mode showing real-time translation, pair switching (:pair en de), and exit.",
    )

    add_screenshot_box(
        5,
        "Reference-Based Quality Evaluation (BLEU Score)",
        'python translate_cli.py "Hello world" --tgt fr --ref "Bonjour le monde"',
        "Figure 5: Translation accuracy benchmark verifying hypothesis vs reference with BLEU = 100.0 / 100.",
    )

    # 6. Conclusion
    add_h1("6. Conclusion")
    add_body(
        "The standalone translate_cli.py implementation successfully demonstrates high-accuracy, zero-server neural machine translation "
        "leveraging pre-trained MarianMT models on NVIDIA CUDA hardware. With both interactive and one-shot CLI execution modalities, "
        "the tool fulfills all academic objectives for the NLPA 2026 project curriculum."
    )

    doc.save(output_path)
    print(f"[OK] Word document successfully created: {output_path}")


if __name__ == "__main__":
    docx_file = "d:/anything/nlpa project/TRANSLATE_CLI_PROJECT_REPORT.docx"
    create_docx_report(docx_file)
