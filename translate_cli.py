#!/usr/bin/env python3
"""
Single-File Terminal-Based Machine Translation System
NLPA Project 2026 — Simplest Standalone Version

Usage:
  Interactive Mode:
      python translate_cli.py
  One-shot CLI Translation:
      python translate_cli.py "Hello, how are you?" --tgt fr
      python translate_cli.py "Good morning" --src en --tgt de --beam 4
  Quick Evaluation:
      python translate_cli.py "Hello world" --tgt fr --ref "Bonjour le monde"
"""

import argparse
import os
import sys
import time
import unicodedata
import re
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress Hugging Face symlinks noise on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# Supported Language Pairs & Models (Helsinki-NLP MarianMT)
# ---------------------------------------------------------------------------
MODEL_MAP = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
}

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
}

# In-memory model cache
_CACHE = {}


def clean_text(text: str) -> str:
    """Normalize input: strip HTML, normalize unicode NFC, collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()


def get_model_and_tokenizer(model_id: str, device: str):
    """Load and cache HuggingFace MarianMT model and tokenizer."""
    if model_id in _CACHE:
        return _CACHE[model_id]

    print(f"[*] Loading model '{model_id}' on {device.upper()} (first time may take a moment) ...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        print("[!] Error: 'transformers' or 'torch' not installed.")
        print("    Please run: pip install transformers torch sentencepiece")
        sys.exit(1)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    elapsed = time.perf_counter() - t0
    print(f"[OK] Model ready ({elapsed:.2f}s)\n")

    _CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


def translate_text(
    text: str,
    src_lang: str = "en",
    tgt_lang: str = "fr",
    beam_size: int = 4,
    device: Optional[str] = None,
) -> tuple[str, float]:
    """Translate a text string and return (translation, latency_seconds)."""
    text = clean_text(text)
    if not text:
        return "", 0.0

    pair = (src_lang.lower(), tgt_lang.lower())
    if pair not in MODEL_MAP:
        supported = ", ".join(f"{s}->{t}" for s, t in MODEL_MAP.keys())
        raise ValueError(
            f"Unsupported language pair '{src_lang}->{tgt_lang}'.\n"
            f"Supported pairs: {supported}"
        )

    model_id = MODEL_MAP[pair]

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model, tokenizer = get_model_and_tokenizer(model_id, device)

    t0 = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        if beam_size <= 1:
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        else:
            generated = model.generate(
                **inputs,
                num_beams=beam_size,
                max_new_tokens=512,
                early_stopping=True,
            )

    translation = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    elapsed = time.perf_counter() - t0

    return translation, elapsed


def compute_quick_bleu(hypothesis: str, reference: str) -> Optional[float]:
    """Calculate BLEU score if sacrebleu is available."""
    try:
        import sacrebleu
        score = sacrebleu.sentence_bleu(hypothesis, [reference]).score
        return round(score, 2)
    except Exception:
        return None


def run_interactive(src_lang: str = "en", tgt_lang: str = "fr", beam_size: int = 4):
    """Run an interactive CLI loop in the terminal."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print("=" * 65)
    print("      NEURAL MACHINE TRANSLATION SYSTEM — TERMINAL CLI")
    print(f"      Device: {device.upper()} | Default Pair: {src_lang.upper()} -> {tgt_lang.upper()}")
    print("=" * 65)
    print("Commands:")
    print("  :pair <src> <tgt>   Switch language pair (e.g., :pair en de)")
    print("  :beam <1-8>         Change beam size (1 = greedy, 4 = default)")
    print("  :pairs              List all available language pairs")
    print("  :quit or :q         Exit the program")
    print("-" * 65)

    while True:
        try:
            src_name = LANGUAGE_NAMES.get(src_lang, src_lang.upper())
            tgt_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang.upper())
            prompt = f"\n[{src_name} -> {tgt_name}] Enter text: "
            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input in (":quit", ":q", "exit", "quit"):
            print("Goodbye!")
            break

        if user_input.startswith(":pair"):
            parts = user_input.split()
            if len(parts) == 3:
                s, t = parts[1].lower(), parts[2].lower()
                if (s, t) in MODEL_MAP:
                    src_lang, tgt_lang = s, t
                    print(f"[OK] Switched language pair to {src_lang.upper()} -> {tgt_lang.upper()}")
                else:
                    print(f"[!] Pair '{s}->{t}' not found. Type :pairs to see supported options.")
            else:
                print("Usage: :pair <src> <tgt>  (e.g., :pair en es)")
            continue

        if user_input.startswith(":beam"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                beam_size = max(1, min(10, int(parts[1])))
                print(f"[OK] Beam size set to {beam_size}")
            else:
                print("Usage: :beam <number> (e.g., :beam 4)")
            continue

        if user_input == ":pairs":
            print("\nSupported pairs:")
            for s, t in MODEL_MAP.keys():
                print(f"  - {s} ({LANGUAGE_NAMES.get(s, s)}) -> {t} ({LANGUAGE_NAMES.get(t, t)})")
            continue

        if user_input.startswith(":"):
            print("Unknown command. Available: :pair <src> <tgt>, :beam <n>, :pairs, :quit")
            continue

        # Translate
        try:
            translation, latency = translate_text(
                user_input,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                beam_size=beam_size,
                device=device,
            )
            print("-" * 65)
            print(f"Output: {translation}")
            print(f"Speed : {latency * 1000:.1f} ms ({latency:.3f} s)")
            print("-" * 65)
        except Exception as e:
            print(f"[!] Translation error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Simplest Single-File Terminal Machine Translation System"
    )
    parser.add_argument("text", nargs="?", default=None, help="Text to translate (optional).")
    parser.add_argument("--src", default="en", help="Source language code (default: en).")
    parser.add_argument("--tgt", default="fr", help="Target language code (default: fr).")
    parser.add_argument("--beam", type=int, default=4, help="Beam size (default: 4).")
    parser.add_argument("--ref", default=None, help="Reference translation for BLEU score evaluation.")
    parser.add_argument("--device", default=None, help="Device to use ('cpu' or 'cuda').")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Launch interactive CLI session."
    )

    args = parser.parse_args()

    # If no text provided or -i flag passed, launch interactive mode
    if args.interactive or args.text is None:
        run_interactive(src_lang=args.src, tgt_lang=args.tgt, beam_size=args.beam)
        return

    # One-shot CLI translation
    try:
        translation, latency = translate_text(
            args.text,
            src_lang=args.src,
            tgt_lang=args.tgt,
            beam_size=args.beam,
            device=args.device,
        )
        print("\n" + "=" * 50)
        print(f"Input       : {args.text}")
        print(f"Translation : {translation}")
        print(f"Latency     : {latency * 1000:.1f} ms")

        if args.ref:
            bleu = compute_quick_bleu(translation, args.ref)
            if bleu is not None:
                print(f"Reference   : {args.ref}")
                print(f"BLEU Score  : {bleu:.2f} / 100")
        print("=" * 50 + "\n")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
