# i18n_po_autofill_nllb.py
# Usage:  python i18n_po_autofill_nllb.py
import os, re, sys, subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent
LOCALE_DIR   = PROJECT_ROOT / "locale"

SITE_LANGS = ["en","pt","es","it","fr","ru","uk","cn","tw","ko","ja","tr","de","ar","hi","vi","tl"]

NLLB_LANG = {
    "pt": "por_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "cn": "zho_Hans",  # Simplified
    "tw": "zho_Hant",  # Traditional
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "tr": "tur_Latn",
    "de": "deu_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "vi": "vie_Latn",
    "tl": "tgl_Latn",
}
SRC_LANG = "eng_Latn"

def ensure(pkg, mod=None):
    try:
        __import__(mod or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        __import__(mod or pkg)

ensure("polib")
ensure("torch", "torch")
ensure("transformers")
ensure("datasets")

import polib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------- regexes for structural tokens we must preserve exactly ----------
PH_NAMED   = re.compile(r"%\([^)]+\)[#0\- +]?\d*(?:\.\d+)?[sdif]")
PH_SIMPLE  = re.compile(r"%(?:\d+\$)?[#0\- +]?\d*(?:\.\d+)?[sdif]")
DJ_BRACES  = re.compile(r"{{\s*[^}]+\s*}}")
HTML_TAG   = re.compile(r"</?[^>]+?>")
PERC_ESC   = re.compile(r"%%")  # literal percent
NL_RE      = re.compile(r"(\r\n|\r|\n)")

# Unified splitter that *keeps* the tokens (capturing group)
STRUCT_RX = re.compile(
    r"("                                   # capture so split keeps tokens
    r"(?:\r\n|\r|\n)"                       # newline tokens
    r"|%%"                                  # literal percent
    r"|%\([^)]+\)[#0\- +]?\d*(?:\.\d+)?[sdif]"   # named printf
    r"|%(?:\d+\$)?[#0\- +]?\d*(?:\.\d+)?[sdif]"  # simple printf
    r"|{{\s*[^}]+\s*}}"                     # django braces
    r"|</?[^>]+?>"                          # html-like tags
    r")"
)

def is_struct_token(piece: str) -> bool:
    return bool(STRUCT_RX.fullmatch(piece))

def count_struct(src: str) -> Dict[str, int]:
    return {
        "nl": len(re.findall(r"\r\n|\r|\n", src)),
        "perc": len(PERC_ESC.findall(src)),
        "ph_named": len(PH_NAMED.findall(src)),
        "ph_simple": len(PH_SIMPLE.findall(src)),
        "dj": len(DJ_BRACES.findall(src)),
        "html": len(HTML_TAG.findall(src)),
    }

def structure_equal(a: str, b: str) -> bool:
    return count_struct(a) == count_struct(b)

def translate_preserving_structure_batched(
    text: str,
    tx_pipe,
    max_gen_len: int = 256,
    batch_size: int = 16
) -> str:
    """
    Split into structural tokens and free-text spans. Copy tokens verbatim.
    Batch-translate only free-text spans. Strip accidental newlines produced
    by the model within a free-text span to avoid changing line counts.
    """
    if not text:
        return text

    parts: List[str] = STRUCT_RX.split(text)
    # Collect indices of translatable spans
    to_tx_indices: List[int] = []
    to_tx_payload: List[str] = []
    for idx, p in enumerate(parts):
        if not p:
            continue
        if is_struct_token(p):
            continue
        # If it's purely whitespace, keep as-is (no need to translate)
        if p.strip() == "":
            continue
        to_tx_indices.append(idx)
        to_tx_payload.append(p)

    # Run batched translation for payload (if any)
    translations: List[str] = []
    if to_tx_payload:
        # HuggingFace pipeline supports list input + batch_size
        raw_out = tx_pipe(to_tx_payload, max_length=max_gen_len, batch_size=batch_size)
        # Normalize to list of strings
        translations = [o["translation_text"] for o in raw_out]

    # Rebuild the parts with translated content
    out_parts = parts[:]  # shallow copy
    t_i = 0
    for idx in to_tx_indices:
        # For safety, handle length mismatches gracefully
        if t_i >= len(translations):
            new_text = to_tx_payload[t_i] if t_i < len(to_tx_payload) else parts[idx]
        else:
            new_text = translations[t_i]
        # Do not allow accidental newlines in a span
        new_text = NL_RE.sub(" ", new_text)
        out_parts[idx] = new_text
        t_i += 1

    # Now we have a string with identical structural tokens
    return "".join(out_parts)

# ---------- PO entry handling ----------
def translate_entry(entry: polib.POEntry, tx_pipe, lang_code: str, nplurals: int) -> bool:
    """
    Translate a POEntry while preserving placeholders/newlines/tags exactly.
    If the translated string changes structure, fall back to source and mark fuzzy.
    Returns True if the entry was modified.
    """
    if entry.obsolete:
        return False

    changed = False

    def tx(s: str) -> str:
        return translate_preserving_structure_batched(s or "", tx_pipe)

    # Helper to ensure we don't accidentally clear 'fuzzy' when we need it
    def add_fuzzy(e: polib.POEntry):
        if "fuzzy" not in e.flags:
            e.flags.append("fuzzy")

    def remove_fuzzy(e: polib.POEntry):
        e.flags = [f for f in e.flags if f != "fuzzy"]

    if not entry.msgid_plural:
        # Singular
        if ("fuzzy" in entry.flags) or (not entry.msgstr):
            candidate = tx(entry.msgid)
            if not structure_equal(candidate, entry.msgid):
                # keep it safe: use source and keep fuzzy so humans can inspect later
                entry.msgstr = entry.msgid
                add_fuzzy(entry)
            else:
                entry.msgstr = candidate
                remove_fuzzy(entry)
            changed = True
    else:
        # Plural
        singular_tx = tx(entry.msgid)
        plural_tx   = tx(entry.msgid_plural)

        singular_ok = structure_equal(singular_tx, entry.msgid)
        plural_ok   = structure_equal(plural_tx, entry.msgid_plural)

        # ensure plural slots exist
        if not entry.msgstr_plural:
            for i in range(nplurals):
                entry.msgstr_plural[i] = ""

        updated_any = False
        for i in range(nplurals):
            # Conventional: index 0 is "one" form (maps to msgid), others to msgid_plural
            desired = singular_tx if i == 0 else plural_tx
            desired_ok = singular_ok if i == 0 else plural_ok

            if ("fuzzy" in entry.flags) or (entry.msgstr_plural.get(i, "") == ""):
                if desired_ok:
                    entry.msgstr_plural[i] = desired
                    updated_any = True
                else:
                    # fallback: copy original English to keep structure valid
                    entry.msgstr_plural[i] = entry.msgid if i == 0 else entry.msgid_plural
                    add_fuzzy(entry)
                    updated_any = True

        if updated_any:
            # Only clear fuzzy if both forms structurally matched
            if singular_ok and plural_ok:
                remove_fuzzy(entry)
            changed = True

    return changed

# ---------- helpers ----------
def run(cmd):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

def ensure_po_for(lang: str):
    po_path = LOCALE_DIR / lang / "LC_MESSAGES" / "django.po"
    if not po_path.exists():
        run([sys.executable, "manage.py", "makemessages", "-l", lang,
             "--ignore=venv", "--ignore=node_modules", "--ignore=static"])
    return po_path

# ---------- main ----------
def main():
    # 1) Ensure catalogs exist, then refresh all
    for lang in SITE_LANGS:
        if lang == "en":
            continue
        ensure_po_for(lang)
    run([sys.executable, "manage.py", "makemessages", "-a",
         "--ignore=venv", "--ignore=node_modules", "--ignore=static"])

    # 2) Load NLLB once
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading NLLB model on {'GPU' if device >= 0 else 'CPU'} …")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    # 3) Translate per target language
    for lang in SITE_LANGS:
        if lang == "en":
            print("[en] Skip translating source language.")
            continue
        tgt = NLLB_LANG.get(lang)
        if not tgt:
            print(f"[{lang}] No NLLB mapping configured; skipping.")
            continue

        print(f"[{lang}] Translating PO with tgt_lang={tgt} …")
        tx_pipe = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=SRC_LANG,
            tgt_lang=tgt,
            device=device
        )

        po_path = LOCALE_DIR / lang / "LC_MESSAGES" / "django.po"
        if not po_path.exists():
            print(f"[{lang}] Missing {po_path}; skipping.")
            continue

        po = polib.pofile(str(po_path))

        # read plural forms (e.g., 'nplurals=2; plural=(n != 1);')
        nplurals = 2
        pf = po.metadata.get("Plural-Forms", "")
        m = re.search(r"nplurals\s*=\s*(\d+)", pf or "")
        if m:
            try:
                nplurals = int(m.group(1))
            except Exception:
                nplurals = 2

        changed = 0
        for e in po:
            needs = ("fuzzy" in e.flags) or \
                    (not e.msgid_plural and not e.msgstr) or \
                    (e.msgid_plural and any(v == "" for v in e.msgstr_plural.values()))
            if needs and translate_entry(e, tx_pipe, lang, nplurals):
                changed += 1

        if changed:
            po.save(str(po_path))
            print(f"[{lang}] Updated {changed} entries.")
        else:
            print(f"[{lang}] No changes needed.")

    # 4) Compile .mo files
    print("Compiling messages …")
    run([sys.executable, "manage.py", "compilemessages"])
    print("✅ Done.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Command failed ({e.returncode}): {e}")
        sys.exit(e.returncode)
