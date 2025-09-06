# i18n_po_autofill_nllb.py
# Usage:  python i18n_po_autofill_nllb.py
import os, re, sys, subprocess
from pathlib import Path
from typing import Dict, Tuple

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
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------- placeholder/tag protection ----------
PH_NAMED   = re.compile(r"%\([^)]+\)[#0\- +]?\d*(?:\.\d+)?[sdif]")
PH_SIMPLE  = re.compile(r"%(?:\d+\$)?[#0\- +]?\d*(?:\.\d+)?[sdif]")
DJ_BRACES  = re.compile(r"{{\s*[^}]+\s*}}")
HTML_TAG   = re.compile(r"</?[^>]+?>")
PERC_ESC   = re.compile(r"%%")  # literal percent

def protect(text: str) -> Tuple[str, Dict[str,str]]:
    token_map: Dict[str,str] = {}
    idx = 0
    def sub_all(rx, prefix, s):
        nonlocal idx
        def repl(m):
            nonlocal idx
            token = f"__{prefix}{idx}__"
            token_map[token] = m.group(0)
            idx += 1
            return token
        return rx.sub(repl, s)
    s = text
    s = sub_all(PERC_ESC, "PERC", s)
    s = sub_all(PH_NAMED, "PHN",  s)
    s = sub_all(PH_SIMPLE,"PHS",  s)
    s = sub_all(DJ_BRACES,"DJB",  s)
    s = sub_all(HTML_TAG, "TAG",  s)
    return s, token_map

def restore(text: str, token_map: Dict[str,str]) -> str:
    for k,v in token_map.items():
        text = text.replace(k, v)
    return text

# ---------- NEW: translate preserving newlines ----------
NL_RE = re.compile(r"(\r\n|\r|\n)")

def _translate_one_chunk(chunk: str, tx_pipe, tokenizer, max_gen_len=256):
    # chunk has NO newline characters
    if not chunk or not chunk.strip():
        return chunk
    protected, tmap = protect(chunk)
    out = tx_pipe(protected, max_length=max_gen_len)[0]["translation_text"]
    out = restore(out, tmap)
    # forbid accidental newlines inside a line (don’t add/remove lines)
    out = NL_RE.sub(" ", out)
    return out

def translate_text_preserving_newlines(text: str, tx_pipe, tokenizer) -> str:
    """
    Translate while keeping the exact count and placement of newline tokens.
    """
    if not text:
        return text
    parts = NL_RE.split(text)  # keeps separators
    out_parts = []
    for p in parts:
        if not p:
            continue
        if NL_RE.fullmatch(p):
            # Keep newline token exactly as in source
            out_parts.append(p)
        else:
            out_parts.append(_translate_one_chunk(p, tx_pipe, tokenizer))
    return "".join(out_parts)

# (kept for rare very long one-liners; not used on multiline strings anymore)
def translate_text(text: str, tx_pipe, tokenizer, max_gen_len=256, chunk_tok=220, batch_size=16) -> str:
    if not text or not text.strip():
        return text
    protected, tmap = protect(text)

    tokens = tokenizer.encode(protected, add_special_tokens=True)
    if len(tokens) <= chunk_tok:
        out = tx_pipe(protected, max_length=max_gen_len)[0]["translation_text"]
        return restore(out, tmap)

    chunks = []
    for i in range(0, len(tokens), chunk_tok):
        chunk = tokenizer.decode(tokens[i:i+chunk_tok], skip_special_tokens=True)
        chunks.append(chunk)
    ds = Dataset.from_dict({"text": chunks})

    def translate_batch(batch):
        res = tx_pipe(batch["text"], max_length=max_gen_len, batch_size=batch_size)
        return {"translation": [r["translation_text"] for r in res]}

    ds_t = ds.map(translate_batch, batched=True, batch_size=batch_size)
    joined = " ".join(ds_t["translation"])
    return restore(joined, tmap)

# ---------- PO entry handling ----------
def translate_entry(entry: polib.POEntry, tx_pipe, tokenizer, lang_code: str, nplurals: int):
    if entry.obsolete:
        return False
    changed = False

    # Decide translator based on presence of newlines
    def tx(s: str) -> str:
        if NL_RE.search(s or ""):
            return translate_text_preserving_newlines(s, tx_pipe, tokenizer)
        return translate_text(s, tx_pipe, tokenizer)

    if not entry.msgid_plural:
        # singular
        if "fuzzy" in entry.flags or not entry.msgstr:
            entry.msgstr = tx(entry.msgid)
            if "fuzzy" in entry.flags:
                entry.flags = [f for f in entry.flags if f != "fuzzy"]
            changed = True
    else:
        # plural: translate both forms; keep same newline pattern as their sources
        singular_tx = tx(entry.msgid)
        plural_tx   = tx(entry.msgid_plural)

        # ensure the expected plural slots exist
        if not entry.msgstr_plural:
            for i in range(nplurals):
                entry.msgstr_plural[i] = ""

        updated_any = False
        for i in range(nplurals):
            desired = singular_tx if i == 0 else plural_tx
            if ("fuzzy" in entry.flags) or (entry.msgstr_plural.get(i, "") == ""):
                entry.msgstr_plural[i] = desired
                updated_any = True
        if updated_any:
            if "fuzzy" in entry.flags:
                entry.flags = [f for f in entry.flags if f != "fuzzy"]
            changed = True

    return changed

def run(cmd):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

def ensure_po_for(lang: str):
    po_path = LOCALE_DIR / lang / "LC_MESSAGES" / "django.po"
    if not po_path.exists():
        run([sys.executable, "manage.py", "makemessages", "-l", lang,
             "--ignore=venv", "--ignore=node_modules", "--ignore=static"])
    return po_path

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
        tx_pipe = pipeline("translation", model=model, tokenizer=tokenizer,
                           src_lang=SRC_LANG, tgt_lang=tgt, device=device)

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
            if needs and translate_entry(e, tx_pipe, tokenizer, lang, nplurals):
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
