# i18n_po_autofill_nllb.py
# Usage:  python i18n_po_autofill_nllb.py
import os, re, sys, subprocess
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
LOCALE_DIR   = PROJECT_ROOT / "locale"

# Your site languages (skip translating 'en')
SITE_LANGS = ["en","pt","es","it","fr","ru","uk","cn","tw","ko","ja","tr","de","ar","hi","vi","tl"]

# Map your UI codes -> NLLB target codes
NLLB_LANG = {
    "pt": "por_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "cn": "zho_Hans",   # Simplified
    "tw": "zho_Hant",   # Traditional
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

# ------- deps -------
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

# ------- protect placeholders/tags so the model doesn't break them -------
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

# ------- translate small strings with chunking safety (rarely needed for .po) -------
def translate_text(text: str, tx_pipe, tokenizer, max_gen_len=256, chunk_tok=220, batch_size=16) -> str:
    if not text or not text.strip():
        return text
    protected, tmap = protect(text)

    tokens = tokenizer.encode(protected, add_special_tokens=True)
    if len(tokens) <= chunk_tok:
        out = tx_pipe(protected, max_length=max_gen_len)[0]["translation_text"]
        return restore(out, tmap)

    # split into chunks (very rare for UI strings)
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

# ------- PO processing -------
def translate_entry(entry: polib.POEntry, tx_pipe, tokenizer, lang_code: str, nplurals: int):
    if entry.obsolete:
        return False
    changed = False

    if not entry.msgid_plural:
        # singular
        if "fuzzy" in entry.flags or not entry.msgstr:
            entry.msgstr = translate_text(entry.msgid, tx_pipe, tokenizer)
            if "fuzzy" in entry.flags:
                entry.flags = [f for f in entry.flags if f != "fuzzy"]
            changed = True
    else:
        # plural: translate both msgid (singular meaning) and msgid_plural
        singular_tx = translate_text(entry.msgid, tx_pipe, tokenizer)
        plural_tx   = translate_text(entry.msgid_plural, tx_pipe, tokenizer)

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
    # Make sure locale/<lang>/LC_MESSAGES/django.po exists (create with makemessages -l if needed)
    po_path = LOCALE_DIR / lang / "LC_MESSAGES" / "django.po"
    if not po_path.exists():
        run([sys.executable, "manage.py", "makemessages", "-l", lang,
             "--ignore=venv", "--ignore=node_modules", "--ignore=static"])
    return po_path

def main():
    # 1) Create/refresh PO catalogs
    #    Do per-language to ensure missing ones are created; then a global -a to refresh all.
    for lang in SITE_LANGS:
        if lang == "en":
            continue
        ensure_po_for(lang)
    run([sys.executable, "manage.py", "makemessages", "-a",
         "--ignore=venv", "--ignore=node_modules", "--ignore=static"])

    # 2) Load NLLB model once
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading NLLB model on {'GPU' if device >= 0 else 'CPU'} …")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    # 3) For each target language, build a pipeline with tgt_lang and translate missing/fuzzy entries
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
            # translate if empty or fuzzy (handles your “Stars” example and the fuzzy “No episodes…” case)
            needs = ("fuzzy" in e.flags) or \
                    (not e.msgid_plural and not e.msgstr) or \
                    (e.msgid_plural and any(v == "" for v in e.msgstr_plural.values()))
            if needs:
                if translate_entry(e, tx_pipe, tokenizer, lang, nplurals):
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
