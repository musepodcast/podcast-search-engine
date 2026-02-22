"""
translate.py

This script loads episode JSON files from an input transcripts directory (organized as
transcripts/channel_name/en or transcripts/channel_name/en-us/episode.json) and creates translated versions 
(e.g. English to Portuguese and English to Spanish) in new subfolders:
    transcripts/channel_name/<target_lang>/episode_<target_lang>.json

It uses the facebook/nllb-200-distilled-600M model via the transformers pipeline.
More languages can be added later by changing the target language list.
It also:
  - Skips files that have already been translated.
  - Translates the transcript, segments (both "text" and "speaker" fields),
    selected metadata fields, and chapter titles.
  - For speaker fields that match "Speaker <number>" (case-insensitive), it replaces "Speaker"
    with a language-specific word (based on a predefined mapping).
  - Reports timing information for each file.
"""

import os
import json
import argparse
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import Dataset

# -----------------------------
# Constants
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "podcast_data", "transcripts"))
DATABASE_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "podcast_data"))
DEFAULT_ORDER_FILE = os.path.join(DATABASE_ROOT_DEFAULT, "watcher_json", "channels_in_order.json")

def translate_many_texts(texts, translation_pipeline, max_gen_length=400, batch_size=16):
    # keep empty entries aligned
    clean = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    ds = Dataset.from_dict({"text": clean})

    def _batched(batch):
        outs = translation_pipeline(
            batch["text"],
            max_length=max_gen_length,
            batch_size=batch_size
        )
        return {"translation": [o["translation_text"] for o in outs]}

    out_ds = ds.map(_batched, batched=True, batch_size=batch_size)
    return out_ds["translation"]

def load_channel_order(order_file: str) -> list[str]:
    """
    order_file is a JSON list of absolute paths to channel language dirs, e.g.
      C:\\Users\\isaac\\podcast_data\\transcripts\\Archaix\\en
    OR channel roots; we will normalize to channel roots.
    Returns a list of channel ROOT directories in correct order:
      ...\\transcripts\\Archaix
    """
    try:
        with open(order_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            return []
    except Exception:
        return []

    channel_roots = []
    seen = set()

    for p in items:
        if not isinstance(p, str) or not p.strip():
            continue
        p = os.path.normpath(p)

        # If list contains ...\transcripts\<channel>\<lang>, convert to channel root
        # by stripping the last component.
        if os.path.isdir(p):
            parent = os.path.dirname(p)  # ...\transcripts\<channel>
            # detect if p looks like a language folder by checking its parent inside transcripts
            # We treat p as lang-dir if its parent exists and contains lang-like folder name.
            # Either way, parent is channel root if p ends in 'en'/'en-us' etc.
            if os.path.isdir(parent):
                # If parent is inside transcripts, keep it as channel root
                # (This will be correct for ...\<channel>\<lang>)
                channel_root = parent
            else:
                channel_root = p
        else:
            continue

        channel_root = os.path.normpath(channel_root)
        if channel_root not in seen and os.path.isdir(channel_root):
            seen.add(channel_root)
            channel_roots.append(channel_root)

    return channel_roots

# -----------------------------
# Custom Speaker Translator
# -----------------------------
def custom_translate_speaker(speaker_text, target_lang_code):
    """
    If the speaker field matches a pattern like "Speaker <number>" (case-insensitive),
    replace "Speaker" with the language-specific equivalent (as defined in SPEAKER_MAPPING).
    Otherwise, simply return the text.
    """
    match = re.match(r"(?i)^speaker\s+(\d+)", speaker_text)
    if match:
        number = match.group(1)
        replacement = SPEAKER_MAPPING.get(target_lang_code, "Speaker")
        return f"{replacement} {number}"
    else:
        return speaker_text

# -----------------------------
# Text Translation Function
# -----------------------------
def translate_text(text, translation_pipeline, tokenizer, max_gen_length=400, chunk_size=350, batch_size=8):
    """
    Translate a long text by chunking input_ids (no full-sequence encode),
    then translating chunks in batches via a HF Dataset.
    """
    if not text:
        return ""

    # ✅ Tokenize WITHOUT creating a single huge encoded sequence with special tokens
    # This avoids warnings like: (108689 > 1024)
    enc = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = enc["input_ids"]

    # If the text is short enough, translate it directly.
    if len(input_ids) <= chunk_size:
        result = translation_pipeline(text, max_length=max_gen_length)
        return result[0]["translation_text"]

    # Split the token list into chunks (by ids), then decode each chunk back to text.
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)

    # Create a Hugging Face dataset from the chunks.
    ds = Dataset.from_dict({"text": chunks})

    def translate_batch(batch):
        translations = translation_pipeline(
            batch["text"],
            max_length=max_gen_length,
            batch_size=batch_size
        )
        return {"translation": [t["translation_text"] for t in translations]}

    ds_translated = ds.map(translate_batch, batched=True, batch_size=batch_size)

    translated_chunks = ds_translated["translation"]
    return " ".join(translated_chunks)

# -----------------------------
# JSON Translation Function
# -----------------------------
def translate_json(data, translation_pipeline, tokenizer, target_lang, folder_code, max_gen_length=400):
    """
    Translate selected fields in the episode JSON data.
    """
    # For the metadata language field and title, derive a short code.
    #short_lang = target_lang.split("_")[0] if "_" in target_lang else target_lang

    short_lang = folder_code
    
    # Translate the main transcript.
    if "transcript" in data and isinstance(data["transcript"], str):
        data["transcript"] = translate_text(data["transcript"], translation_pipeline, tokenizer, max_gen_length)
    
    # Translate each segment's text and speaker.
    # Translate each segment's text in one big batch (much faster), then translate speaker labels.
    if "segments" in data and isinstance(data["segments"], list) and data["segments"]:
        seg_texts = [
            (s.get("text", "") if isinstance(s.get("text", ""), str) else "")
            for s in data["segments"]
        ]

        translated_seg_texts = translate_many_texts(
            seg_texts,
            translation_pipeline,
            max_gen_length=max_gen_length,
            batch_size=16,   # bump this up/down depending on VRAM
        )

        for s, t in zip(data["segments"], translated_seg_texts):
            s["text"] = t

        # Speaker labels are cheap; keep your custom mapping
        for segment in data["segments"]:
            if "speaker" in segment and isinstance(segment["speaker"], str):
                segment["speaker"] = custom_translate_speaker(segment["speaker"], target_lang)
    
    # Translate selected metadata fields.
    if "metadata" in data and isinstance(data["metadata"], dict):
        md = data["metadata"]
        if "episode_title" in md and isinstance(md["episode_title"], str):
            md["episode_title"] = translate_text(md["episode_title"], translation_pipeline, tokenizer, max_gen_length)
            md["episode_title"] += f" ({LANGUAGE_NAMES.get(target_lang, target_lang)})"
        if "summary" in md and isinstance(md["summary"], str):
            md["summary"] = translate_text(md["summary"], translation_pipeline, tokenizer, max_gen_length)
        if "description" in md and isinstance(md["description"], str):
            md["description"] = translate_text(md["description"], translation_pipeline, tokenizer, max_gen_length)
        if "categories" in md and isinstance(md["categories"], list):
            md["categories"] = [
                translate_text(cat, translation_pipeline, tokenizer, max_gen_length)
                if isinstance(cat, str) else cat for cat in md["categories"]
            ]
        # Update language field with the short code.
        md["language"] = short_lang
        # Set translated flag to True.
        md["translated"] = True
    # Translate each chapter's title.
    if "chapters" in data and isinstance(data["chapters"], list):
        for chapter in data["chapters"]:
            if "title" in chapter and isinstance(chapter["title"], str):
                chapter["title"] = translate_text(chapter["title"], translation_pipeline, tokenizer, max_gen_length)
    
    return data

# -----------------------------
# Main Function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Translate podcast episode JSON files from English to multiple target languages."
    )
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT,
        help=f"Path to the transcripts directory (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--feeds_file",
        default=None,
        help="Optional: watcher_json feeds list (e.g. youtube_rss.json). If set, only translate channels from this file, in file order."
    )
    parser.add_argument(
        "--database_root",
        default=None,
        help="Optional: override database root (the folder that contains watcher_json/ and transcripts/)."
    )
    parser.add_argument(
        "--channel_order_file",
        default=DEFAULT_ORDER_FILE,
        help=f"JSON list of channel dirs (in order) written by main.py (default: {DEFAULT_ORDER_FILE})"
    )
    args = parser.parse_args()

    # List of target languages to process.
    # Each tuple is (folder_code, tgt_lang_code_for_pipeline)
    # For example, Portuguese: folder "pt", model target "por_Latn"
    #              Spanish: folder "spa", model target "spa_Latn"
    target_languages = [
        ("pt", "por_Latn"), # Portuguese
        ("es", "spa_Latn"), # Spanish
        ("it", "ita_Latn"), # Italian
        ("fr", "fra_Latn"), # French
        ("ru", "rus_Cyrl"), # Russian
        ("uk", "ukr_Cyrl"), # Ukrainian
        ("cn", "zho_Hans"), # Simplified Chinese
        ("tw", "zho_Hant"), # Traditional Chinese
        ("ko", "kor_Hang"), # Korean
        ("ja", "jpn_Jpan"), # Japanese
        ("tr", "tur_Latn"), # Turkish
        ("de", "deu_Latn"), # German
        ("ar", "arb_Arab"), # Arabic
        ("hi", "hin_Deva"), # Hindi
        ("vi", "vie_Latn"), # Vietnamese
        ("tl", "tgl_Latn"), # Tagalog
        
    ]
    
    # For NLLB, source language is typically "eng_Latn"
    src_lang = "eng_Latn"
    device = 0 if torch.cuda.is_available() else -1

    # Process each target language in turn.
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    for lang_folder, tgt_lang in target_languages:
        translation_pipeline_obj = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device
        )
        print("Translation model loaded for", tgt_lang)

        # Walk through the transcripts folder.
        # Decide channel processing order:
        # 1) If channel_order_file exists and has entries -> use it (exact order from main.py)
        # 2) Else -> fallback to alphabetical os.listdir (old behavior)
        ordered_channel_paths = []
        if args.channel_order_file and os.path.exists(args.channel_order_file):
            ordered_channel_paths = load_channel_order(args.channel_order_file)

        if ordered_channel_paths:
            channel_paths = ordered_channel_paths
        else:
            channel_paths = [
                os.path.join(args.input_dir, d)
                for d in os.listdir(args.input_dir)
                if os.path.isdir(os.path.join(args.input_dir, d))
            ]

        # Walk channels in the chosen order
        for channel_path in channel_paths:
            channel_path = os.path.normpath(channel_path)

            # channel name (for prints only)
            channel = os.path.basename(channel_path)

            print(f"Processing channel: {channel}")

            # Determine the source directory for the transcripts.
            # Check for subdirectories named "en" or "en-us".
            source_dir = None
            for src_candidate in ["en", "en-us"]:
                candidate_path = os.path.join(channel_path, src_candidate)
                if os.path.isdir(candidate_path):
                    source_dir = candidate_path
                    print(f"  Found source subdirectory: {src_candidate}")
                    break
            # If neither exists, use the channel folder itself.
            if source_dir is None:
                source_dir = channel_path

            # Create the target language subfolder in the channel folder if it does not exist.
            output_dir = os.path.join(channel_path, lang_folder)
            os.makedirs(output_dir, exist_ok=True)

            # Process each JSON file in the source directory.
            for filename in os.listdir(source_dir):
                if not filename.endswith(".json"):
                    continue

                file_path = os.path.join(source_dir, filename)
                # Skip files that are already translated (filename contains _<lang_folder>)
                if f"_{lang_folder}" in filename:
                    continue

                # Also, if the output file already exists, skip processing.
                base_name = os.path.splitext(filename)[0]
                out_filename = f"{base_name}_{lang_folder}.json"
                out_file_path = os.path.join(output_dir, out_filename)
                if os.path.exists(out_file_path):
                    print(f"  Skipping file (already exists): {filename}")
                    continue

                print(f"  Translating file: {filename}")
                start_time = time.time()
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"    Error reading {file_path}: {e}")
                    continue

                # Translate JSON (pass the target language code used for speakers and metadata).
                translated_data = translate_json(data, translation_pipeline_obj, tokenizer, tgt_lang, lang_folder)
                try:
                    with open(out_file_path, "w", encoding="utf-8") as outf:
                        json.dump(translated_data, outf, ensure_ascii=False, indent=4)
                    elapsed = time.time() - start_time
                    print(f"    Saved translated file: {out_file_path} (took {elapsed:.1f} seconds)")
                except Exception as e:
                    print(f"    Error saving translated file: {e}")
        print(f"=== Completed translation for {lang_folder} ===\n")


# -----------------------------
# Mapping for 3 letter code to Language Name in Native Language"
# -----------------------------
LANGUAGE_NAMES = {
    "por_Latn": "Português",    # Portuguese
    "spa_Latn": "Español",  # Spanish
    "ita_Latn": "Italiano", # Italian
    "fra_Latn": "Français", # French
    "rus_Cyrl": "Русский",  # Russian
    "ukr_Cyrl": "українська",    # Ukrainian
    "zho_Hans": "中文 (简体)",  # Chinese (Simplified)
    "zho_Hant": "中文 (繁體)",  # Chinese (Traditional)
    "kor_Hang": "한국어",    # Korean
    "jpn_Jpan": "日本語",    # Japanese
    "tur_Latn": "Türkçe",    # Turkish
    "deu_Latn": "Deutsch",    # German
    "arb_Arab": "العربية",    # Arabic
    "hin_Deva": "हिन्दी",    # Hindi
    "vie_Latn": "Tiếng Việt",    # Vietnamese
    "tgl_Latn": "Tagalog",    # Tagalog
    

    
}            
# -----------------------------
# Mapping for translating the word "Speaker"
# -----------------------------
# The keys here are the language codes used by NLLB (e.g. "por_Latn" for Portuguese, "spa_Latn" for Spanish).
# If a language code is not present, the fallback is "Speaker".
SPEAKER_MAPPING = {
    # Arabic and related
    "acm_Arab": "المتحدث",
    "acq_Arab": "المتحدث",
    "aeb_Arab": "المتحدث",
    "arb_Arab": "المتحدث",
    "arb_Latn": "Almutahadith",  # romanized fallback
    "azb_Arab": "المتحدث",
    "pes_Arab": "المتحدث",
    "prs_Arab": "المتحدث",
    "pbt_Arab": "المتحدث",
    "snd_Arab": "المتحدث",
    "uig_Arab": "المتحدث",
    "urd_Arab": "بولنے والا",
    # Afrikaans
    "afr_Latn": "Spreker",
    # Akan
    "aka_Latn": "Kasɛmpafo",
    # Amharic
    "amh_Ethi": "ተናጋሪ",
    # Assamese
    "asm_Beng": "কথক",
    # Basque
    "eus_Latn": "Hizlari",
    # Bengali
    "ben_Beng": "বক্তা",
    # Bhojpuri
    "bho_Deva": "वक्ता",
    # Banjar
    "bjn_Latn": "Pembicara",
    "bjn_Arab": "المتحدث",
    # Bosnian, Croatian, Serbian, Slovenian, etc.
    "bos_Latn": "Govornik",
    "hrv_Latn": "Govornik",
    "srp_Cyrl": "Говорник",
    "slv_Latn": "Govornik",
    # Bulgarian
    "bul_Cyrl": "Лектор",
    # Catalan
    "cat_Latn": "Orador",
    # Cebuano
    "ceb_Latn": "Tagapagsalita",
    # Czech
    "ces_Latn": "Mluvčí",
    # Danish
    "dan_Latn": "Taler",
    # German
    "deu_Latn": "Sprecher",
    # English
    "eng_Latn": "Speaker",
    # Esperanto
    "epo_Latn": "Parolanto",
    # Estonian
    "est_Latn": "Kõneleja",
    # Finnish
    "fin_Latn": "Puhuja",
    # French
    "fra_Latn": "Intervenant",
    # Galician
    "glg_Latn": "Orador",
    # Greek
    "ell_Grek": "Ομιλητής",
    # Hebrew
    "heb_Hebr": "מדבר",
    # Hindi
    "hin_Deva": "वक्ता",
    # Indonesian
    "ind_Latn": "Pembicara",
    # Italian
    "ita_Latn": "Relatore",
    # Japanese
    "jpn_Jpan": "スピーカー",
    # Korean
    "kor_Hang": "스피커",
    # Lithuanian
    "lit_Latn": "Kalbėtojas",
    # Latvian 
    "lvs_Latn": "Runātājs",
    # Malay/Indonesian variants
    "min_Latn": "Pembicara",
    # Dutch
    "nld_Latn": "Spreker",
    # Norwegian
    "nob_Latn": "Talsperson",
    "nno_Latn": "Talsperson",
    # Polish
    "pol_Latn": "Mówca",
    # Portuguese
    "por_Latn": "Locutor",
    # Romanian
    "ron_Latn": "Vorbitor",
    # Russian
    "rus_Cyrl": "Спикер",
    # Spanish
    "spa_Latn": "Orador",
    # Swedish
    "swe_Latn": "Talare",
    # Tagalog
    "tgl_Latn": "Tagapagsalita",
    # Thai
    "tha_Thai": "ลำโพง",
    # Turkish
    "tur_Latn": "Konuşmacı",
    # Ukrainian
    "ukr_Cyrl": "Спікер",
    # Vietnamese
    "vie_Latn": "Diễn giả",
    # Chinese
    "zho_Hans": "扬声器",
    "zho_Hant": "揚聲器",
    # Fallback for any missing code
}

if __name__ == "__main__":
    main()