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
    Translate a long text by:
      1. Encoding the text into tokens,
      2. Splitting the tokens into fixed-size chunks (of size `chunk_size`),
      3. Creating a Hugging Face Dataset from the chunks,
      4. Running the translation pipeline on the dataset in batches,
      5. And joining the results into a final translation.

    Args:
        text (str): The text to translate.
        translation_pipeline: The Hugging Face translation pipeline.
        tokenizer: The tokenizer used by the model.
        max_gen_length (int): Maximum generation length for translation.
        chunk_size (int): Maximum number of tokens per chunk (should be <= model max, e.g. 350).
        batch_size (int): Number of chunks to process in one batch.

    Returns:
        str: The translated text.
    """
    # Tokenize the entire text.
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    # If the text is short enough, translate it directly.
    if len(tokens) <= chunk_size:
        result = translation_pipeline(text, max_length=max_gen_length)
        return result[0]['translation_text']
    
    # Split the token list into chunks.
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    
    # Create a Hugging Face dataset from the chunks.
    ds = Dataset.from_dict({"text": chunks})
    
    # Define a batched translation function.
    def translate_batch(batch):
        # Pass the batch list to the pipeline with explicit batch_size.
        translations = translation_pipeline(
            batch["text"],
            max_length=max_gen_length,
            batch_size=batch_size  # explicitly set the batch size here
        )
        # Extract the translation text for each item.
        return {"translation": [t["translation_text"] for t in translations]}
    
    # Map over the dataset in batches.
    ds_translated = ds.map(translate_batch, batched=True, batch_size=batch_size)
    
    # Extract translated texts and join them.
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
    if "segments" in data and isinstance(data["segments"], list):
        for segment in data["segments"]:
            if "text" in segment and isinstance(segment["text"], str):
                segment["text"] = translate_text(segment["text"], translation_pipeline, tokenizer, max_gen_length)
            if "speaker" in segment and isinstance(segment["speaker"], str):
                # Use our custom function to translate the speaker label.
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
    parser.add_argument("--input_dir", default="transcripts", help="Path to the transcripts directory")
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
    for lang_folder, tgt_lang in target_languages:
        print(f"\n=== Translating to {lang_folder} ({tgt_lang}) ===")
        print("Loading translation model on device:", "GPU" if device >= 0 else "CPU")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
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
        for channel in os.listdir(args.input_dir):
            channel_path = os.path.join(args.input_dir, channel)
            if not os.path.isdir(channel_path):
                continue

            # Skip directories that are already target language folders
            if channel.lower() == lang_folder.lower():
                continue

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
