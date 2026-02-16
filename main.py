# main.py

from dateutil import parser
import time
from datetime import datetime, timezone, timedelta
import subprocess
import os
import sys
import yaml  # For handling YAML files
from transformers import pipeline as transformers_pipeline  # Renamed for clarity
from downloader import parse_feed, download_audio, yt_dlp_info
from transcriber import transcribe_and_diarize, initialize_diarization_pipeline, validate_audio
from utils import sanitize_filename, make_episode_stem_with_suffix, stable_id_suffix
from pydub import AudioSegment
from pydub.utils import make_chunks
import logging
import json
import re
import math
import warnings
import spacy
import nltk  # For NLTK data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch  # For tensor operations
import podcastparser  # Imported for parsing feeds
import requests  # Imported for HTTP requests
import io  # Imported for handling IO operations
from bs4 import BeautifulSoup  # Imported for clean_html function
import torch.nn.functional as F
from pathlib import Path
import argparse
import glob
import gc
import psutil
from urllib.parse import urlparse, parse_qs
import shutil
from logging.handlers import RotatingFileHandler



os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

FFMPEG_BIN = r"C:\ffmpeg\ffmpeg.exe"

BASE = Path(__file__).parent            # …\podcast_news
# where all your artifacts now live
DATABASE_ROOT = BASE.parent / "podcast_data"

# Suppress specific Pyannote Audio warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote.audio')

# If ffmpeg is not in PATH, set the path explicitly
# Uncomment and set the correct paths if necessary
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# Configure logging (centralized configuration) — ROTATING FILES
log_dir = DATABASE_ROOT / "log"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "transcription.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        RotatingFileHandler(
            filename=str(log_file),
            maxBytes=20 * 1024 * 1024,  # 10 MB per file
            backupCount=20,             # keep last 10 files
            encoding="utf-8"
        ),
    ]
)


# Automatic download of NLTK stopwords and punkt if not present
try:
    nltk.data.find('corpora/stopwords')
    logging.info("NLTK stopwords corpus already exists.")
except LookupError:
    logging.info("NLTK stopwords corpus not found. Downloading...")
    nltk.download('stopwords')
    logging.info("NLTK stopwords corpus downloaded successfully.")

try:
    nltk.data.find('tokenizers/punkt')
    logging.info("NLTK punkt tokenizer already exists.")
except LookupError:
    logging.info("NLTK punkt tokenizer not found. Downloading...")
    nltk.download('punkt')
    logging.info("NLTK punkt tokenizer downloaded successfully.")

# Initialize NLP models once
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load spaCy model: {e}")
    nlp = None

# Optionally, load the SentenceTransformer with FP16 precision.
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    sentence_model.eval()
    # Cap sentence transformer input to avoid long-seq weirdness
    try:
        sentence_model.max_seq_length = 256
    except Exception:
        pass


    #sentence_model.half()  # Convert the model's parameters to FP16
    logging.info("Sentence-BERT model loaded successfully on CUDA.")
except Exception as e:
    logging.critical(f"Failed to load Sentence-BERT model: {e}")
    sentence_model = None

def clamp_to_full_sentence(text: str) -> str:
    """
    Return a single full sentence ending in . ! or ? (optionally followed by a closing quote/bracket).
    If the input doesn't contain a full sentence, return "".
    """
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return t
    # If it already ends with terminal punctuation, keep it.
    if re.search(r"[.!?][\"’”)]?$", t):
        return t
    m = re.search(r"(.+[.!?])[\"’”)]?\s*$", t)
    return m.group(1).strip() if m else ""

def best_sentence_by_noun_density(text: str) -> str:
    """
    Extractive fallback: pick the sentence with the highest NOUN/PROPN density.
    """
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', text or "")
    sents = [s.strip() for s in sents if len(s.strip().split()) >= 5]
    if not sents:
        return (text or "").strip()
    if nlp:
        best = None
        best_score = -1.0
        for s in sents:
            doc = nlp(s)
            content = sum(1 for t in doc if t.pos_ in ("NOUN","PROPN"))
            score = (content + 1e-6) / (len(doc) + 1e-6)
            if score > best_score:
                best, best_score = s, score
        return best or sents[0]
    # fallback: longest reasonable sentence
    return max(sents, key=lambda s: len(s))

def resolve_ffmpeg_bin():
    # Prefer explicit env var if you set it
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    # Then the constant you defined in code
    try:
        from __main__ import FFMPEG_BIN as _FFMPEG_BIN
        if _FFMPEG_BIN and os.path.isfile(_FFMPEG_BIN):
            return _FFMPEG_BIN
    except Exception:
        pass

    # Finally, try PATH
    which = shutil.which("ffmpeg")
    if which and os.path.isfile(which):
        return which

    raise FileNotFoundError(
        "ffmpeg not found. Set FFMPEG_BIN to a valid full path or add ffmpeg to PATH."
    )

def convert_to_5min_wav_chunks(input_path, output_dir, chunk_length_ms=5*60*1000):
    input_path = str(Path(input_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    base = Path(input_path).stem
    pattern = str(out_dir / f"{base}_chunk%03d.wav")
    seg_seconds = max(1, int(round(chunk_length_ms / 1000.0)))

    ffmpeg_bin = resolve_ffmpeg_bin()

    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-y", "-nostdin",
        "-i", input_path,
        "-map", "0:a:0",
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "-f", "segment",
        "-segment_time", str(seg_seconds),
        pattern
    ]

    kwargs = {"check": True, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    try:
        subprocess.run(cmd, **kwargs)
    except FileNotFoundError:
        # This means ffmpeg_bin itself was wrong (CreateProcess failed)
        raise FileNotFoundError(f"ffmpeg executable not found at: {ffmpeg_bin}")
    return sorted(glob.glob(str(out_dir / f"{base}_chunk*.wav")))



# A helper function to split long texts and then average the embeddings.
def get_embedding(model, text, max_tokens=256):
    # Ensure the text is non-empty and strip whitespace.
    if not text or not text.strip():
        logging.warning("Empty or whitespace-only text provided to get_embedding.")
        return None
    text = text.strip()

    try:
        tokenizer = getattr(model, "tokenizer", None)
    except Exception as e:
        logging.error("Failed to retrieve tokenizer from model", exc_info=True)
        tokenizer = None

    # If a tokenizer is available, use it to check token count and split if necessary.
    if tokenizer:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True)
        except Exception as e:
            logging.error("Tokenization failed", exc_info=True)
            return None

        if len(tokens) > max_tokens:
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence.strip()
                try:
                    candidate_tokens = tokenizer.encode(candidate, add_special_tokens=True)
                except Exception as e:
                    logging.error("Tokenization failed for candidate text", exc_info=True)
                    continue

                if len(candidate_tokens) > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence.strip()
                else:
                    current_chunk = candidate
            if current_chunk:
                chunks.append(current_chunk)

            embeddings = []
            for chunk in chunks:
                try:
                    start_time = time.perf_counter()
                    embedding = model.encode(chunk, convert_to_tensor=True, show_progress_bar=False)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start_time
                    logging.info(f"Computed embedding for chunk (len {len(chunk)} characters) in {elapsed:.3f} seconds")
                    embeddings.append(embedding)
                except Exception as e:
                    logging.error(f"Failed to compute embedding for chunk: {chunk}", exc_info=True)
            if embeddings:
                return torch.stack(embeddings, dim=0).mean(dim=0)
            else:
                logging.error("No valid embeddings computed from chunks.")
                return None
        else:
            try:
                start_time = time.perf_counter()
                embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                logging.info(f"Computed embedding for text in {elapsed:.3f} seconds")
                return embedding
            except Exception as e:
                logging.error("Failed to compute embedding for text", exc_info=True)
                return None
    else:
        # If no tokenizer is available, proceed directly.
        try:
            start_time = time.perf_counter()
            embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            logging.info(f"Computed embedding for text (no tokenizer) in {elapsed:.3f} seconds")
            return embedding
        except Exception as e:
            logging.error("Failed to compute embedding for text (no tokenizer)", exc_info=True)
            return None


# Function to load configuration
def load_config(config_path='config.yaml'):
    """
    Load YAML configuration file (UTF-8). Falls back to UTF-8-SIG if needed.
    """
    try:
        # First attempt: strict UTF-8
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {config_path} (utf-8)")
    except UnicodeDecodeError:
        # Some editors save with BOM → try utf-8-sig
        with open(config_path, 'r', encoding='utf-8-sig') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {config_path} (utf-8-sig)")

    # Validate required sections and parameters
    required_sections = ['summarizer', 'chapter_generation']
    for section in required_sections:
        if section not in config:
            logging.critical(f"Missing section '{section}' in configuration file. Exiting.")
            raise SystemExit(1)
    logging.info("All required configuration sections are present.")
    return config


# Load the configuration
config = load_config()



# 2) initialize summarizer pipeline
try:
    summarizer = transformers_pipeline(
        "summarization",
        model=config['summarizer']['model'],
        tokenizer=config['summarizer']['model'],
        framework="pt"
    )
    logging.info("Summarization pipeline initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize summarization pipeline: {e}", exc_info=True)
    summarizer = None



# ─── NOW throttle your GPU and do the FP16 conversion ────────────────────
if torch.cuda.is_available():
    frac = config.get('gpu', {}).get('memory_fraction', 1.0)
    torch.cuda.set_per_process_memory_fraction(frac, device=0)
    logging.info(f"🔧 GPU memory fraction set to {int(frac*100)}%")

    if config.get('gpu', {}).get('precision') == 'fp16':
        logging.info("🔧 Converting summarizer & sentence_model to FP16")
        if summarizer is not None and hasattr(summarizer.model, 'half'):
            summarizer.model.half()
        if sentence_model is not None:
            sentence_model.half()
# ────────────────────────────────────────────────────────────────────────────

# --------------------------- Integrated Metadata Extraction Functions ---------------------------

_YT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# In-memory cache: { "<channel_transcript_dir>": {"version":1, "suffixes":{suffix: filename}} }
_INDEX_CACHE: dict[str, dict] = {}

# Matches files already in new scheme: ...__<suffix>.json
_SUFFIX_RE = re.compile(r"__(?P<suffix>[A-Za-z0-9_\-]{4,})\.json$", re.IGNORECASE)

def migrate_legacy_transcripts(
    root: Path,
    max_len: int = 100,
    report_dir: Path | None = None,
    report_name: str = "legacy_migration_report.json",
    dry_run: bool = False,
    size_limit_mb: float | None = None,   # optional safety guard; set None to scan all
) -> dict:
    """
    Bulk migration:
      - Build processed_index.json everywhere
      - Rename legacy files (no __suffix) into new scheme where suffix can be derived
      - Leave unmapped legacy files alone and report them

    Returns a summary dict with:
      total_scanned, already_good, renamed, indexed_only, unmatched, errors
    """

    root = Path(root)
    if report_dir is None:
        # keep reports alongside other state; matches your folder convention
        report_dir = (root.parent / "watcher_json") if root.parent else Path.cwd()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / report_name

    # Local suffix matcher (don’t depend on any duplicate global defs)
    _SUFFIX_RE_LOCAL = re.compile(r"__(?P<suffix>[A-Za-z0-9_\-]{4,})\.json$", re.IGNORECASE)

    def _load_index(dirpath: Path) -> dict:
        idx_file = dirpath / "processed_index.json"
        if idx_file.exists():
            try:
                data = json.loads(idx_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 1 and isinstance(data.get("suffixes"), dict):
                    return data
            except Exception:
                pass
        return {"version": 1, "suffixes": {}}

    def _save_index(dirpath: Path, idx: dict) -> None:
        idx_file = dirpath / "processed_index.json"
        if dry_run:
            return
        idx_file.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")

    def _vid_from_any(guid: str | None, link: str | None, audio_url: str | None) -> str | None:
        # guid canonical
        if guid and isinstance(guid, str) and guid.startswith("yt:video:"):
            v = guid.split("yt:video:", 1)[1].strip()
            return v or None

        # url parse via your helper if available
        for u in (link, audio_url):
            try:
                v = youtube_video_id(u)  # uses your existing helper
                if v:
                    return v
            except Exception:
                pass
        return None

    def _derive_suffix_from_json(fp: Path) -> tuple[str | None, dict | None]:
        """
        Returns (suffix, metadata_dict) or (None, metadata_dict/None).
        Reads JSON; expects combined_data style with 'metadata'.
        """
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            md = (data or {}).get("metadata") or None
            if not isinstance(md, dict):
                return None, None

            guid = md.get("guid")
            link = md.get("link")
            audio_url = md.get("audio_url")

            vid = _vid_from_any(guid, link, audio_url)

            # Use your stable suffix function
            suffix = stable_id_suffix(vid=vid, guid=guid, link=(link or audio_url))
            return suffix, md
        except Exception:
            return None, None

    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "dry_run": dry_run,
        "total_scanned": 0,
        "already_good": 0,     # already *has* __suffix
        "renamed": 0,          # legacy -> renamed into __suffix
        "indexed_only": 0,     # legacy not renamed (name collision), but indexed
        "unmatched": 0,        # legacy suffix not derivable
        "errors": 0,
        "folders_touched": 0,
        "report": str(report_path),
    }

    unmatched_records = []  # detailed report
    touched_dirs = set()

    if not root.exists():
        raise FileNotFoundError(f"transcripts root does not exist: {root}")

    # Find language directories: .../<channel>/<lang>/
    # We’ll treat any directory that contains JSONs (excluding processed_index.json) as a target.
    candidate_dirs = set()
    for fp in root.rglob("*.json"):
        if fp.name == "processed_index.json":
            continue
        candidate_dirs.add(fp.parent)

    for d in sorted(candidate_dirs):
        try:
            idx = _load_index(d)
            suffixes = idx["suffixes"]

            # 1) Index all already-suffixed files (fast)
            json_files = sorted(d.glob("*.json"))
            for fp in json_files:
                if fp.name == "processed_index.json":
                    continue
                m = _SUFFIX_RE_LOCAL.search(fp.name)
                if m:
                    suffix = m.group("suffix")
                    if suffixes.get(suffix) != fp.name:
                        suffixes[suffix] = fp.name
                        touched_dirs.add(d)
                    summary["already_good"] += 1

            # 2) Process legacy files (no suffix)
            for fp in json_files:
                if fp.name == "processed_index.json":
                    continue

                summary["total_scanned"] += 1

                if _SUFFIX_RE_LOCAL.search(fp.name):
                    # already handled
                    continue

                # optional size guard
                if size_limit_mb is not None:
                    try:
                        mb = fp.stat().st_size / (1024 * 1024)
                        if mb > float(size_limit_mb):
                            unmatched_records.append({
                                "path": str(fp),
                                "reason": f"skipped_size>{size_limit_mb}MB",
                            })
                            summary["unmatched"] += 1
                            continue
                    except Exception:
                        pass

                suffix, md = _derive_suffix_from_json(fp)
                if not suffix or not md:
                    unmatched_records.append({
                        "path": str(fp),
                        "reason": "no_metadata_or_no_stable_suffix",
                    })
                    summary["unmatched"] += 1
                    continue

                # Determine desired new name based on metadata episode_title
                ep_title = (md.get("episode_title") or fp.stem or "Unknown Title")
                new_stem = make_episode_stem_with_suffix(ep_title, suffix, max_len=max_len)
                new_name = f"{new_stem}.json"
                new_path = d / new_name

                # If index already has this suffix mapped, prefer the existing mapping
                mapped = suffixes.get(suffix)
                if mapped:
                    mapped_path = d / mapped
                    if mapped_path.exists():
                        # If mapped exists, do nothing except ensure it’s consistent
                        # (This avoids duplicates from weird legacy situations)
                        continue

                # Rename if safe
                if fp.name != new_name:
                    if new_path.exists():
                        # Collision: don't overwrite; just index the existing file and keep legacy in place
                        suffixes[suffix] = new_path.name
                        touched_dirs.add(d)
                        summary["indexed_only"] += 1

                        unmatched_records.append({
                            "path": str(fp),
                            "reason": "name_collision_target_exists_index_points_to_existing",
                            "suffix": suffix,
                            "target": str(new_path),
                        })
                        continue

                    # Actually rename
                    try:
                        if not dry_run:
                            os.replace(str(fp), str(new_path))
                        suffixes[suffix] = new_path.name
                        touched_dirs.add(d)
                        summary["renamed"] += 1
                    except Exception as e:
                        summary["errors"] += 1
                        unmatched_records.append({
                            "path": str(fp),
                            "reason": f"rename_failed: {e}",
                            "suffix": suffix,
                            "target": str(new_path),
                        })
                        continue
                else:
                    # Same name already; just ensure indexed
                    suffixes[suffix] = fp.name
                    touched_dirs.add(d)

            # Persist index if anything changed for this folder
            if d in touched_dirs:
                _save_index(d, idx)

        except Exception as e:
            summary["errors"] += 1
            unmatched_records.append({
                "folder": str(d),
                "reason": f"folder_processing_error: {e}",
            })

    summary["folders_touched"] = len(touched_dirs)

    # Write report
    report_obj = {
        "summary": summary,
        "unmatched": unmatched_records,
    }
    if not dry_run:
        report_path.write_text(json.dumps(report_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary

def _vid_from_any(guid: str | None, link: str | None, audio_url: str | None) -> str | None:
    """
    Try to derive a YouTube video id from guid/link/audio_url.
    Works with:
      - guid: yt:video:<id>
      - link/audio_url: https://www.youtube.com/watch?v=<id> or https://youtu.be/<id>
    """
    # 1) guid canonical
    if guid and isinstance(guid, str) and guid.startswith("yt:video:"):
        v = guid.split("yt:video:", 1)[1].strip()
        return v or None

    # 2) urls
    for u in (link, audio_url):
        try:
            v = youtube_video_id(u)  # you already have this helper
            if v:
                return v
        except Exception:
            pass

    return None

def _derive_suffix_from_legacy_json(fp: Path) -> str | None:
    """
    Read a legacy title-only JSON and derive the stable suffix from its metadata.
    Returns suffix or None if cannot derive.
    """
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        md = (data or {}).get("metadata") or {}

        guid = md.get("guid")
        link = md.get("link")
        audio_url = md.get("audio_url")

        vid = _vid_from_any(guid, link, audio_url)

        # Use the same stable function you already use in process_entry
        suffix = stable_id_suffix(vid=vid, guid=guid, link=link or audio_url)
        return suffix
    except Exception:
        return None

def bootstrap_index_from_folder(channel_transcript_dir: str) -> dict:
    """
    One-time scan of channel_transcript_dir to populate index from existing files.

    Adds:
      - New scheme: any '*__<suffix>.json' -> suffix map
      - Legacy: any '*.json' without suffix -> parse metadata -> derive suffix -> suffix map

    Legacy scan runs ONCE per channel (legacy_scanned flag) to avoid repeated heavy loads.
    """
    idx = load_index(channel_transcript_dir)
    suffixes = idx["suffixes"]
    base = Path(channel_transcript_dir)

    if not base.exists():
        return idx

    # Ensure structure
    if idx.get("version") != 1 or not isinstance(suffixes, dict):
        idx.clear()
        idx.update({"version": 1, "suffixes": {}, "legacy_scanned": False})
        suffixes = idx["suffixes"]

    # 1) Always: index new-scheme files quickly
    added_new = 0
    for fp in base.glob("*.json"):
        m = _SUFFIX_RE.search(fp.name)
        if not m:
            continue
        suffix = m.group("suffix")
        if suffix not in suffixes:
            suffixes[suffix] = fp.name
            added_new += 1

    if added_new:
        logging.info(f"Bootstrapped NEW-scheme index for {base} (+{added_new})")

    # 2) Legacy scan (heavy): only do once
    if not idx.get("legacy_scanned", False):
        added_legacy = 0
        scanned = 0
        skipped_big = 0

        # Tweakable guard: skip huge legacy files to avoid brutal load times
        # (If you want, set this to None to scan everything)
        MAX_LEGACY_MB = 250

        for fp in base.glob("*.json"):
            name = fp.name

            # skip the index file itself if present
            if name == "processed_index.json":
                continue

            # skip already-suffixed files (we already indexed those)
            if _SUFFIX_RE.search(name):
                continue

            # size guard
            if MAX_LEGACY_MB is not None:
                try:
                    size_mb = fp.stat().st_size / (1024 * 1024)
                    if size_mb > MAX_LEGACY_MB:
                        skipped_big += 1
                        continue
                except Exception:
                    pass

            scanned += 1
            suffix = _derive_suffix_from_legacy_json(fp)
            if not suffix:
                continue

            if suffix not in suffixes:
                # Map suffix -> legacy filename (do NOT rename here)
                suffixes[suffix] = fp.name
                added_legacy += 1

        idx["legacy_scanned"] = True
        logging.info(
            f"Bootstrapped LEGACY index for {base} "
            f"(scanned={scanned}, added={added_legacy}, skipped_big={skipped_big})"
        )

    # Persist if anything changed / legacy scan ran
    save_index(channel_transcript_dir)
    return idx

def _index_path(channel_transcript_dir: str) -> Path:
    return Path(channel_transcript_dir) / "processed_index.json"

def load_index(channel_transcript_dir: str) -> dict:
    """
    Load per-channel processed index (cached in memory).
    """
    key = str(Path(channel_transcript_dir).resolve())
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    idx_file = _index_path(channel_transcript_dir)
    if idx_file.exists():
        try:
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("index is not a dict")
        except Exception as e:
            logging.warning(f"Index read failed, will rebuild: {idx_file} ({e})")
            data = {}
    else:
        data = {}

    # normalize structure
    if data.get("version") != 1 or "suffixes" not in data or not isinstance(data.get("suffixes"), dict):
        data = {"version": 1, "suffixes": {}}

    _INDEX_CACHE[key] = data
    return data

def save_index(channel_transcript_dir: str) -> None:
    """
    Persist cached index to disk.
    """
    key = str(Path(channel_transcript_dir).resolve())
    data = _INDEX_CACHE.get(key)
    if not data:
        return
    idx_file = _index_path(channel_transcript_dir)
    try:
        idx_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logging.warning(f"Failed to write index: {idx_file} ({e})", exc_info=True)



def find_existing_episode_json(channel_root: str, suffix: str, guid: str | None, vid: str | None, link: str | None) -> str | None:
    """
    Find an existing JSON for this episode.

    Supports:
      - New scheme: *__<suffix>.json
      - Legacy: title-only .json where metadata.guid/link/audio_url matches this episode
    Searches under channel_root recursively.
    """
    root = Path(channel_root)
    if not root.exists():
        return None

    # 1) New scheme by suffix
    hit = next(root.rglob(f"*__{suffix}.json"), None)
    if hit:
        return str(hit)

    # 2) Legacy scheme: scan json metadata (only if we have an identifier)
    targets = set()
    for t in (guid, link):
        if t:
            targets.add(str(t).strip())
    if vid:
        targets.add(f"yt:video:{vid}")
        targets.add(f"https://www.youtube.com/watch?v={vid}")
        targets.add(f"https://youtu.be/{vid}")

    if not targets:
        return None

    # Scan json files (lightweight read: only metadata)
    for fp in root.rglob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            md = (data or {}).get("metadata") or {}
            mguid = (md.get("guid") or "").strip()
            mlink = (md.get("link") or "").strip()
            maudio = (md.get("audio_url") or "").strip()

            if mguid in targets or mlink in targets or maudio in targets:
                return str(fp)
        except Exception:
            # ignore bad/partial json
            continue

    return None


def episode_already_processed_by_suffix(channel_transcript_dir: str, suffix: str) -> bool:
    p = Path(channel_transcript_dir)
    if not p.exists():
        return False
    # any json whose stem ends with "__<suffix>"
    pattern = f"*__{suffix}.json"
    return any(p.glob(pattern))

def resolve_guid(entry: dict, yt: dict | None, vid: str | None, link: str | None) -> str | None:
    """
    Return a stable GUID for both RSS and YouTube entries.

    Priority:
      1) RSS guid (stable)
      2) YouTube video id in canonical form: yt:video:<id>
      3) Atom/RSS id (only if it is NOT just a URL placeholder)
      4) yt-dlp id (fallback)
      5) link
    """
    # 1) RSS/Feed guid always wins
    if isinstance(entry, dict):
        g = entry.get("guid")
        if isinstance(g, str) and g.strip():
            return g.strip()

    # 2) For YouTube (or anything where vid is known), ALWAYS use canonical form
    if vid:
        return f"yt:video:{vid}"

    # 3) Atom feeds often store stable id in "id" — but your YT synthetic entries put a URL here.
    if isinstance(entry, dict):
        g = entry.get("id")
        if isinstance(g, str) and g.strip():
            s = g.strip()
            # If "id" is just a URL (common in your YT synthetic entries), don't treat it as GUID
            if not (s.startswith("http://") or s.startswith("https://")):
                return s

    # 4) fallback: yt-dlp
    if yt and yt.get("id"):
        return f"yt:video:{yt['id']}"

    # 5) last resort
    return link






def youtube_video_id(url: str) -> str | None:
    try:
        if not url:
            return None
        u = url.strip()
        if "youtu.be/" in u:
            return u.split("youtu.be/", 1)[1].split("?", 1)[0].split("/", 1)[0]
        if "youtube.com/watch" in u:
            qs = parse_qs(urlparse(u).query)
            return (qs.get("v") or [None])[0]
        return None
    except Exception:
        return None

def youtube_channel_id_from_feed_url(feed_url: str) -> str | None:
    # e.g. https://www.youtube.com/feeds/videos.xml?channel_id=UCxxxx
    try:
        qs = parse_qs(urlparse(feed_url).query)
        cid = (qs.get("channel_id") or [None])[0]
        return cid if cid and cid.startswith("UC") else None
    except Exception:
        return None

def youtube_uploads_playlist_id(channel_id: str) -> str | None:
    # UCxxxx -> UUxxxx
    if channel_id and channel_id.startswith("UC") and len(channel_id) > 2:
        return "UU" + channel_id[2:]
    return None


def get_youtube_channel_video_urls(channel_id: str, limit: int | None = None) -> list[str]:
    """
    Uses yt-dlp to list many uploads from the channel uploads playlist.
    Returns watch URLs.
    """
    pl = youtube_uploads_playlist_id(channel_id)
    if not pl:
        return []

    playlist_url = f"https://www.youtube.com/playlist?list={pl}"

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,      # fast list, no per-video deep extraction
    }

    # Only add playlistend if caller requested a limit
    if limit and limit > 0:
        ydl_opts["playlistend"] = limit

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        items = info.get("entries") or []
        urls = []
        for it in items:
            vid = (it or {}).get("id")
            if vid:
                urls.append(f"https://www.youtube.com/watch?v={vid}")
        return urls
    except Exception as e:
        logging.warning(f"yt-dlp playlist fetch failed: {e}", exc_info=True)
        return []


def _pick_largest_url(urls):
    """Pick the 'largest' thumbnail by common size hints if present."""
    if not urls:
        return None
    # Prefer URLs containing s176, s240, s400, s800 etc.
    def score(u: str) -> int:
        m = re.search(r"[=/-]s(\d+)", u)
        return int(m.group(1)) if m else 0
    return sorted(urls, key=score, reverse=True)[0]

def scrape_youtube_channel_avatar(channel_url: str, timeout: int = 15) -> str | None:
    """
    Fetch a YouTube channel page and extract the channel avatar URL.

    Works with:
      - https://www.youtube.com/channel/UC...
      - https://www.youtube.com/@handle
    """
    if not channel_url:
        return None

    # normalize
    channel_url = channel_url.strip()
    if channel_url.startswith("//"):
        channel_url = "https:" + channel_url
    if channel_url.startswith("www."):
        channel_url = "https://" + channel_url

    try:
        r = requests.get(channel_url, headers=_YT_HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        logging.warning(f"Channel page fetch failed for {channel_url}: {e}")
        return None

    # 1) meta og:image
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
    if m:
        return m.group(1)

    # 2) try to parse ytInitialData JSON and find avatar thumbnails
    # ytInitialData = {...};
    m = re.search(r"var ytInitialData\s*=\s*({.*?});\s*</script>", html, re.S)
    if m:
        blob = m.group(1)
        try:
            data = json.loads(blob)
            # Search common paths for avatar thumbnails
            # We'll just walk recursively and collect thumbnail URLs.
            urls = []

            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k in ("avatar", "thumbnails", "thumbnail"):
                            # collect any URLs
                            if isinstance(v, dict):
                                if "thumbnails" in v and isinstance(v["thumbnails"], list):
                                    for t in v["thumbnails"]:
                                        u = t.get("url")
                                        if u:
                                            urls.append(u)
                                if "url" in v and isinstance(v["url"], str):
                                    urls.append(v["url"])
                            elif isinstance(v, list):
                                for t in v:
                                    if isinstance(t, dict) and t.get("url"):
                                        urls.append(t["url"])
                        walk(v)
                elif isinstance(o, list):
                    for it in o:
                        walk(it)

            walk(data)

            # Filter to typical YT avatar hosts
            urls = [u for u in urls if "yt3." in u or "googleusercontent.com" in u or "ggpht.com" in u]
            picked = _pick_largest_url(urls)
            if picked:
                return picked
        except Exception:
            pass

    # 3) brute regex fallback: any yt3.* image URL
    m = re.search(r'(https://yt3\.(?:ggpht|googleusercontent)\.com/[^\s"\'<>]+)', html)
    if m:
        return m.group(1)

    return None

def safe_parse_boolean(value):
    """
    Safely parse a value to a boolean.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ['yes', 'true', '1', 'explicit']:
            return True
        elif value_lower in ['no', 'false', '0', 'clean']:
            return False
        else:
            logging.warning(f"Unrecognized explicit value '{value}'. Setting to None.")
            return None
    elif isinstance(value, (int, float)):
        return bool(value)
    else:
        logging.warning(f"Unsupported type for explicit value '{value}'. Setting to None.")
        return None

def normalize_duration(duration_seconds):
    """
    Normalize duration in seconds to HH:MM:SS format.
    """
    if duration_seconds is None:
        return None
    try:
        # Some feeds may provide duration in 'HH:MM:SS' or 'MM:SS' format as strings
        if isinstance(duration_seconds, str):
            parts = duration_seconds.split(':')
            parts = [int(part) for part in parts]
            while len(parts) < 3:
                parts.insert(0, 0)  # Prepend zeros for missing hours or minutes
            h, m, s = parts[-3:]
            total_seconds = h * 3600 + m * 60 + s
        else:
            total_seconds = int(duration_seconds)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02}:{m:02}:{s:02}"
    except Exception as e:
        logging.warning(f"Failed to normalize duration '{duration_seconds}': {e}")
        return None

def safe_parse_int(value):
    """
    Safely parse a value to an integer.
    
    Parameters:
    - value: The value to parse.
    
    Returns:
    - int or None: Parsed integer or None if parsing fails.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_parse_date(date_input):
    """
    Safely parse a date string or struct_time to a datetime object.

    Parameters:
    - date_input (str or struct_time): The date to parse.

    Returns:
    - datetime or None: Parsed datetime object or None if parsing fails.
    """
    try:
        if isinstance(date_input, time.struct_time):
            # Convert struct_time to datetime
            return datetime.fromtimestamp(time.mktime(date_input), timezone.utc)
        elif isinstance(date_input, str):
            return parser.parse(date_input)
        else:
            return None
    except Exception as e:
        logging.warning(f"Failed to parse date '{date_input}': {e}")
        return None

def parse_podcast_feed(feed_url):
    """
    Parse the podcast feed using podcastparser and extract metadata.
    
    If the URL is an Apple Podcasts page, use iTunes Lookup to get the real RSS feed URL.
    Otherwise fall back to normal requests + HTML/RSS parsing.
    """
    try:
        # 1) Apple page detection → iTunes Lookup
        if 'podcasts.apple.com' in feed_url:
            m = re.search(r'/id(\d+)', feed_url)
            if not m:
                logging.error(f"Can't find Apple ID in URL: {feed_url}")
                return None
            pid = m.group(1)
            lookup = requests.get(
                'https://itunes.apple.com/lookup',
                params={'id': pid},
                timeout=10
            )
            lookup.raise_for_status()
            results = lookup.json().get('results', [])
            if not results or not results[0].get('feedUrl'):
                logging.error(f"iTunes lookup failed for ID {pid}")
                return None

            real_feed = results[0]['feedUrl']
            logging.info(f"iTunes lookup: {feed_url} → {real_feed}")
            feed_url = real_feed

        # 2) Fetch the RSS/XML (either the original URL, or the feedUrl from lookup)
        resp = requests.get(feed_url, allow_redirects=True, timeout=10)
        resp.raise_for_status()
        content = resp.text

        # 3) Parse as RSS/XML
        feed_io   = io.StringIO(content)
        feed_data = podcastparser.parse(feed_url, feed_io)
        logging.debug(f"Parsed feed successfully: {feed_url}")

        # 4) (Optional) grab <itunes:author>
        m2 = re.search(r'<itunes:author>(.*?)</itunes:author>', content, re.IGNORECASE)
        if m2:
            feed_data['manual_author'] = m2.group(1).strip()

        return feed_data

    except Exception as e:
        logging.error(f"Error parsing podcast feed '{feed_url}': {e}", exc_info=True)
        return None


def extract_feed_categories(feed_data):
    """
    Extract categories from the feed data.

    Parameters:
    - feed_data: dict, parsed feed data.

    Returns:
    - categories: list of str, list of category names.
    """
    categories = []
    itunes_categories = feed_data.get('itunes_categories', [])

    if not itunes_categories:
        logging.debug("No itunes_categories found in feed data.")
        return categories

    # Log the structure of 'itunes_categories' for debugging
    logging.debug(f"itunes_categories: {itunes_categories}")

    # Flatten the list if it's a list of lists
    for category in itunes_categories:
        if isinstance(category, list):
            for sub_cat in category:
                if isinstance(sub_cat, str):
                    categories.append(sub_cat)
                    logging.debug(f"Extracted category: {sub_cat}")
                elif isinstance(sub_cat, dict):
                    text = sub_cat.get('text')
                    if text:
                        categories.append(text)
                        logging.debug(f"Extracted category: {text}")
                    else:
                        logging.debug("Sub-category dictionary does not contain 'text' key.")
                else:
                    logging.debug(f"Unrecognized sub-category format: {sub_cat}")
        elif isinstance(category, dict):
            # Handle dictionary format
            text = category.get('text')
            if text:
                categories.append(text)
                logging.debug(f"Extracted category: {text}")
            else:
                logging.debug("Category dictionary does not contain 'text' key.")
        elif isinstance(category, str):
            # Handle string format
            categories.append(category)
            logging.debug(f"Extracted category: {category}")
        else:
            logging.debug(f"Unrecognized category format: {category}")

    return categories

def extract_feed_language(feed_data):
    """
    Extract the language from the feed data.

    Parameters:
    - feed_data: dict, parsed feed data.

    Returns:
    - language: str or None, language of the podcast feed.
    """
    language = feed_data.get('language')
    if language:
        logging.debug(f"Extracted language: {language}")
    else:
        logging.debug("No language found in feed data.")
    return language

def extract_channel_image_url(feed_data):
    """
    Extracts the channel image URL from the feed data.

    Parameters:
    - feed_data: dict, parsed feed data.

    Returns:
    - channel_image_url: str or None, URL of the channel image.
    """
    channel_image_url = None
    # First, check 'cover_url' in feed_data
    if 'cover_url' in feed_data and feed_data.get('cover_url'):
        channel_image_url = feed_data.get('cover_url')
        logging.debug(f"Channel Image URL from 'cover_url': {channel_image_url}")
    # Then, check 'image' in feed_data
    elif 'image' in feed_data:
        image = feed_data.get('image')
        if isinstance(image, dict):
            channel_image_url = image.get('href')
            if channel_image_url:
                logging.debug(f"Channel Image URL from 'image' dict href: {channel_image_url}")
        elif isinstance(image, str):
            channel_image_url = image
            logging.debug(f"Channel Image URL from 'image' str: {channel_image_url}")
    else:
        logging.debug("No channel image URL found in feed data.")
    return channel_image_url

def extract_episode_image_url(episode):
    """
    Extracts the episode image URL from the episode data.

    Parameters:
    - episode: dict, parsed episode data.

    Returns:
    - episode_image_url: str or None, URL of the episode image.
    """
    episode_image_url = None
    # Check 'itunes_image' in episode
    if 'itunes_image' in episode:
        itunes_image = episode.get('itunes_image')
        if isinstance(itunes_image, dict):
            episode_image_url = itunes_image.get('href')
            if episode_image_url:
                logging.debug(f"Episode Image URL from 'itunes_image' dict href: {episode_image_url}")
        elif isinstance(itunes_image, str):
            episode_image_url = itunes_image
            logging.debug(f"Episode Image URL from 'itunes_image' str: {episode_image_url}")
    # Fallback: Check 'image' in episode
    if not episode_image_url and 'image' in episode:
        image = episode.get('image')
        if isinstance(image, dict):
            episode_image_url = image.get('href')
            if episode_image_url:
                logging.debug(f"Episode Image URL from 'image' dict href: {episode_image_url}")
        elif isinstance(image, str):
            episode_image_url = image
            logging.debug(f"Episode Image URL from 'image' str: {episode_image_url}")
    # Additional Fallback: Check 'episode_art_url' in episode
    if not episode_image_url and 'episode_art_url' in episode:
        episode_image_url = episode.get('episode_art_url')
        if episode_image_url:
            logging.debug(f"Episode Image URL from 'episode_art_url': {episode_image_url}")
    # If still not found, set to None
    if not episode_image_url:
        logging.debug("No episode image URL found in episode data.")
    return episode_image_url

def clean_html(raw_html):
    """
    Remove HTML tags from a string using BeautifulSoup.

    Parameters:
    - raw_html: str, the raw HTML string.

    Returns:
    - str, the cleaned text without HTML tags.
    """
    if not isinstance(raw_html, str):
        logging.warning(f"Description is not a string: {raw_html}")
        return "No Description Available"

    # Simple regex to check for HTML tags
    if not re.search(r'<[^>]+>', raw_html):
        logging.debug(f"Description does not contain HTML tags: {raw_html}")
        return raw_html  # Return as is or handle accordingly

    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

# Global cache for embeddings
_embedding_cache = {}

def cached_get_embedding(model, text, max_tokens=256):
    if text in _embedding_cache:
        logging.info("Cache hit for text.")
        return _embedding_cache[text]
    else:
        logging.info("Cache miss for text; computing embedding.")
        embedding = get_embedding(model, text, max_tokens)
        _embedding_cache[text] = embedding
        return embedding



# Updated compute_similarity that uses get_embedding().
def compute_similarity(model, text1, text2):
    if not model:
        logging.error("Sentence-BERT model is not loaded.")
        return 0.0
    try:
        # Get (possibly averaged) embeddings for each text using the cache.
        emb1 = cached_get_embedding(model, text1, max_tokens=256)
        emb2 = cached_get_embedding(model, text2, max_tokens=256)
        # Normalize and compute cosine similarity.
        emb1 = F.normalize(emb1, p=2, dim=0)
        emb2 = F.normalize(emb2, p=2, dim=0)
        similarity = torch.dot(emb1, emb2).item()
        return similarity
    except Exception as e:
        logging.error(f"Error computing similarity: {e}", exc_info=True)
        return 0.0

# Optionally update is_title_unique so that it uses get_embedding() as well.
def is_title_unique(new_title, chapters, model, similarity_threshold=0.6):
    if not model:
        logging.error("Sentence-BERT model is not loaded.")
        return False
    try:
        if not chapters:
            return True

        new_emb = get_embedding(model, new_title, max_tokens=256)
        new_emb = F.normalize(new_emb, p=2, dim=0)

        # Cache the embeddings for existing chapter titles.
        existing_texts = [chapter['title'] for chapter in chapters]
        # Compute embeddings for all titles in one batch
        existing_embeddings = model.encode(existing_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
        if existing_embeddings.dim() == 1:
            existing_embeddings = existing_embeddings.unsqueeze(0)
        existing_embeddings = F.normalize(existing_embeddings, p=2, dim=1)
        similarities = F.cosine_similarity(new_emb.unsqueeze(0), existing_embeddings)
        max_sim = similarities.max().item() if similarities.numel() > 0 else 0.0
        logging.debug(f"Max similarity of '{new_title}' with existing titles: {max_sim}")
        return max_sim < similarity_threshold
    except Exception as e:
        logging.error(f"Error in uniqueness check: {e}", exc_info=True)
        return False

def _expand_semicolon_list(items):
    out = []
    for it in items or []:
        parts = [p.strip() for p in str(it).split(';')]
        out.extend([p for p in parts if p])
    return out

def build_text_rules(cfg):
    tr = (cfg or {}).get('text_rules', {})

    small_words = set(_expand_semicolon_list(tr.get('small_words')))
    pronoun_starts = tuple(_expand_semicolon_list(tr.get('pronoun_starts')))
    boundary_tokens = set(_expand_semicolon_list(tr.get('boundary_tokens')))
    trailing_bad = set(_expand_semicolon_list(tr.get('trailing_bad')))

    sponsor_patterns = [re.compile(p, re.I) for p in tr.get('sponsor_patterns', [])]

    domain_re = re.compile(tr.get('domain_re', r"\b([a-z0-9\-]+\.)+(com|net|org|io|co|tv|fm)\b"), re.I)
    urlish_re = re.compile(tr.get('urlish_re', r"https?://|www\."), re.I)
    cta_re = re.compile(tr.get('cta_re', r"\b(click|tap|visit|use|enter|subscribe|follow|download|learn more|shop now)\b"), re.I)

    return {
        "SMALL_WORDS": small_words or {"a","an","and","as","at","but","by","for","in","nor","of","on","or","per","so","the","to","via"},
        "PRONOUN_STARTS": pronoun_starts or tuple(["i ","i’m","im ","you ","we ","they "]),
        "BOUNDARY_TOKENS": boundary_tokens or {'.','!','?','…','—','–',';',':'},
        "TRAILING_BAD": trailing_bad or set(),
        "SPONSOR_PATTERNS": sponsor_patterns,
        "DOMAIN_RE": domain_re,
        "URLISH_RE": urlish_re,
        "CTA_RE": cta_re,
    }



TEXT_RULES = build_text_rules(config)
SMALL_WORDS      = TEXT_RULES["SMALL_WORDS"]
PRONOUN_STARTS   = TEXT_RULES["PRONOUN_STARTS"]
BOUNDARY_TOKENS  = TEXT_RULES["BOUNDARY_TOKENS"]
TRAILING_BAD     = TEXT_RULES["TRAILING_BAD"]
SPONSOR_PATTERNS = TEXT_RULES["SPONSOR_PATTERNS"]
DOMAIN_RE        = TEXT_RULES["DOMAIN_RE"]
URLISH_RE        = TEXT_RULES["URLISH_RE"]
CTA_RE           = TEXT_RULES["CTA_RE"]

def aggregate_segments_with_stride(segments, window_size=30, stride=None):
    """
    Aggregate segments with overlap to create more candidate windows.
    Default stride is half the window size.
    Returns: list[dict] with {"text": str, "start": float}
    """
    if not segments:
        return []

    if stride is None or stride <= 0:
        stride = max(1, window_size // 2)

    windows = []
    i = 0
    while i < len(segments):
        window = segments[i:i + window_size]
        if not window:
            break
        text = " ".join(seg.get("text", "") for seg in window).strip()
        start = float(window[0].get("start", 0.0))
        windows.append({"text": text, "start": start})
        if i + window_size >= len(segments):
            break
        i += stride
    return windows


def clamp_to_complete_phrase(text: str, min_words: int, max_words: int) -> str:
    """
    Prefer the first full sentence; if too long, cut at a natural boundary (., !, ?)
    within a small headroom past max_words. Trim dangling function words at the end.
    """
    t = re.sub(r'\s+', ' ', text or '').strip()
    if not t:
        return t

    # 1) take the first full sentence if we can
    if nltk:
        sents = nltk.sent_tokenize(t)
    else:
        sents = re.split(r'(?<=[\.\!\?])\s+', t)
    first = (sents[0] if sents else t).strip()

    words = first.split()
    if min_words <= len(words) <= max_words:
        while words and words[-1].lower() in TRAILING_BAD:
            words.pop()
        return ' '.join(words)

    # 2) too long → search for punctuation boundary inside a headroom window
    window_words = t.split()
    window = ' '.join(window_words[:max_words + 8])  # small headroom to find punctuation
    m = re.search(r'(.+?[\.\!\?])(\s|$)', window)
    candidate = (m.group(1) if m else ' '.join(window_words[:max_words])).strip()

    cand_words = candidate.split()
    while cand_words and cand_words[-1].lower() in TRAILING_BAD:
        cand_words.pop()
    if len(cand_words) < min_words:
        cand_words = window_words[:min_words]
    return ' '.join(cand_words)

def titlecase_compact(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    words = s.split(" ")
    if not words:
        return s
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i != 0 and lw in SMALL_WORDS:
            out.append(lw)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)

def is_sponsor_segment(text: str) -> bool:
    t = " " + (text or "").lower() + " "
    for pat in SPONSOR_PATTERNS:
        if pat.search(t):
            return True
    return False


def is_promotional_title(title: str) -> bool:
    """Reject titles that *look* like ad reads."""
    t = title.strip().lower()
    return is_sponsor_segment(t)

def compact_title_from_text(text: str, min_words=4, max_words=8) -> str:
    """
    Fallback when the abstractive model meanders:
    - prefer proper-noun phrases / noun chunks,
    - avoid dangling fragments,
    - clamp to complete phrase and Title Case.
    """
    text = re.sub(r"[\.\!\?]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    phrases = []

    if nlp:
        doc = nlp(text)
        proper = [t.text for t in doc if t.pos_ == "PROPN"]
        noun_chunks = [nc.text for nc in getattr(doc, "noun_chunks", [])]
        candidates = proper + noun_chunks
        seen = set()
        for c in candidates:
            c = c.strip()
            if c and c.lower() not in seen and 2 <= len(c.split()) <= 6:
                seen.add(c.lower())
                phrases.append(c)
    else:
        toks = [w for w in re.findall(r"[A-Za-z0-9']+", text) if len(w) > 2]
        phrases = toks[:max_words]

    if phrases:
        left = phrases[0]
        right = phrases[1] if len(phrases) >= 2 else ""
        # only join when both sides look like solid phrases
        if right and 2 <= len(left.split()) <= 6 and 2 <= len(right.split()) <= 6:
            draft = f"{left}: {right}"
        else:
            draft = left
    else:
        draft = text

    draft = clamp_to_complete_phrase(draft, min_words, max_words)
    return titlecase_compact(draft)

def preprocess_text(text):
    """
    Clean filler, normalize spaces, strip bracketed asides; DO NOT scrub profanity.
    """
    try:
        fillers = ['uh', 'um', 'you know', 'like', 'sort of', 'kind of']
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, fillers)) + r')\b', flags=re.I)
        text = pattern.sub('', text)
        text = re.sub(r"[\[\(].*?[\]\)]", " ", text)  # [laughter], (applause), etc.
        text = re.sub(r"[^\w\s\-\':\.]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}", exc_info=True)
        return text

def get_entry_field(entry, key, default=None):
    return entry.get(key, default) if isinstance(entry, dict) else getattr(entry, key, default)

def get_entry_enclosures(entry):
    enc = get_entry_field(entry, "enclosures", []) or []
    return enc if isinstance(enc, list) else []


def clean_title(title):
    """
    Normalize spacing and Title Case (do NOT force dropping punctuation or add periods).
    """
    try:
        title = (title or "").strip().strip('"').strip("'")
        title = re.sub(r"\s+", " ", title).strip()
        return titlecase_compact(title)
    except Exception as e:
        logging.error(f"Error cleaning title '{title}': {e}", exc_info=True)
        return title

def is_title_valid(title, config):
    """
    Enforce concise, topical titles; avoid first-person rambles and sponsor reads.
    Sentence style: allow longer ranges and require at least one content word.
    """
    try:
        t = (title or "").strip()
        if not t:
            return False
        t_low = t.lower()
        rules = config.get('chapter_generation', {}).get('title', {})
        style = (rules.get('style') or 'title').strip().lower()

        if t_low.startswith(PRONOUN_STARTS):
            return False
        if is_sponsor_segment(t):
            return False

        if style == "sentence":
            min_words = int(rules.get('min_words', 8))
            max_words = int(rules.get('max_words', 24))
        else:
            min_words = int(rules.get('min_words', 4))
            max_words = int(rules.get('max_words', 8))

        wc = len(t.split())
        if wc < min_words or wc > max_words:
            return False

        # Require at least one content word (NOUN/PROPN/VERB) to avoid empty phrasal junk
        if nlp:
            doc = nlp(t)
            if not any(tok.pos_ in ('NOUN','PROPN','VERB') for tok in doc):
                return False

        # For sentence style, prefer endings with terminal punctuation
        if style == "sentence" and not re.search(r"[.!?][\"’”)]?$", t):
            return False

        return True
    except Exception as e:
        logging.error(f"Error validating title '{title}': {e}", exc_info=True)
        return False


def verify_entities(title, segment_text):
    """
    Verify that the entities in the title are present in the segment text.
    """
    if not nlp:
        logging.warning("spaCy model is not available for entity verification.")
        return True
    try:
        title_doc = nlp(title)
        segment_doc = nlp(segment_text)
        title_entities = set([ent.text.lower() for ent in title_doc.ents])
        segment_entities = set([ent.text.lower() for ent in segment_doc.ents])
        if not title_entities:
            return True
        verified = bool(title_entities & segment_entities)
        logging.debug(f"Entity verification: {verified} for title '{title}'")
        return verified
    except Exception as e:
        logging.error(f"Error during entity verification: {e}", exc_info=True)
        return False

def generate_chapter_title(segment_text, config=None):
    try:
        if not summarizer:
            logging.error("Summarization pipeline is not available.")
            return None

        rules = config.get('chapter_generation', {}).get('title', {}) if config else {}
        style = (rules.get('style') or 'title').strip().lower()   # ← NEW: 'sentence'|'bullet'|'title'
        min_words = int(rules.get('min_words', 4))
        max_words = int(rules.get('max_words', 8))
        drop_sponsors = bool(rules.get('drop_sponsor_segments', True))

        if not segment_text or not segment_text.strip():
            return None

        if drop_sponsors and is_sponsor_segment(segment_text):
            logging.info("Skipping sponsor segment for chapter generation.")
            return None

        segment_text = preprocess_text(segment_text)

        # Hard input clamp for focus (avoid long sequences to the model)
        max_input_length = 2000
        if len(segment_text) > max_input_length:
            segment_text = segment_text[:max_input_length]

        # --- Summarize succinctly, but with truncation to avoid tokenizer overflows
        try:
            raw = summarizer(
                segment_text,
                max_length=config['summarizer']['max_length'],
                min_length=config['summarizer']['min_length'],
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True,
                truncation=True,                 # ← IMPORTANT
            )[0]['summary_text']
        except Exception as e:
            logging.warning(f"Summarizer failed, using extractive fallback: {e}")
            raw = ""

        raw = re.sub(r"\s+", " ", (raw or "").strip())

        # === Style switch ===
        if style == "sentence":
            # Keep exactly one full sentence; fallback to best extractive if needed
            sent = clamp_to_full_sentence(raw)
            if not sent:
                sent = best_sentence_by_noun_density(segment_text)

            # Light length clamp to keep it crisp; don't mutilate the sentence
            words = sent.split()
            if len(words) > max_words:
                # Try trimming to last punctuation inside the window; else hard-trim
                head = " ".join(words[:max_words + 8])
                m = re.search(r"(.+[.!?])[\"’”)]?\s*$", head)
                sent = (m.group(1) if m else " ".join(words[:max_words])).strip()
            title = sent.strip()

        elif style == "bullet":
            # (optional) If you ever want bullet mode in prod later
            # use compact_title_from_text() as you were doing in the test script
            draft = raw if raw else segment_text
            sent = clamp_to_full_sentence(draft) or best_sentence_by_noun_density(segment_text)
            # compress to a punchy phrase
            draft = compact_title_from_text(sent, min_words=min_words, max_words=max_words)
            title = clean_title(draft)

        else:
            # Legacy: compact "title case" (your previous default)
            # Ensure sentence end, then compact if needed
            if not re.search(r"[\.!?]$", raw):
                cut = re.search(r".*[\.!?]", raw)
                if cut:
                    raw = cut.group(0)
            draft = raw
            words = draft.split()
            if len(words) > max_words:
                draft = " ".join(words[:max_words])
            if len(draft.split()) < min_words or draft.lower().startswith(PRONOUN_STARTS):
                draft = compact_title_from_text(segment_text, min_words=min_words, max_words=max_words)
            title = clean_title(draft)

        # Hard block promo-ish titles
        if is_promotional_title(title):
            return None

        # Relevance sanity check (fallback to extractive compact if way off)
        title_similarity = compute_similarity(sentence_model, title, segment_text) if sentence_model else 1.0
        if title_similarity < 0.20:
            if style == "sentence":
                title = best_sentence_by_noun_density(segment_text)
            else:
                title = clean_title(compact_title_from_text(segment_text, min_words=min_words, max_words=max_words))
            if is_promotional_title(title):
                return None

        # Style-aware validation (sentence can be longer)
        if not is_title_valid(title, config):
            return None

        return title
    except Exception as e:
        logging.error(f"Error generating chapter title: {e}", exc_info=True)
        return None



def aggregate_segments_non_overlapping(segments, window_size=5):
    """
    Aggregate segments into non-overlapping windows.
    
    Parameters:
    - segments (list): List of transcript segments.
    - window_size (int): Number of segments to aggregate.
    
    Returns:
    - list of str: Aggregated texts.
    """
    aggregated_texts = []
    for i in range(0, len(segments), window_size):
        window = segments[i:i + window_size]
        aggregated_text = ' '.join([segment.get('text', '') for segment in window])
        aggregated_texts.append(aggregated_text)
    return aggregated_texts

def add_chapters_to_transcript(transcript_json_path, config):
    similarity_threshold = config['chapter_generation']['similarity_threshold']
    max_chapters = config['chapter_generation']['max_chapters']
    aggregation_window_size = config['chapter_generation']['aggregation_window_size']
    title_rules = config.get('chapter_generation', {}).get('title', {})
    drop_sponsors = bool(title_rules.get('drop_sponsor_segments', True))

    # timing guards (still honored; gentler handling of the first window)
    min_first_chapter_sec = int(title_rules.get('min_first_chapter_sec', 90))
    min_gap_sec = int(title_rules.get('min_gap_sec', 45))

    try:
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data.get('segments', [])
        chapters = []

        if not segments:
            logging.warning("No segments found in transcript.")
            return

        # Always add Intro @ 0:00
        chapters.append({'title': 'Intro', 'time': '0:00'})
        last_chapter_start_sec = 0.0
        logging.info("Added chapter: 'Intro' at 0:00")

        # Overlapping windows: more chances to find solid topics even with large windows
        windows = aggregate_segments_with_stride(
            segments,
            window_size=aggregation_window_size,
            stride=max(1, aggregation_window_size // 2),
        )
        logging.debug(f"Aggregated into {len(windows)} overlapping windows "
                      f"(size={aggregation_window_size}, stride={max(1, aggregation_window_size // 2)}).")

        skipped = {"sponsor":0, "similarity":0, "invalid_title":0, "duplicate":0, "too_close":0}

        for idx, win in enumerate(windows):
            aggregated_text = win["text"]
            start_time = float(win["start"])

            if not aggregated_text:
                continue

            # Gentle handling for the very first *content* chapter:
            # If the first window starts before min_first_chapter_sec, pin it forward to that mark
            # (instead of skipping the whole window).
            if idx == 0 and start_time < min_first_chapter_sec:
                start_time = float(min_first_chapter_sec)

            # After the first window, enforce min_first_chapter_sec normally
            if idx > 0 and start_time < min_first_chapter_sec:
                continue

            # Sponsor skip (window text)
            if drop_sponsors and is_sponsor_segment(aggregated_text):
                skipped["sponsor"] += 1
                continue

            # Avoid clustering; ensure gap from last chapter
            if (start_time - last_chapter_start_sec) < min_gap_sec:
                skipped["too_close"] += 1
                continue

            # Similarity gate (vs previous window text) – but allow long gaps to force a chapter
            attempt = True
            if idx > 0:
                prev_text = windows[idx - 1]["text"]
                sim = compute_similarity(sentence_model, aggregated_text, prev_text)
                logging.debug(f"[win {idx}] Similarity vs prev: {sim:.3f} (threshold={similarity_threshold})")
                if sim >= similarity_threshold and (start_time - last_chapter_start_sec) < (min_gap_sec * 2):
                    attempt = False
                    skipped["similarity"] += 1

            if not attempt:
                continue

            title = generate_chapter_title(aggregated_text, config=config)
            if not title:
                skipped["invalid_title"] += 1
                continue

            if not is_title_unique(title, chapters, sentence_model, similarity_threshold=0.6):
                skipped["duplicate"] += 1
                continue

            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"{minutes}:{seconds:02d}"

            chapters.append({'title': title, 'time': timestamp})
            last_chapter_start_sec = start_time
            logging.info(f"Added chapter: '{title}' at {timestamp}")

            if len(chapters) >= max_chapters:
                logging.info(f"Reached maximum number of chapters: {max_chapters}")
                break

        logging.info(f"Chaptering skipped: {skipped}")

        data['chapters'] = chapters
        with open(transcript_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Chapters added successfully to {transcript_json_path}")

    except Exception as e:
        logging.error(f"Failed to add chapters: {e}", exc_info=True)

def restart_self(reason: str):
    logging.critical(f"{reason} → restarting process with same CLI args: {' '.join(sys.argv)}")
    # flush logs so you don’t lose the last lines
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    os.execv(sys.executable, [sys.executable] + sys.argv)



def process_chunk(chunk, pipeline):
    try:
        transcription_data = transcribe_and_diarize(chunk, pipeline)
        return transcription_data

    except Exception as e:
        msg = str(e)

        # 🔥 Fatal CUDA state: must restart whole process
        if "device-side assert triggered" in msg:
            restart_self(f"FATAL CUDA device-side assert triggered while processing chunk: {chunk}")

        logging.error(f"Unexpected error processing chunk {chunk}: {e}", exc_info=True)
        return None



def parse_cli_args():
    parser = argparse.ArgumentParser(description="Podcast pipeline runner")
    # Use the mini file instead of master_rss.json
    parser.add_argument('--master_rss_mini', action='store_true',
                        help='Use watcher_json/master_rss_mini.json instead of master_rss.json')
    # Explicit file override, e.g. -f my_list.json
    parser.add_argument('-f', '--feeds-file', default=None,
                        help='Feeds JSON filename under watcher_json (e.g. master_rss.json, master_rss_mini.json)')
    # Limit how many entries per feed to process (0 or absent = ALL)
    parser.add_argument('-n', '--limit', type=int, default=None,
                        help='Max entries per feed (default: all)')

    # NEW: restart interval (prefer seconds for test convenience; hours still supported)
    parser.add_argument('-r', '--restart-hours', type=float, default=None,
                        help='If set, re-run the entire feed list from the top every N hours.')
    parser.add_argument('--restart-seconds', type=float, default=None,
                        help='Like --restart-hours, but in seconds (useful for testing).')

    # TEST-ONLY simulation knobs to avoid long real runs
    parser.add_argument('--simulate-feed-secs', type=float, default=0.0,
                        help='TEST ONLY: sleep this many seconds at the start of each feed to simulate work.')
    parser.add_argument('--simulate-entry-secs', type=float, default=0.0,
                        help='TEST ONLY: sleep this many seconds for each entry to simulate work.')
    parser.add_argument("--migrate-legacy", action="store_true")
    parser.add_argument("--migrate-dry-run", action="store_true")
    parser.add_argument('--dry-run', action='store_true',
                    help='Do not download/transcribe; only resolve IDs and report what would be processed/skipped')
    parser.add_argument("--migrate-size-limit-mb", type=float, default=None)
    parser.add_argument("--migrate-report-name", default="legacy_migration_report.json")
    parser.add_argument("--migrate-max-len", type=int, default=100)


    args, unknown = parser.parse_known_args()

    # Support your existing shorthand like "--30"
    for tok in unknown:
        m = re.fullmatch(r'--(\d{1,4})', tok)
        if m:
            args.limit = int(m.group(1))
    return args



def process_entry(entry, channel_transcript_dir, download_dir, channel_title, pipeline, config,
                  feed_data, channel_summary, channel_author, dry_run=False, suffix_map=None):
    """
    Process a single podcast episode entry.
    Optimized:
      - For YouTube: compute vid/guid/suffix WITHOUT yt-dlp, skip fast if already processed
      - Only call yt-dlp when we truly need to process (download/transcribe or metadata fill)
      - suffix_map can be passed in from the feed loop to avoid repeated bootstraps
    """
    chunks_since_refresh = 0
    REFRESH_EVERY = 80  # keep 5-min chunks; refresh every ~6.5 hours if busy

    try:
        # 0) Resolve link early
        link = entry.get("link") or (entry.get("id") if isinstance(entry.get("id"), str) else None)
        is_youtube = bool(link) and ("youtube.com/watch" in link or "youtu.be/" in link)

        # Skip shorts early
        if link and "/shorts/" in link:
            logging.info(f"Skipping YouTube Short: {link}")
            return None, pipeline

        # 1) FAST ID path (NO yt-dlp): compute vid/guid/suffix for YouTube
        yt = None
        vid = None

        if is_youtube:
            # URL parse is cheap and usually enough to get stable id
            vid = youtube_video_id(link)
            # Canonical GUID for YouTube, stable across title changes
            guid = resolve_guid(entry, yt=None, vid=vid, link=link)  # -> yt:video:<id> when vid exists
        else:
            guid = resolve_guid(entry, yt=None, vid=None, link=link)

        suffix = stable_id_suffix(vid=vid, guid=guid, link=link)

        logging.debug(f"Resolved guid={guid} suffix={suffix} channel_transcript_dir={channel_transcript_dir}")
        logging.debug(f"Looking for existing json pattern: *__{suffix}.json (under {channel_transcript_dir})")

        # Ensure we have a suffix_map (prefer caller-provided)
        if suffix_map is None:
            # fallback (still works) but slower; ideally you pass it from the feed loop
            idx = bootstrap_index_from_folder(channel_transcript_dir)
            suffix_map = idx["suffixes"]

        # DRY RUN: report and stop here (no yt-dlp)
        if dry_run:
            logging.info(f"[DRY RUN] Would process (suffix={suffix}) link={link}")
            return None, pipeline

        # Desired filename for THIS run (needs title, but we can defer title until we know it’s new)
        old_name = suffix_map.get(suffix)
        if old_name:
            # Light recovery if index points to missing file (rare, but keep your behavior)
            old_path = Path(channel_transcript_dir) / old_name
            if not old_path.exists():
                hit = next(Path(channel_transcript_dir).glob(f"*__{suffix}.json"), None)
                if hit:
                    suffix_map[suffix] = hit.name
                    save_index(channel_transcript_dir)

            logging.info(f"Already processed (suffix={suffix}); skipping.")
            return None, pipeline

        # 2) Only NOW pay for yt-dlp metadata (because we know it’s not processed)
        if is_youtube:
            yt = yt_dlp_info(link)  # can be slow; we avoided it for already-processed entries
            if yt and yt.get("id"):
                vid = yt["id"]  # may be same as URL id; ok if it’s more canonical
                guid = f"yt:video:{vid}"
                suffix = stable_id_suffix(vid=vid, guid=guid, link=link)

                # If re-derived suffix differs, check skip again (rare edge case)
                old_name2 = suffix_map.get(suffix)
                if old_name2:
                    logging.info(f"Already processed after yt-dlp canonicalization (suffix={suffix}); skipping.")
                    return None, pipeline

        # 3) Now we can pick the real title
        raw_title = (yt.get("title") if yt else None) or entry.get("title") or "Unknown Title"
        sanitized_title = sanitize_filename(raw_title)

        # Now that we know it hasn't been processed, compute the stem/paths for THIS run
        file_stem = make_episode_stem_with_suffix(raw_title, suffix)

        mp3_filename = f"{file_stem}.mp3"
        mp3_file_path = os.path.join(download_dir, mp3_filename)
        transcript_filename = os.path.join(channel_transcript_dir, f"{file_stem}.json")

        logging.debug(f"Episode title: {raw_title}")
        logging.debug(f"File stem: {file_stem}")

        # Check if transcript already exists
        if os.path.exists(transcript_filename):
            logging.info(f"Transcript already exists: {transcript_filename}")
            return None, pipeline  # Skip to the next entry

        # Download audio file if it doesn't exist
        if not os.path.exists(mp3_file_path):
            logging.info(f"Downloading audio for: {sanitized_title}")
            audio_file = download_audio(entry, download_dir, mp3_filename)
            if not audio_file:
                logging.error(f"Failed to download audio for entry: {sanitized_title}")
                return None, pipeline  # Skip to the next entry
        else:
            logging.info(f"MP3 file already exists: {mp3_file_path}")

        # Create fixed-length WAV chunks directly from the MP3
        chunks = convert_to_5min_wav_chunks(
            mp3_file_path,
            download_dir,
            chunk_length_ms=5 * 60 * 1000  # 5 minutes
        )
        if not chunks:
            logging.error(f"Failed to split {mp3_file_path} into WAV chunks. Skipping.")
            return None, pipeline


        combined_data = {
            'transcript': '',
            'segments': [],
            'metadata': {}
            # Removed 'summary' as per user request
        }

        

        cumulative_time = 0  # To adjust segment times based on chunk offsets
        chunk_length_ms = 300000  # 5 minutes in milliseconds

        # Process chunks sequentially
        for idx, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {idx + 1}/{len(chunks)}: {chunk}")

            # Validate audio file
            if not validate_audio(chunk):
                logging.error(f"Invalid audio file: {chunk}. Skipping.")
                cumulative_time += chunk_length_ms / 1000.0  # Still increment time
                continue


            # Transcribe and diarize
            transcription_data = process_chunk(chunk, pipeline)  # Corrected call
            if transcription_data:
                # Append transcript
                combined_data['transcript'] += transcription_data.get('transcript', '') + ' '

                # Adjust and append segments
                for segment in transcription_data.get('segments', []):
                    adjusted_segment = {
                        'start': segment['start'] + cumulative_time,
                        'end': segment['end'] + cumulative_time,
                        'text': segment['text'],
                        'speaker': segment['speaker']
                    }
                    combined_data['segments'].append(adjusted_segment)
            else:
                logging.error(f"Failed to retrieve transcription data from chunk: {chunk}")

            # --- memory hygiene between chunks ---
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # --- soft guard: refresh pipeline if process RSS climbs too high ---
            try:
                rss_gb = psutil.Process().memory_info().rss / (1024**3)
                if rss_gb > 6.0:  # tune for your box (16 GB system → 6–7 GB is a good tripwire)
                    logging.warning(f"RSS {rss_gb:.2f} GB > 6.0 GB; re-initializing diarization pipeline")
                    try:
                        del pipeline
                    except Exception:
                        pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pipeline = initialize_diarization_pipeline()
                    chunks_since_refresh = 0
            except Exception:
                pass
            # --- scheduled refresh of the diarization pipeline to avoid fragmentation ---
            chunks_since_refresh += 1
            if chunks_since_refresh >= REFRESH_EVERY:
                logging.info("Refreshing diarization pipeline to avoid long-run fragmentation…")
                try:
                    del pipeline
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pipeline = initialize_diarization_pipeline()   # rebinds local 'pipeline' for the remaining chunks
                chunks_since_refresh = 0

            cumulative_time += chunk_length_ms / 1000.0  # Increment by 5 minutes

        # ✅ GUARD: if GPU blew up and we got nothing, DO NOT WRITE/OVERWRITE JSON
        if (not combined_data.get("segments")) and (not (combined_data.get("transcript") or "").strip()):
            logging.critical(
                f"EMPTY transcript for suffix={suffix} title={raw_title} "
                f"(likely CUDA failure). Not writing JSON."
            )
            return None, pipeline
        
        # Extract and clean description from 'description' field
        raw_description = get_entry_field(entry, "description", "No Description Available")
        logging.debug(f"Raw Description for '{sanitized_title}': {raw_description}")

        if raw_description != 'No Description Available':
            clean_description = clean_html(raw_description)
            logging.debug(f"Cleaned Description for '{sanitized_title}': {clean_description}")
        else:
            logging.warning(f"No description found for episode: {sanitized_title}")
            clean_description = 'No Description Available'

        # Extract publication date
        publication_date = safe_parse_date(
            entry.get('published_parsed') or
            entry.get('published') or
            entry.get('updated') or
            entry.get('pubDate') or
            entry.get('date')
        )
        # ✅ YouTube fallback (yt-dlp) if RSS was null
        if publication_date is None and yt:
            try:
                # 1) Best: exact timestamp (seconds since epoch)
                ts = yt.get("timestamp") or yt.get("release_timestamp")
                if ts:
                    publication_date = datetime.fromtimestamp(int(ts), tz=timezone.utc)

                # 2) Sometimes release_date exists (YYYYMMDD)
                if publication_date is None:
                    rd = yt.get("release_date")
                    if rd:
                        publication_date = datetime.strptime(str(rd), "%Y%m%d").replace(tzinfo=timezone.utc)

                # 3) Fallback: upload_date (YYYYMMDD)
                if publication_date is None:
                    ud = yt.get("upload_date")
                    if ud:
                        # upload_date has no time-of-day; pick midnight UTC
                        publication_date = datetime.strptime(str(ud), "%Y%m%d").replace(tzinfo=timezone.utc)

            except Exception as e:
                logging.warning(f"Failed to derive publication_date from yt-dlp fields: {e}", exc_info=True)

        publication_date_str = publication_date.isoformat() if publication_date else None

        # Extract episode image URL using the integrated function
        episode_image_url = extract_episode_image_url(entry)
        logging.info(f"Episode Image URL: {episode_image_url if episode_image_url else 'None'}")

        # Extract channel image URL using the integrated function
        channel_image_url = extract_channel_image_url(feed_data)
        logging.info(f"Channel Image URL: {channel_image_url if channel_image_url else 'None'}")

        

        # Extract explicit value
        explicit_value = entry.get('itunes_explicit')
        if explicit_value is None:
            # Fallback to feed-level itunes_explicit using integrated metadata extraction
            if feed_data:
                explicit_value = feed_data.get('explicit')
                logging.debug(f"Feed-level explicit value found: {explicit_value}")
            else:
                explicit_value = None
        # Parse the explicit value using safe_parse_boolean
        explicit = safe_parse_boolean(explicit_value)

        

        # Extract categories and language using the integrated metadata extraction
        categories = extract_feed_categories(feed_data) if feed_data else None
        language = extract_feed_language(feed_data) if feed_data else None

        authors_list = entry.get("authors")
        episode_author = None

        # 1) Check 'authors' array (like feedparser can store for some feeds)
        if authors_list and len(authors_list) > 0:
            # e.g. authors_list = [{"name": "Shawn Ryan"}]
            episode_author = authors_list[0].get("name")

        # 2) Fallback to 'author' or 'itunes_author'
        if not episode_author:
            episode_author = entry.get("author") or entry.get("itunes_author")

        # 3) Fallback to channel_author
        if not episode_author:
            episode_author = channel_author




        # Extract duration from multiple possible fields
        duration = normalize_duration(
            entry.get('itunes_duration') or
            entry.get('total_time') or
            entry.get('duration') or
            entry.get('length') or
            entry.get('time')  # Add more as needed
        )

        # Extract episode number from multiple possible fields
        episode_number = safe_parse_int(
            entry.get('itunes_episode') or
            entry.get('episode_number') or
            entry.get('number') or
            entry.get('episode')  # Add more as needed
        )

        #Translated is False unless the .json has been modified by translated.py then the value is equal to true
        translated = False

        link = entry.get("link") or (entry.get("id") if isinstance(entry.get("id"), str) else None)

        is_youtube = bool(link) and ("youtube.com/watch" in link or "youtu.be/" in link)


        link = entry.get("link")
        if link and "/shorts/" in link:
            logging.info(f"Skipping YouTube Short: {link}")
            return None, pipeline


        # Fill missing pieces from yt-dlp
        if yt:
            # duration (seconds -> HH:MM:SS)
            if not duration and yt.get("duration") is not None:
                duration = normalize_duration(int(yt["duration"]))

            # episode thumbnail
            if not episode_image_url:
                episode_image_url = yt.get("thumbnail")

            # channel avatar (sometimes present)
            if not channel_image_url:
                channel_image_url = yt.get("channel_avatar") or yt.get("uploader_avatar") or yt.get("channel_thumbnail") or yt.get("uploader_thumbnail")

            # channel avatar (scraped)    
            if (yt is not None) and (not channel_image_url):
                ch_url = yt.get("channel_url") or yt.get("uploader_url")
                if ch_url:
                    channel_image_url = scrape_youtube_channel_avatar(ch_url)
                    logging.info(f"Channel Image URL (scraped): {channel_image_url if channel_image_url else 'None'}")

            # description fallback
            if (not clean_description) or (clean_description == "No Description Available"):
                clean_description = yt.get("description") or clean_description

            # explicit-ish (yt doesn't always have a clean boolean; keep yours unless missing)
            # categories/tags
            if not categories:
                categories = yt.get("categories") or []
            # language: yt-dlp may not give; keep feed default unless you add detect later

        # For YouTube specifically: ensure audio_url isn't null
        enclosures = get_entry_enclosures(entry)
        audio_url = None
        if enclosures:
            first = enclosures[0]
            audio_url = (first.get("href") if isinstance(first, dict) else getattr(first, "href", None)) \
                        or (first.get("url") if isinstance(first, dict) else getattr(first, "url", None))
        audio_url = audio_url or link



        # Compile episode metadata
        metadata = {
            'channel_title': channel_title,
            'episode_title': raw_title,
            'sanitized_episode_title': sanitized_title,
            'publication_date': publication_date_str,
            'duration': duration,
            'episode_number': episode_number,
            'explicit': explicit,
            "author": episode_author,
            'summary': channel_summary,
            'guid': guid,
            'audio_url': audio_url,                 # ✅ will not be null for YT now
            'image_url': episode_image_url,         # ✅ from yt if missing
            'channel_image_url': channel_image_url, # ✅ from yt if missing
            'description': clean_description,
            'categories': categories or [],
            'language': language,
            'link': link,
            'translated': translated,
            'extra': (  # optional, safe
                {
                    "view_count": yt.get("view_count"),
                    "like_count": yt.get("like_count"),
                    "comment_count": yt.get("comment_count"),
                    "upload_date": yt.get("upload_date"),
                    "channel_id": yt.get("channel_id"),
                    "channel_url": yt.get("channel_url"),
                    "tags": yt.get("tags"),
                } if yt else {}
            ),
        }


        # Log the extracted metadata
        logging.info("\n--- Extracted Episode Metadata ---")
        for key, value in metadata.items():
            # Format key for better readability
            formatted_key = key.replace('_', ' ').capitalize()
            logging.info(f"{formatted_key}: {value}")

        # Log feed-level categories separately
        logging.info("\n--- Feed-Level Categories ---")
        if categories:
            for idx, category in enumerate(categories, start=1):
                logging.info(f"Category {idx}: {category}")
        else:
            logging.info("No categories found.")

        # Combine metadata with transcript and segments
        combined_data['metadata'] = metadata

        # Save combined data as a single JSON file
        logging.info(f"Saving combined transcription, metadata, and segments to {transcript_filename}")
        try:
            os.makedirs(os.path.dirname(transcript_filename), exist_ok=True)  # Ensure directory exists
            with open(transcript_filename, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=4, ensure_ascii=False)
            logging.info(f"Combined data saved successfully: {transcript_filename}")
        except Exception as e:
            logging.error(f"Failed to save combined data: {e}", exc_info=True)
            return None, pipeline
        
        
        # ✅ record in index AFTER successful save (so next run skips instantly)
        try:
            idx = load_index(channel_transcript_dir)
            idx["suffixes"][suffix] = Path(transcript_filename).name
            save_index(channel_transcript_dir)
        except Exception as e:
            logging.warning(f"Failed to update processed index for suffix={suffix}: {e}", exc_info=True)

        # Add chapters to the transcript with summarization
        add_chapters_to_transcript(transcript_filename, config)

        # Delete chunk WAV files first
        for chunk in chunks:
            try:
                os.remove(chunk)
                logging.info(f"Deleted chunk file: {chunk}")
            except FileNotFoundError:
                logging.warning(f"Chunk file not found, could not delete: {chunk}")
            except Exception as e:
                logging.warning(f"Could not delete chunk file '{chunk}': {e}")

        # Delete the original MP3 file after successful processing
        try:
            os.remove(mp3_file_path)
            logging.info(f"Deleted MP3 file: {mp3_file_path}")
        except Exception as e:
            logging.warning(f"Could not delete MP3 file: {e}")

        return combined_data, pipeline
    except Exception as e:
        logging.error(f"An error occurred while processing entry '{sanitized_title}': {e}", exc_info=True)
        return None, pipeline

# ── 3a) Legacy Main Function — NOW deadline-aware and testable ──────────────
def run_cycle(feeds_filename='master_rss.json', limit_per_feed=0,
              deadline_mono=None, simulate_feed_secs=0.0, simulate_entry_secs=0.0):
    """
    Returns True if the entire feeds list was processed (full pass),
    or False if aborted early due to time budget (deadline_mono).
    """
    def _deadline_hit():
        return (deadline_mono is not None) and (time.monotonic() >= deadline_mono)

    # pick the feeds file
    master_file = DATABASE_ROOT / "watcher_json" / feeds_filename
    try:
        with open(master_file, 'r', encoding='utf-8') as f:
            master = json.load(f)
    except FileNotFoundError:
        logging.critical(f"Could not find {master_file}; run update_master_rss.py first.")
        return True  # treat as "full" so we don't loop forever on a missing file

    logging.info(f"🔧 Feeds file: {master_file.name} | Entry limit per feed: "
                 f"{'ALL' if not limit_per_feed or limit_per_feed <= 0 else limit_per_feed}")

    podcast_feeds = [entry['url'] for entry in master]
    failed_feeds = []
    download_dir        = DATABASE_ROOT / "podcasts"
    base_transcript_dir = DATABASE_ROOT / "transcripts"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(base_transcript_dir, exist_ok=True)

    diarization_pipeline = initialize_diarization_pipeline()
    if not diarization_pipeline:
        logging.critical("Diarization pipeline failed to initialize. Exiting.")
        return False

    if not summarizer:
        logging.critical("Summarization pipeline is not available. Exiting.")
        return False

    full_pass = True  # assume success unless we abort early

    for feed_idx, feed_url in enumerate(podcast_feeds, start=1):
        if _deadline_hit():
            logging.warning(f"⏰ Time budget exhausted before feed {feed_idx}/{len(podcast_feeds)}; aborting pass.")
            full_pass = False
            break

        if simulate_feed_secs > 0:
            time.sleep(simulate_feed_secs)

        logging.info(f"[{feed_idx}/{len(podcast_feeds)}] Processing feed: {feed_url}")

        feed = parse_feed(feed_url)
        if not feed:
            failed_feeds.append(feed_url)
            continue

        entries = getattr(feed, 'entries', [])
        if not entries:
            logging.warning(f"No entries to process for feed: {feed_url}")
            failed_feeds.append(feed_url)
            continue
        is_youtube = "youtube.com/feeds/videos.xml" in feed_url.lower()

        if is_youtube:
            # Minimal feed_data so the rest of your code has what it needs
            channel_title = (getattr(feed.feed, "title", None) or "YouTube_Channel")
            feed_data = {
                "title": channel_title,
                "language": "en",
                "summary": getattr(feed.feed, "subtitle", None) or getattr(feed.feed, "description", None),
                "authors": [getattr(feed.feed, "author", None)] if getattr(feed.feed, "author", None) else [],
            }
        else:
            feed_data = parse_podcast_feed(feed_url)
            if not feed_data:
                logging.error(f"Failed to parse feed data for: {feed_url}")
                continue



        channel_title = feed_data.get('title', 'Unknown_Channel')
        sanitized_channel_title = sanitize_filename(channel_title)
        channel_author = (feed_data.get('authors', [None])[0]
                          or feed_data.get('itunes_tags', {}).get('author')
                          or feed_data.get('author')
                          or feed_data.get('manual_author'))
        channel_summary = feed_data.get('summary') or feed_data.get('description')
        source_lang = feed_data.get('language', 'eng').lower()

        channel_transcript_dir = os.path.join(base_transcript_dir, sanitized_channel_title, source_lang)
        os.makedirs(channel_transcript_dir, exist_ok=True)

        if is_youtube:
            # channel_id usually exists on yt feeds; log it to verify
            channel_id = (
                youtube_channel_id_from_feed_url(feed_url)
                or (getattr(feed.feed, "yt_channelid", None) if str(getattr(feed.feed, "yt_channelid", "")).startswith("UC") else None)
                or (getattr(feed.feed, "channelid", None) if str(getattr(feed.feed, "channelid", "")).startswith("UC") else None)
            )

            logging.info(f"YouTube channel_id resolved: {channel_id} (feed_url={feed_url})")


            video_urls = get_youtube_channel_video_urls(
                channel_id,
                limit=limit_per_feed if (limit_per_feed and limit_per_feed > 0) else None
            )
            logging.info(f"yt-dlp uploads playlist returned {len(video_urls)} video urls")

            # Make "entry-like" dicts for your existing process_entry()
            entries_to_process = [{
                "title": "",     # process_entry will fetch real title via yt-dlp_info
                "link": url,
                "id": url,
                "description": None,
                "published": None,
                "enclosures": [{"url": url}],
            } for url in video_urls]
        else:
            # Normal RSS handling for podcasts
            if limit_per_feed and limit_per_feed > 0:
                entries_to_process = entries[:limit_per_feed]
            else:
                entries_to_process = entries


        logging.info(f"Will process {len(entries_to_process)} entries for this feed.")

        for entry_idx, entry in enumerate(entries_to_process, start=1):
            if _deadline_hit():
                logging.warning(
                    f"⏰ Time budget exhausted mid-feed on entry {entry_idx}/{len(entries_to_process)}; aborting pass."
                )
                full_pass = False
                break

            if simulate_entry_secs > 0:
                time.sleep(simulate_entry_secs)

            logging.debug(f"Starting processing for entry: {entry.get('title', 'No Title')}")
            result, diarization_pipeline = process_entry(
                entry, channel_transcript_dir, download_dir, channel_title,
                diarization_pipeline, config, feed_data, channel_summary, channel_author
            )

        if not full_pass:
            break

    failed_dir = DATABASE_ROOT / "watcher_json"
    failed_dir.mkdir(parents=True, exist_ok=True)
    failed_file = failed_dir / "failed_feeds.json"
    with open(failed_file, 'w', encoding='utf-8') as f:
        json.dump(failed_feeds, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(failed_feeds)} failed feeds to {failed_file}")

    # Only run translate when we actually finished the whole pass
    if full_pass:
        logging.info("All feeds processed. Running translation script...")
        try:
            subprocess.run(["python", "translate.py"], check=True)
            logging.info("Translation script completed successfully.")
        except Exception as e:
            logging.error(f"Translation script failed: {e}")
    else:
        logging.info("Partial pass; skipping translation this cycle.")

    return full_pass

# ── 3b) New main(...) with hard deadline + fixed cadence ────────────────────
def main(feeds_filename='master_rss.json', limit_per_feed=0, interval_seconds=None,
         simulate_feed_secs=0.0, simulate_entry_secs=0.0):
    """
    Run once (default), or repeat from the top every `interval_seconds`.
    Each cycle has a hard time budget: once the deadline hits, the pass aborts
    and the next cycle restarts from the top of the feeds file.
    """
    # One pass (current behavior)
    if not interval_seconds or interval_seconds <= 0:
        _ = run_cycle(feeds_filename, limit_per_feed,
                      deadline_mono=None,
                      simulate_feed_secs=simulate_feed_secs,
                      simulate_entry_secs=simulate_entry_secs)
        return

    cycle = 1
    next_tick = time.monotonic()  # schedule-based cadence
    while True:
        start_wall = datetime.now().isoformat(timespec='seconds')
        start_mono = time.monotonic()
        deadline_mono = start_mono + interval_seconds

        logging.info(f"=== Cycle {cycle} start @ {start_wall} (budget={interval_seconds/3600:.2f}h) ===")

        try:
            full_pass = run_cycle(
                feeds_filename,
                limit_per_feed,
                deadline_mono=deadline_mono,
                simulate_feed_secs=simulate_feed_secs,
                simulate_entry_secs=simulate_entry_secs
            )
        except Exception:
            logging.exception("Cycle crashed; continuing to the next scheduled run.")
            full_pass = False

        # Keep fixed cadence relative to prior scheduled tick
        next_tick += interval_seconds
        now = time.monotonic()
        delay = max(0.0, next_tick - now)
        elapsed = now - start_mono
        next_wall = (datetime.now() + timedelta(seconds=delay)).isoformat(timespec='seconds')

        logging.info(
            f"=== Cycle {cycle} complete in {elapsed/3600:.2f}h (full_pass={full_pass}). "
            f"Next run in {delay/3600:.2f}h @ {next_wall} ==="
        )

        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            logging.info("Received Ctrl+C; exiting cleanly.")
            break

        cycle += 1


if __name__ == "__main__":
    try:
        args = parse_cli_args()

        feeds_filename = (
            args.feeds_file
            or ("master_rss_mini.json" if args.master_rss_mini else "master_rss.json")
        )
        limit = args.limit or 0  # 0/None = ALL

        # Resolve restart interval (seconds has priority)
        if args.restart_seconds is not None and args.restart_hours is not None:
            logging.warning("Both --restart-hours and --restart-seconds provided; using seconds.")

        if args.restart_seconds is not None:
            interval_seconds = float(args.restart_seconds)
        elif args.restart_hours is not None:
            interval_seconds = float(args.restart_hours) * 3600.0
        else:
            interval_seconds = None

        logging.info(
            f"CLI parsed → feeds={feeds_filename}, limit={limit}, "
            f"interval_seconds={interval_seconds}, simulate_feed_secs={args.simulate_feed_secs}, "
            f"simulate_entry_secs={args.simulate_entry_secs}, migrate_legacy={args.migrate_legacy}"
        )

        main(
            feeds_filename=feeds_filename,
            limit_per_feed=limit,
            interval_seconds=interval_seconds,
            simulate_feed_secs=args.simulate_feed_secs,
            simulate_entry_secs=args.simulate_entry_secs
        )

        # ✅ ONLY run migration when requested
        if args.migrate_legacy:
            stats = migrate_legacy_transcripts(
                root=DATABASE_ROOT / "transcripts",
                max_len=args.migrate_max_len,
                dry_run=args.migrate_dry_run,
                size_limit_mb=args.migrate_size_limit_mb,
                report_name=args.migrate_report_name,
            )
            logging.info(f"Legacy migration stats: {stats}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user; exiting.")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
