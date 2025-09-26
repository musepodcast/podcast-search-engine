# transcriber.py

import os
import time
import json
import re
import shutil
import subprocess
import logging

import torch
import whisper

from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook  # progress bar for diarization

# --------------------------------------------------------------------------------------
# Environment & threading setup
# --------------------------------------------------------------------------------------

# Load environment variables from .env file (for PYANNOTE_AUTH_TOKEN, etc.)
load_dotenv()

# Only synchronize CUDA for deep debugging when explicitly requested
if os.getenv("CUDA_DEBUG_SYNC", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Prevent CPU oversubscription on large boxes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass  # not critical if unavailable

# --------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------

# Optional local cache (extend if you want to cache more things)
_embedding_cache = {}

# Global cache for the Whisper model so it loads only once
WHISPER_MODEL = None


def get_whisper_model(device: str = "cuda", model_name: str = "turbo"):
    """
    Load Whisper once and reuse for all chunks.
    """
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            WHISPER_MODEL = whisper.load_model(model_name, device=device)
            logging.info(f"Loaded Whisper model: {model_name} on device {device}")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}", exc_info=True)
            WHISPER_MODEL = None
    return WHISPER_MODEL


# --------------------------------------------------------------------------------------
# Audio validation (lightweight; no full decode)
# --------------------------------------------------------------------------------------

def validate_audio(file_path: str) -> bool:
    """
    Quick integrity/stream check via ffprobe (no full decode).
    Returns True if ffprobe can parse the audio stream.
    Falls back to whisper.load_audio() if ffprobe is not installed.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Audio file not found for validation: {file_path}")
            return False

        if shutil.which("ffprobe") is None:
            logging.warning("ffprobe not found; falling back to whisper.load_audio for validation.")
            _ = whisper.load_audio(file_path)  # this will fully decode, but on 5-min chunks it's OK
            return True

        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,channels,sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True

    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        logging.error(f"ffprobe failed for {file_path}: {msg}")
        return False
    except Exception as e:
        logging.error(f"Audio validation failed for {file_path}: {e}", exc_info=True)
        return False


# --------------------------------------------------------------------------------------
# Diarization
# --------------------------------------------------------------------------------------

def initialize_diarization_pipeline(
    segmentation_batch_size: int = 2,
    embedding_batch_size: int = 4,
    segmentation_step: float = 0.1,
    min_speakers: int | None = None,
    max_speakers: int | None = None
):
    """
    Initialize the Pyannote speaker diarization pipeline and move it to GPU if available.
    """
    try:
        token = os.getenv("PYANNOTE_AUTH_TOKEN")
        if not token:
            logging.critical("Hugging Face token not found. Set PYANNOTE_AUTH_TOKEN in your .env.")
            return None

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

        # Move pipeline to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        logging.info(f"Speaker diarization pipeline initialized on {device}.")

        # Tweak segmentation step and batch sizes when available
        if hasattr(pipeline, "_segmentation") and hasattr(pipeline._segmentation, "duration"):
            old_step = getattr(pipeline._segmentation, "step", None)
            pipeline._segmentation.step = segmentation_step * pipeline._segmentation.duration
            logging.info(f"Segmentation step set to {pipeline._segmentation.step} (was {old_step})")
            try:
                pipeline._segmentation.batch_size = segmentation_batch_size
            except Exception:
                pass

        # Some versions expose embedding batch size differently; set if present
        if hasattr(pipeline, "embedding_batch_size"):
            pipeline.embedding_batch_size = embedding_batch_size
            logging.info(f"Set embedding_batch_size to {embedding_batch_size}")

        # Stash speaker constraints for use in perform_speaker_diarization
        pipeline.config = {}
        if min_speakers is not None:
            pipeline.config["min_speakers"] = min_speakers
        if max_speakers is not None:
            pipeline.config["max_speakers"] = max_speakers

        return pipeline

    except Exception as e:
        logging.critical(f"Failed to initialize diarization pipeline: {e}", exc_info=True)
        return None


def perform_speaker_diarization(pipeline, audio_file_path: str):
    """
    Run diarization on a file path. Respects min/max speakers if set in pipeline.config.
    """
    if not pipeline:
        logging.error("Diarization pipeline is not initialized. Skipping.")
        return None

    try:
        min_spk = pipeline.config.get("min_speakers", None) if hasattr(pipeline, "config") else None
        max_spk = pipeline.config.get("max_speakers", None) if hasattr(pipeline, "config") else None

        with ProgressHook() as hook:
            diarization = pipeline(
                {"uri": "audio", "audio": audio_file_path},
                min_speakers=min_spk,
                max_speakers=max_spk,
                hook=hook
            )

        logging.info("Speaker diarization completed.")
        return diarization

    except Exception as e:
        logging.error(f"Diarization failed: {e}", exc_info=True)
        return None


def assign_speaker(diarization, segment_start: float, segment_end: float, speaker_mapping: dict):
    """
    Determine which speaker label overlaps the most with [segment_start, segment_end].
    """
    if not diarization:
        return "Unknown"

    max_overlap = 0.0
    assigned_speaker = "Unknown"

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(float(turn.start), float(segment_start))
        overlap_end = min(float(turn.end), float(segment_end))
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > max_overlap:
            max_overlap = overlap
            assigned_speaker = speaker

    return speaker_mapping.get(assigned_speaker, "Unknown") if max_overlap > 0 else "Unknown"


# --------------------------------------------------------------------------------------
# Transcription + diarization orchestrator
# --------------------------------------------------------------------------------------

def transcribe_and_diarize(audio_file_path: str, pipeline=None):
    """
    Transcribe an audio file with Whisper and assign speakers via Pyannote diarization.
    """
    if not os.path.exists(audio_file_path):
        logging.error(f"Audio file not found: {audio_file_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Whisper model selection
    model_name = "turbo"   # Change if you prefer base/small/large-v3, etc.
    model = get_whisper_model(device=device, model_name=model_name)
    if model is None:
        return None

    if not validate_audio(audio_file_path):
        return None

    # ----- Transcription -----
    logging.info(f"Starting transcription for: {audio_file_path}")
    t0 = time.time()
    with torch.inference_mode():
        result = model.transcribe(
            audio_file_path,
            verbose=False,
            temperature=0.0,                   # deterministic decoding
            beam_size=5,                       # small beam for stability without huge cost
            fp16=torch.cuda.is_available(),    # half precision on GPU
            condition_on_previous_text=False,  # treat each chunk independently
            # language="en",                   # set if known; speeds up language ID
            # task="transcribe",              # or "translate"
        )
        if device == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    logging.info(f"Transcription took {t1 - t0:.2f} seconds")

    full_text = (result.get("text") or "").strip()
    segments = result.get("segments", [])

    # ----- Diarization -----
    logging.info(f"Starting speaker diarization for: {audio_file_path}")
    t0 = time.time()
    diarization = perform_speaker_diarization(pipeline, audio_file_path)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    logging.info(f"Speaker diarization took {t1 - t0:.2f} seconds")

    if not diarization:
        logging.error("Diarization returned None. Skipping speaker assignment.")
        return None

    # Build speaker mapping (e.g., {'SPEAKER_00': 'Speaker 1', ...})
    unique_speakers = sorted({spkr for _, _, spkr in diarization.itertracks(yield_label=True)})
    speaker_mapping = {spkr: f"Speaker {i + 1}" for i, spkr in enumerate(unique_speakers)}
    logging.info(f"Speaker mapping: {speaker_mapping}")

    # Apply speaker labels to Whisper segments
    enriched_segments = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        assigned = assign_speaker(diarization, start, end, speaker_mapping)
        enriched_segments.append(
            {"start": start, "end": end, "text": text, "speaker": assigned}
        )

    # Optional: free some VRAM heap between chunks (your crash was system RAM, but this helps VRAM)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "transcript": full_text,
        "segments": enriched_segments,
    }


# --------------------------------------------------------------------------------------
# CLI for single-file testing
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and diarize a single podcast episode.")
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Full path to the audio file (e.g., 'C:/path/to/chunk1.wav')."
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=None,
        help="If known, force a minimum speaker count."
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="If known, force a maximum speaker count."
    )
    parser.add_argument(
        "--segmentation_batch_size",
        type=int,
        default=2,
        help="Batch size for segmentation model (default=2)."
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=4,
        help="Batch size for embedding extraction (default=4)."
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Overlap ratio for segmentation windows (default=0.1 => 90% overlap)."
    )

    args = parser.parse_args()

    diarization_pipeline = initialize_diarization_pipeline(
        segmentation_batch_size=args.segmentation_batch_size,
        embedding_batch_size=args.embedding_batch_size,
        segmentation_step=args.segmentation_step,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    if not diarization_pipeline:
        logging.critical("Diarization pipeline initialization failed. Exiting.")
        raise SystemExit(1)

    data = transcribe_and_diarize(
        audio_file_path=args.audio_file,
        pipeline=diarization_pipeline
    )
    if data:
        logging.info(f"Transcription + diarization completed for {args.audio_file}")
        out_path = os.path.splitext(args.audio_file)[0] + "_transcription.json"
        try:
            with open(out_path, "w", encoding="utf-8") as outf:
                json.dump(data, outf, indent=4, ensure_ascii=False)
            logging.info(f"Saved results to {out_path}")
        except Exception as e:
            logging.error(f"Failed to save output: {e}", exc_info=True)
    else:
        logging.error("Failed to process audio file.")
