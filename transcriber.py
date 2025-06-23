import os
import time
import whisper
import torch
from utils import sanitize_filename
import logging
import json
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from pyannote.audio.pipelines.utils.hook import ProgressHook  # Import ProgressHook

# Load environment variables from .env file
load_dotenv()

# Set the CUDA_LAUNCH_BLOCKING environment variable
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Global cache for embeddings
_embedding_cache = {}

# Global cache for the Whisper model
WHISPER_MODEL = None

def get_whisper_model(device="cuda", model_name="turbo"):
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            WHISPER_MODEL = whisper.load_model(model_name, device=device)
            logging.info(f"Loaded Whisper model: {model_name} on device {device}")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}", exc_info=True)
            WHISPER_MODEL = None
    return WHISPER_MODEL


def cached_get_embeddings(pipeline_instance, file, *args, **kwargs):
    """
    Wrapper that caches the embeddings to avoid re-computation.
    """
    
    step = kwargs.get('step', 0.5)
    key = f"{file['audio']}_step{step}"

    if key in _embedding_cache:
        logging.info(f"Using cached embeddings for key: {key}")
        return _embedding_cache[key]

    t0 = time.time()
    embeddings = pipeline_instance._original_get_embeddings(file, *args, **kwargs)
    t1 = time.time()
    logging.info(f"Computed embeddings for key: {key} in {t1 - t0:.2f} seconds")

    _embedding_cache[key] = embeddings
    return embeddings

def initialize_diarization_pipeline(
    segmentation_batch_size=2,
    embedding_batch_size=4,
    segmentation_step=0.1,
    min_speakers=None,
    max_speakers=None
):
    """
    Initialize the PyAnnote.Audio speaker diarization pipeline and move it to GPU if available.

    Parameters
    ----------
    segmentation_batch_size : int
        How many audio segments to batch together for the segmentation model.
    embedding_batch_size : int
        How many waveforms to batch for speaker embeddings.
    segmentation_step : float
        Overlap ratio for segmentation. 0.1 => 90% overlap; 0.3 => 70% overlap, etc.
        Reducing overlap can reduce compute cost but might degrade accuracy.
    min_speakers : int or None
        If you have a minimum known speaker count, pass it here.
    max_speakers : int or None
        If you have a maximum known speaker count, pass it here.

    Returns
    -------
    pipeline : pyannote.audio Pipeline
    """
    try:
        token = os.getenv("PYANNOTE_AUTH_TOKEN")
        if not token:
            logging.critical("Hugging Face token not found. Please set PYANNOTE_AUTH_TOKEN in your .env file.")
            return None

        # Load the standard speaker-diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        logging.info(f"Speaker diarization pipeline initialized on {device}.")

        # Expose advanced parameters:
        # The pipeline can be manipulated to set batch sizes + segmentation step
        if hasattr(pipeline, 'segmentation'):
            # segmentation_step is a ratio of overlap
            # For example, pipeline._segmentation.step = segmentation_step * pipeline._segmentation.duration
            # but we can also do it via pipeline._segmentation
            if hasattr(pipeline._segmentation, 'step'):
                old_step = pipeline._segmentation.step
                # The pipeline default step might be 0.1 * duration
                # We can override ratio by adjusting pipeline._segmentation.step
                pipeline._segmentation.step = segmentation_step * pipeline._segmentation.duration
                logging.info(f"Changed segmentation_step ratio from {old_step} to {pipeline._segmentation.step}")

            # batch sizes for the segmentation model
            pipeline._segmentation.batch_size = segmentation_batch_size

        # The pipeline has a "clustering" that uses embeddings. We can set the embedding batch size:
        if hasattr(pipeline, 'embedding_batch_size'):
            pipeline.embedding_batch_size = embedding_batch_size
            logging.info(f"Set embedding_batch_size to {embedding_batch_size}")

        # If the pipeline supports forced min/max speakers (some do):
        #   This depends on the pipeline's usage of "apply(..., min_speakers=..., max_speakers=...)".
        #   We'll store these in pipeline.config for convenience, so we can pass them later.
        pipeline.config = {}
        if min_speakers is not None:
            pipeline.config['min_speakers'] = min_speakers
        if max_speakers is not None:
            pipeline.config['max_speakers'] = max_speakers

        # Override get_embeddings with cache
#        if not hasattr(pipeline, '_original_get_embeddings'):
#            pipeline._original_get_embeddings = pipeline.get_embeddings
#            pipeline.get_embeddings = lambda file, *args, **kwargs: cached_get_embeddings(
#                pipeline, file, *args, **kwargs
#            )

        return pipeline

    except Exception as e:
        logging.critical(f"Failed to initialize diarization pipeline: {e}", exc_info=True)
        return None

def perform_speaker_diarization(pipeline, audio_file_path):
    """
    Perform speaker diarization on the given audio file, possibly with min_speakers or max_speakers.

    Returns
    -------
    diarization : pyannote.core.Annotation
    """
    if not pipeline:
        logging.error("Diarization pipeline is not initialized. Skipping.")
        return None

    try:
       
        # If we stored min_speakers / max_speakers in pipeline.config, pass them:
        min_spk = pipeline.config.get('min_speakers', None) if hasattr(pipeline, 'config') else None
        max_spk = pipeline.config.get('max_speakers', None) if hasattr(pipeline, 'config') else None

        # The pipeline’s apply method supports "min_speakers" and "max_speakers" optional arguments
        with ProgressHook() as hook:
            diarization = pipeline(
                {
                    "uri": "audio",
                    "audio": audio_file_path
                },
                min_speakers=min_spk,
                max_speakers=max_spk,
                hook=hook
            )

        logging.info("Speaker diarization completed.")
        return diarization

    except Exception as e:
        logging.error(f"Diarization failed: {e}", exc_info=True)
        return None

def assign_speaker(diarization, segment_start, segment_end, speaker_mapping):
    """
    Determine which speaker label overlaps the most with [segment_start, segment_end].
    """
    if not diarization:
        return "Unknown"

    max_overlap = 0.0
    assigned_speaker = "Unknown"

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(turn.start, segment_start)
        overlap_end = min(turn.end, segment_end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > max_overlap:
            max_overlap = overlap
            assigned_speaker = speaker

    return speaker_mapping.get(assigned_speaker, "Unknown") if max_overlap > 0 else "Unknown"

def validate_audio(file_path):
    """
    Validate if the audio file is playable and not corrupted.
    """
    try:
        _ = whisper.load_audio(file_path)
        return True
    except Exception as e:
        logging.error(f"Audio validation failed for {file_path}: {e}", exc_info=True)
        return False

def transcribe_and_diarize(audio_file_path, pipeline=None):
    if not os.path.exists(audio_file_path):
        logging.error(f"Audio file not found: {audio_file_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    model_name = "turbo"  # Use Whisper 'turbo' for speed

    # Instead of loading the model every time, get the cached model
    model = get_whisper_model(device=device, model_name=model_name)
    if model is None:
        return None

    if not validate_audio(audio_file_path):
        return None

    logging.info(f"Starting transcription for: {audio_file_path}")
    t0 = time.time()
    result = model.transcribe(audio_file_path, verbose=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    logging.info(f"Transcription took {t1 - t0:.2f} seconds")

    english_text = result.get("text", "").strip()
    segments = result.get("segments", [])

    # Now do diarization
    logging.info(f"Starting speaker diarization for: {audio_file_path}")
    t0 = time.time()
    diarization = perform_speaker_diarization(pipeline, audio_file_path)
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    logging.info(f"Speaker diarization took {t1 - t0:.2f} seconds")

    if not diarization:
        logging.error("Diarization returned None. Skipping speaker assignment.")
        return None

    # Build speaker mapping
    unique_speakers = sorted(set([spkr for _, _, spkr in diarization.itertracks(yield_label=True)]))
    speaker_mapping = {spkr: f"Speaker {i+1}" for i, spkr in enumerate(unique_speakers)}
    logging.info(f"Speaker mapping: {speaker_mapping}")

    # Assign each transcribed segment to a speaker
    enriched_segments = []
    for seg in segments:
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        text = seg.get('text', "").strip()
        assigned = assign_speaker(diarization, start, end, speaker_mapping)

        enriched_segments.append({
            'start': start,
            'end': end,
            'text': text,
            'speaker': assigned
        })

    return {
        'transcript': english_text,
        'segments': enriched_segments
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe and diarize a single podcast episode.")
    parser.add_argument('--audio_file', type=str, required=True,
                        help="Full path to the audio file (e.g., 'C:/path/to/chunk1.wav').")
    parser.add_argument('--min_speakers', type=int, default=None,
                        help="If known, force a minimum speaker count.")
    parser.add_argument('--max_speakers', type=int, default=None,
                        help="If known, force a maximum speaker count.")
    parser.add_argument('--segmentation_batch_size', type=int, default=2,
                        help="Batch size for segmentation model (default=2).")
    parser.add_argument('--embedding_batch_size', type=int, default=4,
                        help="Batch size for embedding extraction (default=4).")
    parser.add_argument('--segmentation_step', type=float, default=0.1,
                        help="Overlap ratio for segmentation windows (default=0.1 => 90% overlap).")

    args = parser.parse_args()

    # Initialize pipeline with user-provided parameters
    diarization_pipeline = initialize_diarization_pipeline(
        segmentation_batch_size=args.segmentation_batch_size,
        embedding_batch_size=args.embedding_batch_size,
        segmentation_step=args.segmentation_step,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    if not diarization_pipeline:
        logging.critical("Diarization pipeline initialization failed. Exiting.")
        exit(1)

    data = transcribe_and_diarize(
        audio_file_path=args.audio_file,
        pipeline=diarization_pipeline
    )
    if data:
        logging.info(f"Transcription + diarization completed for {args.audio_file}")
        out_path = os.path.splitext(args.audio_file)[0] + "_transcription.json"
        try:
            with open(out_path, 'w', encoding='utf-8') as outf:
                json.dump(data, outf, indent=4, ensure_ascii=False)
            logging.info(f"Saved results to {out_path}")
        except Exception as e:
            logging.error(f"Failed to save output: {e}", exc_info=True)
    else:
        logging.error("Failed to process audio file.")
