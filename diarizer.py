# diarizer.py
import os
from pyannote.audio import Pipeline

def perform_diarization(audio_file_path, hf_token):
    """
    Perform speaker diarization on the given audio file.

    Parameters:
    - audio_file_path: str, path to the audio file.
    - hf_token: str, Hugging Face access token.

    Returns:
    - diarization: pyannote.core.Annotation, the diarization result.
    """
    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return None

    # Perform speaker diarization
    print("Performing speaker diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token, embedding_batch_size=64, device="cuda")
    diarization = pipeline(audio_file_path)

    return diarization
