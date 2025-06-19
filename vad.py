# vad.py
import webrtcvad
from pydub import AudioSegment

def get_vad_segments(wav_file_path, frame_duration=30):
    """
    Perform Voice Activity Detection on the WAV file.

    Parameters:
    - wav_file_path: str, path to the WAV file.
    - frame_duration: int, duration of each frame in ms (10, 20, or 30).

    Returns:
    - segments: list of tuples, each containing (start_time, end_time) in seconds.
    """
    audio = AudioSegment.from_wav(wav_file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Whisper expects 16kHz, mono audio
    raw_audio = audio.raw_data
    sample_rate = audio.frame_rate

    vad = webrtcvad.Vad(2)  # Aggressiveness mode (0-3)

    n_bytes_per_frame = int(sample_rate * (frame_duration / 1000) * 2)  # 16-bit audio
    total_frames = len(raw_audio) // n_bytes_per_frame

    segments = []
    is_speech = False
    start_time = 0

    for i in range(total_frames):
        start = i * n_bytes_per_frame
        end = start + n_bytes_per_frame
        frame = raw_audio[start:end]
        timestamp = (i * frame_duration) / 1000.0

        speech = vad.is_speech(frame, sample_rate)

        if speech and not is_speech:
            is_speech = True
            start_time = timestamp
        elif not speech and is_speech:
            is_speech = False
            end_time = timestamp
            segments.append((start_time, end_time))

    if is_speech:
        segments.append((start_time, len(raw_audio) / (sample_rate * 2)))

    return segments
