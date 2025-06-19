import os
from pydub import AudioSegment

def convert_mp3_to_wav(source_dir, target_dir):
    """
    Convert all MP3 files in the source_dir to WAV format and save them in target_dir.

    Parameters:
    - source_dir (str): Path to the directory containing MP3 files.
    - target_dir (str): Path to the directory where WAV files will be saved.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.mp3'):
            mp3_path = os.path.join(source_dir, filename)
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(target_dir, wav_filename)

            try:
                print(f"Converting {mp3_path} to {wav_path}...")
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format='wav')
                print(f"Successfully converted to {wav_path}")
            except Exception as e:
                print(f"Failed to convert {mp3_path}: {e}")

if __name__ == "__main__":
    source_directory = 'podcasts'  # Source directory containing MP3 files
    target_directory = 'podcasts_wav'  # Target directory for WAV files
    convert_mp3_to_wav(source_directory, target_directory)
