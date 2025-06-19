# downloader.py
import os
import requests
import feedparser
import urllib.parse
from utils import sanitize_filename
import logging
import time
from pathlib import Path

BASE          = Path(__file__).parent
DATABASE_ROOT = BASE.parent / "podcast_data"

def parse_feed(feed_url):
    """
    Parse the RSS feed and return the full feed object.
    """
    logging.info("Parsing RSS feed...")
    try:
        feed = feedparser.parse(feed_url)
        if not feed.entries:
            logging.warning(f"No entries found in feed: {feed_url}")
        logging.info("RSS feed parsed successfully.")
        return feed  # Return the entire feed object
    except Exception as e:
        logging.error(f"Error parsing feed {feed_url}: {e}")
        return None

def download_audio(entry, download_dir=DATABASE_ROOT / "podcasts", filename=None, retries=3, backoff=5):
    """
    Download the audio file from the podcast entry with retry logic.
    """
    if 'enclosures' in entry and len(entry.enclosures) > 0:
        audio_url = entry.enclosures[0].href
        logging.info(f"Audio URL found: {audio_url}")

        if filename is None:
            # Generate the filename from the entry title
            raw_title = entry.get('title', 'Unknown Title')
            sanitized_title = sanitize_filename(raw_title)
            # Parse the URL to get the path and extract the file extension
            url_parts = urllib.parse.urlparse(audio_url)
            path = url_parts.path
            file_extension = os.path.splitext(path)[1] or '.mp3'
            # Build the filename
            filename = f"{sanitized_title}{file_extension}"
        else:
            # Use the provided filename (already sanitized in main.py)
            filename = filename

        # Combine with download directory
        filepath = os.path.join(download_dir, filename)
        # Convert to absolute path
        filepath = os.path.abspath(filepath)

        # Log debug information
        logging.debug(f"Download directory: {download_dir}")
        logging.debug(f"Filename: {filename}")
        logging.debug(f"Filepath: {filepath}")
        logging.debug(f"Current working directory: {os.getcwd()}")

        # Create directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Skip if file already exists
        if os.path.exists(filepath):
            logging.info(f"File already exists: {filepath}")
            return filename  # Return the filename

        # Download the file with retries
        for attempt in range(1, retries + 1):
            try:
                logging.info(f"Attempt {attempt} to download audio.")
                response = requests.get(audio_url, stream=True, timeout=10)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"Downloaded: {filepath}")
                return filename  # Return the filename
            except requests.exceptions.RequestException as e:
                logging.error(f"Attempt {attempt} failed to download {audio_url}: {e}")
                if attempt < retries:
                    logging.info(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                else:
                    logging.error(f"All {retries} attempts failed for {audio_url}.")
                    return None
    else:
        logging.warning(f"No audio URL found for entry: {entry.get('title', 'Unknown Title')}")
        return None
