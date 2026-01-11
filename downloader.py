# downloader.py
import os
import time
import logging
import urllib.parse
from pathlib import Path

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils import sanitize_filename

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
        return feed
    except Exception as e:
        logging.error(f"Error parsing feed {feed_url}: {e}")
        return None


def _make_session() -> requests.Session:
    """
    A tuned Session with retries for transient errors.
    Note: many hosts return 403 for bot protection; we do not rely on retry to fix 403,
    but having a session + headers is what fixes most cases.
    """
    s = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


_SESSION = _make_session()


def _headers_for_url(audio_url: str) -> dict:
    """
    Many podcast CDNs (including Acast/Sphinx) will 403 default python clients.
    These headers make the request look like a normal browser fetch.
    """
    host = urllib.parse.urlparse(audio_url).netloc.lower()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) "
            "Gecko/20100101 Firefox/146.0"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    # Hotlink/WAF mitigations for common podcast CDNs
    if "acast.com" in host:
        headers["Referer"] = "https://play.acast.com/"
        headers["Origin"] = "https://play.acast.com"

    # Often helps CDNs serve the file without extra checks
    headers["Range"] = "bytes=0-"

    return headers


def download_audio(entry, download_dir=DATABASE_ROOT / "podcasts", filename=None, retries=3, backoff=5):
    """
    Download the audio file from the podcast entry with retry logic.
    Returns the filename on success, or None on failure.
    """
    if "enclosures" not in entry or len(entry.enclosures) == 0:
        logging.warning(f"No audio URL found for entry: {entry.get('title', 'Unknown Title')}")
        return None

    audio_url = entry.enclosures[0].href
    logging.info(f"Audio URL found: {audio_url}")

    # Build filename
    if filename is None:
        raw_title = entry.get("title", "Unknown Title")
        sanitized_title = sanitize_filename(raw_title)

        url_parts = urllib.parse.urlparse(audio_url)
        file_extension = os.path.splitext(url_parts.path)[1] or ".mp3"
        filename = f"{sanitized_title}{file_extension}"

    # Normalize download_dir to str path
    download_dir = str(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    filepath = os.path.abspath(os.path.join(download_dir, filename))
    partpath = filepath + ".part"

    logging.debug(f"Download directory: {download_dir}")
    logging.debug(f"Filename: {filename}")
    logging.debug(f"Filepath: {filepath}")
    logging.debug(f"Current working directory: {os.getcwd()}")

    # If final file exists and is non-trivial size, skip
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logging.info(f"File already exists: {filepath}")
        return filename

    # If a previous partial exists, remove it (fresh attempt)
    if os.path.exists(partpath):
        try:
            os.remove(partpath)
        except Exception:
            pass

    headers = _headers_for_url(audio_url)

    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Attempt {attempt} to download audio.")

            # Use a more realistic timeout: (connect, read)
            resp = _SESSION.get(
                audio_url,
                stream=True,
                headers=headers,
                timeout=(10, 300),
                allow_redirects=True,
            )

            # If forbidden, log useful diagnostics
            if resp.status_code == 403:
                preview = ""
                try:
                    preview = (resp.text or "")[:250]
                except Exception:
                    preview = "<unable to decode body>"
                logging.error(f"403 Forbidden downloading {audio_url}")
                logging.error(f"403 response preview: {preview!r}")
                raise requests.HTTPError("403 Forbidden", response=resp)

            resp.raise_for_status()

            # Write to temp file first
            with open(partpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            # Atomically replace
            os.replace(partpath, filepath)

            logging.info(f"Downloaded: {filepath}")
            return filename

        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt} failed to download {audio_url}: {e}")

            # Clean up partial
            if os.path.exists(partpath):
                try:
                    os.remove(partpath)
                except Exception:
                    pass

            if attempt < retries:
                logging.info(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                logging.error(f"All {retries} attempts failed for {audio_url}.")
                return None

    return None
