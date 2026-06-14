# downloader.py
import os
import time
import logging
import urllib.parse
from pathlib import Path

import re
import shutil
import subprocess
import sys
import json

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils import sanitize_filename

BASE          = Path(__file__).parent
DATABASE_ROOT = BASE.parent / "podcast_data"

_YT_CHANNEL_RE = re.compile(r"^https?://(www\.)?youtube\.com/channel/(UC[0-9A-Za-z_-]+)", re.I)
_YT_HANDLE_RE  = re.compile(r"^https?://(www\.)?youtube\.com/@([0-9A-Za-z_.-]+)", re.I)

def normalize_feed_url(url: str) -> str:
    """
    Accept:
      - youtube channel page: https://www.youtube.com/channel/UC...
      - youtube handle page:  https://www.youtube.com/@handle
      - youtube feed:         https://www.youtube.com/feeds/videos.xml?channel_id=UC...
      - normal RSS feeds:     unchanged

    Note: @handle → channel_id requires a lookup (see note below). We only auto-rewrite /channel/UC... reliably.
    """
    if not url:
        return url

    m = _YT_CHANNEL_RE.match(url.strip())
    if m:
        channel_id = m.group(2)
        return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

    # Handle URLs can’t be reliably converted without fetching/looking up the channel id.
    # Keep as-is so you notice it and replace with /feeds/videos.xml?channel_id=...
    return url

def yt_dlp_info(url: str) -> dict | None:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-J", url]

    kwargs = {"capture_output": True, "text": True}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # no console window

    p = subprocess.run(cmd, **kwargs)
    if p.returncode != 0:
        logging.error(f"(YouTube) yt-dlp -J failed rc={p.returncode}")
        if p.stderr:
            logging.error(p.stderr[:2000])
        return None

    try:
        return json.loads(p.stdout)
    except Exception as e:
        logging.error(f"(YouTube) Failed to parse yt-dlp JSON: {e}")
        return None

def parse_feed(feed_url):
    """
    Parse the RSS feed and return the full feed object.
    """
    logging.info("Parsing RSS feed...")
    feed_url = normalize_feed_url(feed_url)
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

NON_RETRYABLE_HTTP_STATUS = {400, 401, 403, 404, 410, 451}


def _headers_for_url(audio_url: str) -> dict:
    """
    Many podcast CDNs (including Acast/Sphinx) will 403 default python clients.
    These headers make the request look like a normal browser fetch.
    """
    parsed = urllib.parse.urlparse(audio_url)
    host = parsed.netloc.lower()
    url_l = (audio_url or "").lower()
    is_spreaker = "spreaker.com" in host or "spreaker.com" in url_l
    is_podtrac = "podtrac.com" in host or "podtrac.com" in url_l

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "audio/*,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    # Hotlink/WAF mitigations for common podcast CDNs
    if "acast.com" in host:
        headers["Referer"] = "https://play.acast.com/"
        headers["Origin"] = "https://play.acast.com"

    if is_spreaker or is_podtrac:
        headers["Referer"] = "https://www.spreaker.com/"
        headers["Origin"] = "https://www.spreaker.com"
    else:
        # Often helps CDNs serve the file without extra checks, but Spreaker/Podtrac
        # can reject scripted range requests even when a normal browser download works.
        headers["Range"] = "bytes=0-"

    return headers


def _retry_without_range_on_forbidden(resp, headers: dict, audio_url: str):
    if resp.status_code != 403 or "Range" not in headers:
        return resp

    retry_headers = dict(headers)
    retry_headers.pop("Range", None)
    logging.info(f"Retrying HTTP 403 once without Range header: {audio_url}")
    return _SESSION.get(
        audio_url,
        stream=True,
        headers=retry_headers,
        timeout=(10, 300),
        allow_redirects=True,
    )


# ----------------------------- YouTube Support -----------------------------

_YT_FEED_RE = re.compile(r"youtube\.com/feeds/videos\.xml", re.I)
_YT_URL_RE  = re.compile(r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)", re.I)

def _is_youtube_feed_url(feed_url: str) -> bool:
    return bool(feed_url and _YT_FEED_RE.search(feed_url))

def _is_youtube_video_url(url: str) -> bool:
    return bool(url and _YT_URL_RE.search(url))

def _is_youtube_shorts_url(url: str) -> bool:
    return bool(url and re.search(r"youtube\.com/shorts/", url, re.I))

def _extract_youtube_video_url(entry) -> str:
    """
    feedparser entries usually expose .link and also entry.get('link').
    """
    try:
        return getattr(entry, "link", None) or entry.get("link")
    except Exception:
        return None

def _ensure_yt_dlp_available():
    exe = shutil.which("yt-dlp") or shutil.which("yt_dlp")
    if exe:
        return exe
    return None  # we can still try `python -m yt_dlp`

def _download_youtube_audio(entry, download_dir, filename, ffmpeg_bin=None) -> str | None:
    """
    Download YouTube audio as MP3 using yt-dlp.
    Returns filename on success, None on failure.
    """
    video_url = _extract_youtube_video_url(entry)

    if not _is_youtube_video_url(video_url):
        logging.warning(f"No YouTube video link found for entry: {entry.get('title', 'Unknown Title')}")
        return None

    # ✅ skip Shorts
    if _is_youtube_shorts_url(video_url):
        logging.info(f"(YouTube) Skipping SHORTS entry: {video_url}")
        return None

    download_dir = str(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    filepath = os.path.abspath(os.path.join(download_dir, filename))

    # Skip if exists
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logging.info(f"(YouTube) File already exists: {filepath}")
        return filename

    # Build a clean yt-dlp output template WITHOUT double extensions
    # Example: C:\...\Propaganda_collapses...%(ext)s
    base_no_ext = os.path.splitext(filepath)[0]
    out_tmpl = base_no_ext + ".%(ext)s"

    yt_exe = _ensure_yt_dlp_available()

    # Prefer yt-dlp executable if present, else use current interpreter -m yt_dlp
    if yt_exe:
        base_cmd = [yt_exe]
    else:
        base_cmd = [sys.executable, "-m", "yt_dlp"]

    cmd = base_cmd + [
        "--no-playlist",
        "-f", "bestaudio/best",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-progress",
        "-o", out_tmpl,
        video_url,
    ]

    # If you have ffmpeg path, tell yt-dlp exactly where it is
    if ffmpeg_bin:
        cmd += ["--ffmpeg-location", str(Path(ffmpeg_bin).parent)]

    kwargs = {
        "check": False,             # we'll handle non-zero ourselves to log stderr
        "capture_output": True,     # ✅ capture stdout/stderr
        "text": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    logging.info(f"(YouTube) Running: {' '.join(cmd[:6])} ... (url hidden in log)")
    try:
        proc = subprocess.run(cmd, **kwargs)

        if proc.returncode != 0:
            logging.error(f"(YouTube) yt-dlp exit code {proc.returncode}")
            if proc.stdout:
                logging.error(f"(YouTube) yt-dlp stdout:\n{proc.stdout.strip()[:2000]}")
            if proc.stderr:
                logging.error(f"(YouTube) yt-dlp stderr:\n{proc.stderr.strip()[:2000]}")
            return None

        produced_mp3 = base_no_ext + ".mp3"
        if not os.path.exists(produced_mp3) or os.path.getsize(produced_mp3) == 0:
            # If yt-dlp produced a different ext, show what exists
            candidates = list(Path(download_dir).glob(Path(base_no_ext).name + ".*"))
            logging.error(f"(YouTube) Expected MP3 not found: {produced_mp3}")
            logging.error(f"(YouTube) Candidates: {[str(c) for c in candidates][:20]}")
            return None

        # Ensure final filename is exactly what main.py expects (filename)
        if produced_mp3.lower() != filepath.lower():
            os.replace(produced_mp3, filepath)

        logging.info(f"(YouTube) Downloaded: {filepath}")
        return filename

    except Exception as e:
        logging.error(f"(YouTube) yt-dlp invocation crashed: {e}", exc_info=True)
        return None



# ----------------------------- Main Downloader -----------------------------

def download_audio(entry, download_dir=DATABASE_ROOT / "podcasts", filename=None, retries=3, backoff=5, ffmpeg_bin=None):
    """
    Download the audio file from a podcast entry with retry logic.

    Supports:
      - Standard podcast RSS items with enclosure audio URLs
      - YouTube channel RSS items (no enclosure) by downloading audio via yt-dlp
        (Shorts are skipped)

    Returns the filename on success, or None on failure.
    """
    # Build filename once (works for both RSS and YouTube)
    if filename is None:
        raw_title = entry.get("title", "Unknown Title")
        sanitized_title = sanitize_filename(raw_title)
        filename = f"{sanitized_title}.mp3"

    # --- YouTube branch: no enclosures, but has entry.link ---
    if ("enclosures" not in entry) or (len(getattr(entry, "enclosures", [])) == 0):
        yt_url = _extract_youtube_video_url(entry)
        if _is_youtube_video_url(yt_url):
            # ✅ skip Shorts early
            if _is_youtube_shorts_url(yt_url):
                logging.info(f"(YouTube) Skipping SHORTS entry: {yt_url}")
                return None
            return _download_youtube_audio(entry, download_dir, filename, ffmpeg_bin=ffmpeg_bin)

        logging.warning(f"No audio URL found for entry: {entry.get('title', 'Unknown Title')}")
        return None

    # --- Standard RSS enclosure branch ---
    audio_url = entry.enclosures[0].href
    logging.info(f"Audio URL found: {audio_url}")

    # If caller didn’t provide filename, preserve your original extension behavior
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
            resp = _retry_without_range_on_forbidden(resp, headers, audio_url)

            # Permanent/client-side failures are not fixed by retrying the same URL.
            if resp.status_code in NON_RETRYABLE_HTTP_STATUS:
                preview = ""
                try:
                    preview = (resp.text or "")[:250]
                except Exception:
                    preview = "<unable to decode body>"
                logging.error(f"Non-retryable HTTP {resp.status_code} downloading {audio_url}")
                logging.error(f"HTTP {resp.status_code} response preview: {preview!r}")
                return None

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
