# utils.py
import re
import hashlib

def sanitize_filename(filename):
    """
    Sanitize the filename by removing or replacing invalid characters.

    Parameters:
    - filename: str, the original filename.

    Returns:
    - str, the sanitized filename.
    """
    # Remove any characters that are not alphanumeric, spaces, underscores, or hyphens
    sanitized = re.sub(r'[^\w\s-]', '', filename).strip()
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized




def make_episode_stem_with_suffix(raw_title: str, suffix: str, max_len: int = 100) -> str:
    """
    Build a filename stem like:
        <truncated_sanitized_title>__<suffix>

    - Title is sanitized and truncated to max_len chars (only the title part).
    - Suffix is always appended, so identity is stable even if title changes.
    """
    from utils import sanitize_filename  # if sanitize_filename is in same module, remove this import and call directly

    title = sanitize_filename(raw_title or "Unknown Title")

    # Normalize underscores a bit
    title = re.sub(r"_+", "_", title).strip("_").strip()

    # Truncate title part only (keep suffix intact)
    if max_len and len(title) > max_len:
        title = title[:max_len].rstrip("_").rstrip()

    # Final stem
    return f"{title}__{suffix}"



def stable_id_suffix(vid: str | None = None, guid: str | None = None, link: str | None = None) -> str:
    """
    Returns a stable identifier for filenames.
    - YouTube: yt_<vid>
    - RSS/other: g_<sha1(guid)[:12]>
    """
    if vid:
        return f"yt_{vid}"

    base = guid or link or ""
    if not base:
        # absolute last resort; caller should avoid this
        return "g_unknown"

    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"g_{h}"