# utils.py
import re
import logging

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

