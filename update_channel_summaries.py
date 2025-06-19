#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path

import psycopg2
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# CONFIG: adjust these for your environment
# ----------------------------------------------------------------------------
TRANSCRIPTS_DIR = Path("transcripts")
DB_NAME   = "podcast_db"
DB_USER   = "postgres"
DB_PASS   = "root"
DB_HOST   = "localhost"
DB_PORT   = "5432"

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def clean_html(raw_html: str) -> str:
    """
    Strip all HTML tags, collapse whitespace, return plain text.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        # collapse any runs of whitespace into single spaces
        return " ".join(text.split())
    except Exception:
        return ""

def connect_db():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )

# ----------------------------------------------------------------------------
# MAIN UPDATE LOOP
# ----------------------------------------------------------------------------
def main():
    conn = connect_db()
    cur  = conn.cursor()
    updated = 0

    for path in TRANSCRIPTS_DIR.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[!] Skipping {path.name}: JSON parse error: {e}")
            continue

        # ensure metadata is a dict, never None
        meta = data.get("metadata") or {}
        if not isinstance(meta, dict):
            print(f"[!] Skipping {path.name}: no metadata")
            continue

        title = meta.get("channel_title")
        if not title:
            print(f"[!] Skipping {path.name}: missing channel_title")
            continue

        # clean out HTML from summary
        summary_raw = meta.get("summary", "")
        summary = clean_html(summary_raw)

        # sanitized slug to match your table
        # e.g. "My Podcast Show" -> "My_Podcast_Show"
        sanitized = meta.get("sanitized_channel_title")
        if not sanitized:
            # fallback if not in JSON
            sanitized = title.replace(" ", "_")

        # update master channel
        cur.execute(
            """
            UPDATE Channels
               SET channel_summary = %s
             WHERE sanitized_channel_title = %s
            """,
            (summary, sanitized)
        )

        # if this JSON is a translation, also update the translations table
        lang = (meta.get("language") or "").lower()
        if lang and lang != "en":
            cur.execute(
                """
                UPDATE ChannelsTranslations
                   SET channel_summary = %s
                 WHERE sanitized_channel_title = %s
                   AND language = %s
                """,
                (summary, sanitized, lang)
            )

        if cur.rowcount:
            updated += 1
            print(f"→ {path.name}: updated summary for '{sanitized}' ({lang or 'master'})")
        else:
            print(f"→ {path.name}: no matching row for '{sanitized}' ({lang or 'master'})")

        # commit after each file to avoid long transactions
        conn.commit()
        # small sleep to be polite to the DB if you have thousands of files
        time.sleep(0.01)

    print(f"\nDone. {updated} channel(s) updated.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
