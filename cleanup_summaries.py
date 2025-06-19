#!/usr/bin/env python3
import json
from pathlib import Path
from bs4 import BeautifulSoup

# —— adjust this to point at wherever your JSONs live
TRANSCRIPTS_DIR = Path("transcripts")

def clean_html(raw_html: str) -> str:
    """
    Strip all tags, collapse whitespace, return plain text.
    """
    soup = BeautifulSoup(raw_html or "", "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    # collapse runs of whitespace
    return " ".join(text.split())

def process_file(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    summary = meta.get("summary", "")
    cleaned = clean_html(summary)
    # only rewrite if it actually changed
    if cleaned != summary:
        meta["summary"] = cleaned
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✔ cleaned {path.relative_to(TRANSCRIPTS_DIR)}")
    else:
        print(f"  – no change: {path.relative_to(TRANSCRIPTS_DIR)}")

def main():
    print(f"Scanning {TRANSCRIPTS_DIR} for JSON files…")
    for fn in TRANSCRIPTS_DIR.rglob("*.json"):
        try:
            process_file(fn)
        except Exception as e:
            print(f"✘ failed on {fn}: {e}")

if __name__ == "__main__":
    main()
