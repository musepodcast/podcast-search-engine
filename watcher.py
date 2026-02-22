# watcher.py: Polling watcher that only inserts episodes once chapters are present, with full translation support

import time
import json
from django.db import close_old_connections
import sys
import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from zoneinfo import ZoneInfo
import psycopg2
from bs4 import BeautifulSoup
from dateutil import parser as duparser
from utils import sanitize_filename
from datetime import timezone as dt_timezone

# watcher.py lives here:
BASE = Path(__file__).parent        
# sibling folder where all of your artifacts now live:
DATABASE_ROOT = BASE.parent / "podcast_data"
WEBAPP = os.path.join(BASE, 'podcast_web_app')          
sys.path.insert(0, WEBAPP)

# tell Django which settings to use
#os.environ.setdefault('DJANGO_SETTINGS_MODULE',
#                      'podcast_project.settings.base')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'podcast_project.settings.production')
# now we can bootstrap Django
import django
django.setup()

# finally our models import will work
from podcasts.models import Episode, Transcript
from podcasts.search.documents import TranscriptDocument
import warnings
from elastic_transport import SecurityWarning
from urllib3.exceptions import InsecureRequestWarning
from django.utils.deprecation import RemovedInDjango60Warning
import urllib3

SITE_TZ = ZoneInfo("America/Chicago")

SKIP_JSON_FILENAMES = {
    "processed_index.json",
    "last_seen.json",
    "failed_index.json",
    "failed_feeds.json",
}


## Silence ES’s SecurityWarning about TLS+verify_certs=False
warnings.filterwarnings("ignore", category=SecurityWarning)
# And also silence urllib3’s InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ignore the naive‐datetime warning
warnings.filterwarnings(
    "ignore",
    r"DateTimeField .+ received a naive datetime",
    RuntimeWarning,
)

# ignore the insecure‐request warning
#urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
# where your incoming JSONs really are:
ROOT_DIR         = DATABASE_ROOT / "transcripts"
# where you want to stash your state & failure files:
STATE_FILE       = DATABASE_ROOT / "watcher_json" / "last_seen.json"
FAILED_INDEX_FILE= DATABASE_ROOT / "watcher_json" / "failed_index.json"
# where you want your rolling log:
LOG_FILE         = DATABASE_ROOT / "log" / "watcher.log"

POLL_INTERVAL = 60                    # seconds between scans

DB_USER = "postgres"
DB_NAME = "podcast_db"
DB_PASSWORD = "root"
DB_HOST = "localhost"
DB_PORT = "5432"

# ensure the folders exist before you start writing into them:
for d in (DATABASE_ROOT / "watcher_json", DATABASE_ROOT / "log"):
    d.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# SETUP LOGGING (human-readable timestamps)
# ----------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(
    str(LOG_FILE), maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(console_handler)

# ----------------------------------------------------------------------------
# STATE MANAGEMENT
# ----------------------------------------------------------------------------
def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
            return data.get('last_seen', 0.0)
        except Exception:
            logger.warning("Failed to parse state file; starting from epoch 0.")
    return 0.0


def save_state(ts):
    STATE_FILE.write_text(
        json.dumps({"last_seen": ts}), encoding='utf-8'
    )

def load_failed_index():
    if FAILED_INDEX_FILE.exists():
        try:
            return set(json.loads(FAILED_INDEX_FILE.read_text()))
        except:
            return set()
    return set()

def save_failed_index(failures):
    FAILED_INDEX_FILE.write_text(json.dumps(sorted(failures)), encoding="utf-8")


# ----------------------------------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------------------------------
def connect_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        logger.info("Connected to database.")
        return conn
    except Exception as e:
        logger.critical(f"DB connection failed: {e}")
        raise

# ----------------------------------------------------------------------------
# HELPERS: HTML cleaning, parsing, normalization
# ----------------------------------------------------------------------------
def clean_html(raw_html):
    if not raw_html or not isinstance(raw_html, str):
        return 'No Description Available'
    if os.path.exists(raw_html):
        logger.warning(f"Description looks like a file path: {raw_html}")
        return 'No Description Available'
    try:
        soup = BeautifulSoup(raw_html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logger.warning(f"clean_html error: {e}")
        return 'No Description Available'


def safe_parse_int(val):
    try:
        return int(val)
    except Exception:
        return None


def normalize_pub_date(raw: str):
    if not raw:
        return None
    dt = duparser.parse(raw)
    if dt.tzinfo is None:
        # choose what naive means for your data; here we treat it as Central wall time
        dt = dt.replace(tzinfo=SITE_TZ)
    return dt.astimezone(dt_timezone.utc)


def normalize_duration(dur):
    try:
        if isinstance(dur, int):
            h = dur // 3600; m = (dur % 3600) // 60; s = dur % 60
        else:
            parts = list(map(int, dur.split(':')))
            parts = ([0] * (3 - len(parts))) + parts
            h, m, s = parts
        return f"{h:02}:{m:02}:{s:02}"
    except Exception as e:
        logger.warning(f"normalize_duration error: {e}")
        return None


def seconds_to_hms(sec):
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02}:{m:02}:{s:02}"

# ----------------------------------------------------------------------------
# VALIDATION & EXISTENCE CHECKS
# ----------------------------------------------------------------------------
def validate_json(data):
    if 'metadata' not in data or 'transcript' not in data:
        logger.warning("JSON missing 'metadata' or 'transcript'; skipping.")
        return False
    return True

def get_master_episode_and_channel(conn, guid):
    """
    Return a tuple (master_episode_id, channel_id) for the given GUID,
    or (None, None) if not found.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, channel_id FROM Episodes WHERE guid = %s;",
            (guid,)
        )
        row = cur.fetchone()
        return row if row else (None, None)

def master_exists(conn, guid):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM Episodes WHERE guid = %s;", (guid,))
        return cur.fetchone() is not None

# ----------------------------------------------------------------------------
# DB INSERTION FUNCTIONS
# ----------------------------------------------------------------------------
def insert_channel(conn, title, sanitized, image_url=None, author=None, summary=None):
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO Channels(
                   channel_title, sanitized_channel_title,
                   channel_image_url, channel_author, channel_summary)
               VALUES(%s,%s,%s,%s,%s)
               ON CONFLICT(channel_title) DO NOTHING
               RETURNING id;''',
            (title, sanitized, image_url, author, summary)
        )
        res = cur.fetchone()
        if not res:
            cur.execute("SELECT id FROM Channels WHERE channel_title=%s;", (title,))
            res = cur.fetchone()
        conn.commit()
        return res[0]


def insert_episode(conn, channel_id, meta):
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO Episodes(
                   channel_id, episode_title, sanitized_episode_title, publication_date,
                   duration, episode_number, explicit, guid, audio_url, image_url,
                   description, categories, language, link, transcript, translated, tsv_transcript
               ) VALUES(
                   %(channel_id)s, %(episode_title)s, %(sanitized_episode_title)s, %(publication_date)s,
                   %(duration)s, %(episode_number)s, %(explicit)s, %(guid)s, %(audio_url)s, %(image_url)s,
                   %(description)s, %(categories)s, %(language)s, %(link)s, %(transcript)s, %(translated)s,
                   to_tsvector('english', COALESCE(%(description)s,'') || ' ' || COALESCE(%(transcript)s,''))
               ) ON CONFLICT(channel_id, episode_title) DO NOTHING
               RETURNING id;''',
            meta
        )
        res = cur.fetchone()
        if res:
            eid, inserted = res[0], True
        else:
            cur.execute(
                "SELECT id FROM Episodes WHERE channel_id=%s AND episode_title=%s;",
                (meta['channel_id'], meta['episode_title'])
            )
            eid = cur.fetchone()[0]
            inserted = False
        conn.commit()
        return eid, inserted


def insert_chapters(conn, episode_id, chapters):
    if not chapters:
        logger.warning(f"No chapters yet for ep_id {episode_id}; will retry next scan.")
        return False
    with conn.cursor() as cur:
        rows = [(episode_id, ch.get('title'), ch.get('time'))
                for ch in chapters if ch.get('title') and ch.get('time')]
        if rows:
            cur.executemany(
                'INSERT INTO Chapters(episode_id,chapter_title,chapter_time) VALUES(%s,%s,%s);',
                rows
            )
        conn.commit()
        return True


def insert_transcript(conn, episode_id, segments):
    with conn.cursor() as cur:
        rows = []
        for seg in segments:
            st, en = seg.get('start'), seg.get('end')
            if st is None or en is None: continue
            t = f"{seconds_to_hms(st)} - {seconds_to_hms(en)}"
            rows.append((episode_id, t, seg.get('text',''), seg.get('speaker','Unknown')))
        if rows:
            cur.executemany(
                'INSERT INTO Transcripts(episode_id,segment_time,segment_text,speaker) VALUES(%s,%s,%s,%s);',
                rows
            )
        conn.commit()

# Translation inserts

def insert_channel_translation(conn, channel_id, meta):
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO ChannelsTranslations(
                   channel_id, channel_title, sanitized_channel_title, channel_image_url,
                   channel_author, channel_summary, language, translated)
               VALUES(%(channel_id)s,%(channel_title)s,%(sanitized_channel_title)s,
                     %(channel_image_url)s,%(channel_author)s,%(channel_summary)s,
                     %(language)s,%(translated)s)
               ON CONFLICT(channel_id,language) DO UPDATE SET
                     channel_title=EXCLUDED.channel_title,
                     sanitized_channel_title=EXCLUDED.sanitized_channel_title,
                     channel_image_url=EXCLUDED.channel_image_url,
                     channel_author=EXCLUDED.channel_author,
                     channel_summary=EXCLUDED.channel_summary,
                     translated=EXCLUDED.translated
               RETURNING id;''',
            meta
        )
        res = cur.fetchone()
        conn.commit()
        return res[0] if res else None

def insert_episode_translation(conn, episode_id, meta):
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO EpisodesTranslations(
                   episode_id, episode_title, sanitized_episode_title, publication_date,
                   duration, episode_number, explicit, guid, audio_url, image_url,
                   description, categories, language, link, transcript, translated, tsv_transcript
               ) VALUES(
                   %(episode_id)s,%(episode_title)s,%(sanitized_episode_title)s,%(publication_date)s,
                   %(duration)s,%(episode_number)s,%(explicit)s,%(guid)s,%(audio_url)s,%(image_url)s,
                   %(description)s,%(categories)s,%(language)s,%(link)s,%(transcript)s,%(translated)s,
                   to_tsvector('english', COALESCE(%(description)s,'')||' '||COALESCE(%(transcript)s,''))
               ) ON CONFLICT(episode_id,language) DO UPDATE SET
                   episode_title=EXCLUDED.episode_title,
                   sanitized_episode_title=EXCLUDED.sanitized_episode_title,
                   publication_date=EXCLUDED.publication_date,
                   duration=EXCLUDED.duration,
                   episode_number=EXCLUDED.episode_number,
                   explicit=EXCLUDED.explicit,
                   guid=EXCLUDED.guid,
                   audio_url=EXCLUDED.audio_url,
                   image_url=EXCLUDED.image_url,
                   description=EXCLUDED.description,
                   categories=EXCLUDED.categories,
                   link=EXCLUDED.link,
                   transcript=EXCLUDED.transcript,
                   translated=EXCLUDED.translated,
                   tsv_transcript=EXCLUDED.tsv_transcript
               RETURNING id;''',
            meta
        )
        res = cur.fetchone()
        conn.commit()
        return res[0] if res else None

def insert_chapters_translation(conn, eptr_id, chapters, language):
    with conn.cursor() as cur:
        rows = []
        for ch in chapters:
            t = ch.get('title'); tm = ch.get('time')
            if t and tm:
                rows.append((eptr_id, t, tm, language))
        if rows:
            cur.executemany(
                '''
                INSERT INTO ChaptersTranslations(episodetranslations_id, chapter_title, chapter_time, language)
                VALUES(%s,%s,%s,%s)
                ON CONFLICT(episodetranslations_id, chapter_title, language) DO NOTHING;
                ''',
                rows
            )
        conn.commit()

def insert_transcript_translation(conn, eptr_id, segments, language):
    with conn.cursor() as cur:
        rows = []
        for seg in segments:
            st, en = seg.get('start'), seg.get('end')
            if st is None or en is None:
                continue
            t = f"{seconds_to_hms(st)} - {seconds_to_hms(en)}"
            rows.append((eptr_id, t, seg.get('text',''), seg.get('speaker','Unknown'), language))
        if rows:
            cur.executemany(
                '''
                INSERT INTO TranscriptsTranslations(episodetranslations_id, segment_time, segment_text, speaker, language)
                VALUES(%s,%s,%s,%s,%s)
                ON CONFLICT(episodetranslations_id, segment_time, language) DO NOTHING;
                ''',
                rows
            )
        conn.commit()

def index_episode_transcripts_in_es(episode_id: int):
    """
    Ensure ALL Transcript rows for a given episode_id are indexed into ES 'transcripts'.

    This fixes the core issue:
    - psycopg2 inserts bypass Django signals
    - so TranscriptDocument will not update unless we do it explicitly.

    Safe to call repeatedly.
    """
    close_old_connections()

    qs = (
        Transcript.objects
        .filter(episode_id=episode_id)
        .only("id", "episode_id", "segment_text", "segment_time")
    )

    TranscriptDocument().update(qs.iterator(chunk_size=5000), refresh=False)

    # Optional sanity logging (can be removed once stable)
    try:
        db_count = Transcript.objects.filter(episode_id=episode_id).count()
        es_count = (
            TranscriptDocument.search()
            .query("term", episode_id=episode_id)
            .extra(size=0, track_total_hits=True)
            .execute()
            .hits.total.value
        )
        logger.info(f"Transcript count DB={db_count} ES={es_count} for ep_id={episode_id}")
    except Exception as e:
        logger.warning(f"Transcript count sanity check failed for ep_id={episode_id}: {e}")




# ----------------------------------------------------------------------------
# PROCESS A SINGLE FILE
# ----------------------------------------------------------------------------
def process_one_file(conn, json_path: Path):
    logger.info(f"Processing file: {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error(f"JSON load error {json_path}: {e}")
        return

    if not validate_json(data):
        return

    meta = data['metadata']
    guid = meta.get('guid')
    if not guid:
        logger.warning(f"Missing GUID; skipping {json_path}")
        return

    chapters = data.get('chapters', [])
    if not chapters:
        logger.warning(f"Chapters missing/empty for {json_path}")
        logger.info("Waiting for chapters; will retry next scan.")
        return

    segments = data.get('segments', [])
    is_translated = meta.get('translated', False)

    if not is_translated:
        # MASTER EPISODE INSERT
        if master_exists(conn, guid):
            logger.info(f"Master exists for GUID {guid}; skipping master insert.")
            return

        ch_title = meta.get('channel_title')
        if not ch_title:
            logger.warning(f"Missing channel_title; skipping {json_path}")
            return
        ch_id = insert_channel(conn,
                               ch_title,
                               sanitize_filename(ch_title),
                               meta.get('channel_image_url'),
                               meta.get('author'),
                               clean_html(meta.get('summary')))

        ep_meta = {
            'channel_id': ch_id,
            'episode_title': meta.get('episode_title'),
            'sanitized_episode_title': sanitize_filename(meta.get('episode_title') or ''),
            'publication_date': normalize_pub_date(meta.get('publication_date')),
            'duration': normalize_duration(meta.get('duration')),
            'episode_number': safe_parse_int(meta.get('episode_number')),
            'explicit': meta.get('explicit'),
            'guid': guid,
            'audio_url': meta.get('audio_url'),
            'image_url': meta.get('image_url'),
            'description': clean_html(meta.get('description', '')),
            'categories': json.dumps(meta.get('categories')) if meta.get('categories') else None,
            'language': meta.get('language'),
            'link': meta.get('link'),
            'transcript': data.get('transcript', ''),
            'translated': False
        }
        ep_id, inserted = insert_episode(conn, ch_id, ep_meta)
        if inserted:
            insert_chapters(conn, ep_id, chapters)
            insert_transcript(conn, ep_id, segments)
            logger.info(f"Inserted master episode: {ep_meta['episode_title']}")

            # TRIGGER ES INDEXING (Episode + Transcript)
            try:
                close_old_connections()

                # Index the episode document (episodes index)
                ep = Episode.objects.get(pk=ep_id)
                ep.save()

                # ✅ Index all transcripts for this episode (transcripts index)
                index_episode_transcripts_in_es(ep_id)

                logger.info(f"Indexed episode + transcripts in Elasticsearch (id={ep_id})")

                # On success, clear from any prior failures
                failed = load_failed_index()
                if ep_id in failed:
                    failed.remove(ep_id)
                    save_failed_index(failed)

            except Exception as e:
                logger.error(f"Failed to index episode/transcripts id={ep_id}: {e}")

                # record it for retry later
                failed = load_failed_index()
                failed.add(ep_id)
                save_failed_index(failed)

        else:
            logger.info(f"Skipped existing master episode: {ep_meta['episode_title']}")

    else:
        # TRANSLATION BRANCH
        master_id, channel_db_id = get_master_episode_and_channel(conn, guid)
        if not master_id:
            logger.warning(f"Master not found for GUID {guid}; skipping translation {json_path}")
            return

        # Channel-level translation
        ch_meta = {
            'channel_id': channel_db_id,
            'channel_title': meta.get('channel_title'),
            'sanitized_channel_title': sanitize_filename(meta.get('channel_title') or ''),
            'channel_image_url': meta.get('channel_image_url'),
            'channel_author': meta.get('author'),
            'channel_summary': clean_html(meta.get('summary')),
            'language': meta.get('language'),
            'translated': True
        }
        insert_channel_translation(conn, channel_db_id, ch_meta)

        # Episode-level translation
        ep_meta = {
            'episode_id': master_id,
            'episode_title': meta.get('episode_title'),
            'sanitized_episode_title': sanitize_filename(meta.get('episode_title') or ''),
            'publication_date': normalize_pub_date(meta.get('publication_date')),
            'duration': normalize_duration(meta.get('duration')),
            'episode_number': safe_parse_int(meta.get('episode_number')),
            'explicit': meta.get('explicit'),
            'guid': guid,
            'audio_url': meta.get('audio_url'),
            'image_url': meta.get('image_url'),
            'description': clean_html(meta.get('description', '')),
            'categories': json.dumps(meta.get('categories')) if meta.get('categories') else None,
            'language': meta.get('language'),
            'link': meta.get('link'),
            'transcript': data.get('transcript', ''),
            'translated': True
        }
        eptr_id = insert_episode_translation(conn, master_id, ep_meta)

        # Chapters & transcript translations
        insert_chapters_translation(conn, eptr_id, chapters, meta.get('language'))
        insert_transcript_translation(conn, eptr_id, segments, meta.get('language'))

        logger.info(f"Inserted translation for episode: {ep_meta['episode_title']}")
        # RE-INDEX MASTER EPISODE WITH TRANSLATION
        try:
            ep = Episode.objects.get(id=master_id)
            ep.save()
            logger.info(f"Re-indexed episode translation for id={master_id}")
        except Exception as e:
            logger.error(f"Failed to re-index translation for episode id={master_id}: {e}")



# ----------------------------------------------------------------------------
# SCAN & MAIN LOOP
# ----------------------------------------------------------------------------
def scan_and_process(conn, last_seen):
    files = [
        p for p in ROOT_DIR.rglob("*.json")
        if p.name not in SKIP_JSON_FILENAMES
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    max_seen = last_seen
    for f in files:
        m = f.stat().st_mtime
        if m <= last_seen:
            break
        process_one_file(conn, f)
        max_seen = max(max_seen, m)
    return max_seen


if __name__ == '__main__':
    conn = connect_db()
    failed = load_failed_index()
    if failed:
        logger.info(f"Retrying ES indexing for {len(failed)} episodes...")
        for pk in list(failed):
            try:
                close_old_connections()

                ep = Episode.objects.get(pk=pk)
                ep.save()  # episodes index

                # ✅ transcripts index
                index_episode_transcripts_in_es(pk)

                failed.remove(pk)
                logger.info(f"✅ Re-indexed episode + transcripts #{pk}")

            except Exception as e:
                logger.warning(f"❌ Still failing to index #{pk}: {e}")

        save_failed_index(failed)
    last_seen = load_state()
    human = datetime.fromtimestamp(last_seen).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Starting watcher; last seen: {human}")
    try:
        while True:
            last_seen = scan_and_process(conn, last_seen)
            save_state(last_seen)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed.")
