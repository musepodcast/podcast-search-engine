import os
import json
import logging
from pathlib import Path
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup
from dateutil import parser
from datetime import datetime
from logging.handlers import RotatingFileHandler
from utils import sanitize_filename

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a rotating file handler
handler = RotatingFileHandler("populate_db.log", maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def clean_html(raw_html):
    if not raw_html or not isinstance(raw_html, str):
        return 'No Description Available'
    if os.path.exists(raw_html):
        logging.warning(f"Description field contains a filename or path: {raw_html}")
        return 'No Description Available'
    try:
        soup = BeautifulSoup(raw_html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logging.warning(f"Failed to clean HTML: {e}")
        return 'No Description Available'

def safe_parse_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_parse_date(date_str):
    try:
        return parser.isoparse(date_str)
    except (ValueError, TypeError):
        return None

def normalize_duration(duration):
    try:
        if isinstance(duration, int):
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        elif isinstance(duration, str):
            parts = duration.split(":")
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return f"{h:02}:{m:02}:{s:02}"
            elif len(parts) == 2:
                m, s = map(int, parts)
                return f"00:{m:02}:{s:02}"
            elif len(parts) == 1:
                s = int(parts[0])
                return f"00:00:{s:02}"
        return None
    except Exception as e:
        logging.warning(f"Failed to normalize duration: {e}")
        return None

def seconds_to_hms(seconds):
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"

def connect_db(dbname, user, password, host='localhost', port='5432'):
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        logging.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        raise

def insert_channel(conn, channel_title, sanitized_channel_title,
                   channel_image_url=None, channel_author=None, channel_summary=None):
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO Channels (
                    channel_title,
                    sanitized_channel_title,
                    channel_image_url,
                    channel_author,
                    channel_summary
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (channel_title) DO NOTHING
                RETURNING id;
            '''
            cursor.execute(insert_query, (
                channel_title,
                sanitized_channel_title,
                channel_image_url,
                channel_author,
                channel_summary,
            ))
            result = cursor.fetchone()
            if result:
                channel_id = result[0]
                logging.info(f"Inserted new channel: {channel_title} with ID {channel_id}")
            else:
                select_query = ''' SELECT id FROM Channels WHERE channel_title = %s; '''
                cursor.execute(select_query, (channel_title,))
                fetched = cursor.fetchone()
                if fetched:
                    channel_id = fetched[0]
                    logging.info(f"Channel already exists: {channel_title} with ID {channel_id}")
                else:
                    channel_id = None
                    logging.warning(f"Could not retrieve existing channel ID for: {channel_title}")
            conn.commit()
            return channel_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert/select channel '{channel_title}': {e}")
        raise

def insert_episode(conn, channel_id, metadata):
    """
    Insert a new episode into the Episodes table.
    Now includes a new column 'translated'.
    """
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO Episodes (
                    channel_id, episode_title, sanitized_episode_title, publication_date,
                    duration, episode_number, explicit, guid, audio_url,
                    image_url, description, categories, language, link,
                    transcript, translated, tsv_transcript
                ) VALUES (
                    %(channel_id)s, %(episode_title)s, %(sanitized_episode_title)s, %(publication_date)s,
                    %(duration)s, %(episode_number)s, %(explicit)s, %(guid)s, %(audio_url)s,
                    %(image_url)s, %(description)s, %(categories)s, %(language)s, %(link)s,
                    %(transcript)s, %(translated)s,
                    to_tsvector('english', COALESCE(%(description)s, '') || ' ' || COALESCE(%(transcript)s, ''))
                )
                ON CONFLICT (channel_id, episode_title) DO NOTHING
                RETURNING id;
            '''
            episode_values = {
                'channel_id': channel_id,
                'episode_title': metadata.get('episode_title') or None,
                'sanitized_episode_title': metadata.get('sanitized_episode_title') or None,
                'publication_date': metadata.get('publication_date') or None,
                'duration': normalize_duration(metadata.get('duration')),
                'episode_number': safe_parse_int(metadata.get('episode_number')),
                'explicit': metadata.get('explicit'),
                'guid': metadata.get('guid') or None,
                'audio_url': metadata.get('audio_url') or None,
                'image_url': metadata.get('image_url') or None,
                'description': metadata.get('description') or None,
                'categories': json.dumps(metadata.get('categories')) if metadata.get('categories') else None,
                'language': metadata.get('language') or None,
                'link': metadata.get('link') or None,
                'transcript': metadata.get('transcript') or None,
                'translated': metadata.get('translated') or False,  # Default to False if not set.
            }
            cursor.execute(insert_query, episode_values)
            result = cursor.fetchone()
            if result:
                episode_id = result[0]
                logging.info(f"Inserted new episode: {metadata.get('episode_title')} with ID {episode_id}")
                inserted = True
            else:
                select_query = '''
                    SELECT id FROM Episodes WHERE channel_id = %s AND episode_title = %s;
                '''
                cursor.execute(select_query, (channel_id, metadata.get('episode_title')))
                fetched = cursor.fetchone()
                if fetched:
                    episode_id = fetched[0]
                    logging.info(f"Episode already exists: {metadata.get('episode_title')} with ID {episode_id}")
                    inserted = False
                else:
                    episode_id = None
                    logging.warning(f"Could not retrieve existing episode ID for: {metadata.get('episode_title')}")
                    inserted = False
            conn.commit()
            return episode_id, inserted
    except psycopg2.errors.UniqueViolation as e:
        conn.rollback()
        logging.error(f"Unique constraint violated for episode '{metadata.get('episode_title')}': {e}")
        return None, False
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert/select episode '{metadata.get('episode_title')}': {e}")
        raise

def insert_chapters(conn, episode_id, chapters):
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO Chapters (episode_id, chapter_title, chapter_time)
                VALUES (%s, %s, %s);
            '''
            chapters_to_insert = []
            for chapter in chapters:
                title = chapter.get('title')
                time_str = chapter.get('time')
                if title and time_str:
                    chapters_to_insert.append((episode_id, title, time_str))
                else:
                    logging.warning(f"Incomplete chapter data for Episode ID {episode_id}: {chapter}")
            if chapters_to_insert:
                cursor.executemany(insert_query, chapters_to_insert)
                logging.info(f"Inserted {len(chapters_to_insert)} chapters for Episode ID {episode_id}")
                conn.commit()
                return len(chapters_to_insert)
            else:
                logging.warning(f"No valid chapters to insert for Episode ID {episode_id}")
                return 0
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert chapters for Episode ID {episode_id}: {e}")
        raise

def insert_transcript(conn, episode_id, segments):
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO Transcripts (episode_id, segment_time, segment_text, speaker)
                VALUES (%s, %s, %s, %s);
            '''
            segments_to_insert = []
            for segment in segments:
                start = segment.get('start')
                end = segment.get('end')
                text = segment.get('text', "")
                speaker = segment.get('speaker', "Unknown")
                if start is not None and end is not None:
                    segment_time = f"{seconds_to_hms(start)} - {seconds_to_hms(end)}"
                    segments_to_insert.append((episode_id, segment_time, text, speaker))
                else:
                    logging.warning(f"Invalid segment data for Episode ID {episode_id}: {segment}")
            if segments_to_insert:
                cursor.executemany(insert_query, segments_to_insert)
                logging.info(f"Inserted {len(segments_to_insert)} segments for Episode ID {episode_id}")
            else:
                logging.warning(f"No valid segments to insert for Episode ID {episode_id}")
            conn.commit()
            return len(segments_to_insert)
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert segments for Episode ID {episode_id}: {e}")
        raise

# ======================= TRANSLATION TABLE INSERTIONS ==========================
def insert_channel_translation(conn, channel_id, metadata):
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO ChannelsTranslations (
                    channel_id, channel_title, sanitized_channel_title, channel_image_url,
                    channel_author, channel_summary, language, translated
                ) VALUES (
                    %(channel_id)s, %(channel_title)s, %(sanitized_channel_title)s, %(channel_image_url)s,
                    %(channel_author)s, %(channel_summary)s, %(language)s, %(translated)s
                )
                ON CONFLICT (channel_id, language) DO UPDATE SET
                    channel_title = EXCLUDED.channel_title,
                    sanitized_channel_title = EXCLUDED.sanitized_channel_title,
                    channel_image_url = EXCLUDED.channel_image_url,
                    channel_author = EXCLUDED.channel_author,
                    channel_summary = EXCLUDED.channel_summary,
                    translated = EXCLUDED.translated
                RETURNING id;
            '''
            values = {
                'channel_id': channel_id,
                'channel_title': metadata.get('channel_title'),
                'sanitized_channel_title': metadata.get('sanitized_channel_title'),
                'channel_image_url': metadata.get('channel_image_url'),
                'channel_author': metadata.get('channel_author'),
                'channel_summary': metadata.get('channel_summary'),
                'language': metadata.get('language'),
                'translated': metadata.get('translated', True),
            }
            cursor.execute(insert_query, values)
            result = cursor.fetchone()
            if result:
                translation_id = result[0]
                logging.info(f"Inserted/Updated channel translation for channel ID {channel_id} with translation ID {translation_id}")
            else:
                translation_id = None
                logging.warning(f"Failed to insert/update channel translation for channel ID {channel_id}")
            conn.commit()
            return translation_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert/update channel translation for channel ID {channel_id}: {e}")
        raise

def insert_episode_translation(conn, episode_id, metadata):
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO EpisodesTranslations (
                    episode_id, episode_title, sanitized_episode_title, publication_date,
                    duration, episode_number, explicit, guid, audio_url,
                    image_url, description, categories, language, link,
                    transcript, translated, tsv_transcript
                ) VALUES (
                    %(episode_id)s, %(episode_title)s, %(sanitized_episode_title)s, %(publication_date)s,
                    %(duration)s, %(episode_number)s, %(explicit)s, %(guid)s, %(audio_url)s,
                    %(image_url)s, %(description)s, %(categories)s, %(language)s, %(link)s,
                    %(transcript)s, %(translated)s,
                    to_tsvector('english', COALESCE(%(description)s, '') || ' ' || COALESCE(%(transcript)s, ''))
                )
                ON CONFLICT (episode_id, language) DO UPDATE SET
                    episode_title = EXCLUDED.episode_title,
                    sanitized_episode_title = EXCLUDED.sanitized_episode_title,
                    publication_date = EXCLUDED.publication_date,
                    duration = EXCLUDED.duration,
                    episode_number = EXCLUDED.episode_number,
                    explicit = EXCLUDED.explicit,
                    guid = EXCLUDED.guid,
                    audio_url = EXCLUDED.audio_url,
                    image_url = EXCLUDED.image_url,
                    description = EXCLUDED.description,
                    categories = EXCLUDED.categories,
                    link = EXCLUDED.link,
                    transcript = EXCLUDED.transcript,
                    translated = EXCLUDED.translated,
                    tsv_transcript = EXCLUDED.tsv_transcript
                RETURNING id;
            '''
            values = {
                'episode_id': episode_id,
                'episode_title': metadata.get('episode_title'),
                'sanitized_episode_title': metadata.get('sanitized_episode_title'),
                'publication_date': metadata.get('publication_date'),
                'duration': normalize_duration(metadata.get('duration')),
                'episode_number': safe_parse_int(metadata.get('episode_number')),
                'explicit': metadata.get('explicit'),
                'guid': metadata.get('guid'),
                'audio_url': metadata.get('audio_url'),
                'image_url': metadata.get('image_url'),
                'description': metadata.get('description'),
                'categories': json.dumps(metadata.get('categories')) if metadata.get('categories') else None,
                'language': metadata.get('language'),
                'link': metadata.get('link'),
                'transcript': metadata.get('transcript'),
                'translated': metadata.get('translated') or True,
            }
            cursor.execute(insert_query, values)
            result = cursor.fetchone()
            if result:
                translation_id = result[0]
                logging.info(f"Inserted/Updated episode translation for episode ID {episode_id} with translation ID {translation_id}")
            else:
                translation_id = None
                logging.warning(f"Failed to insert/update episode translation for episode ID {episode_id}")
            conn.commit()
            return translation_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert/update episode translation for episode ID {episode_id}: {e}")
        raise

def insert_chapters_translation(conn, episode_id, chapters, language):
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO ChaptersTranslations (episode_id, chapter_title, chapter_time, language)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (episode_id, chapter_title, language) DO NOTHING;
            '''
            chapters_to_insert = []
            for chapter in chapters:
                title = chapter.get('title')
                time_str = chapter.get('time')
                if title and time_str:
                    chapters_to_insert.append((episode_id, title, time_str, language))
                else:
                    logging.warning(f"Incomplete chapter translation data for Episode ID {episode_id}: {chapter}")
            if chapters_to_insert:
                cursor.executemany(insert_query, chapters_to_insert)
                logging.info(f"Inserted {len(chapters_to_insert)} translated chapters for Episode ID {episode_id}")
                conn.commit()
                return len(chapters_to_insert)
            else:
                logging.warning(f"No valid translated chapters to insert for Episode ID {episode_id}")
                return 0
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert translated chapters for Episode ID {episode_id}: {e}")
        raise


def insert_transcript_translation(conn, episode_id, segments, language):
    try:
        with conn.cursor() as cursor:
            insert_query = '''
                INSERT INTO TranscriptsTranslations (episode_id, segment_time, segment_text, speaker, language)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (episode_id, segment_time, language) DO NOTHING;
            '''
            segments_to_insert = []
            for segment in segments:
                start = segment.get('start')
                end = segment.get('end')
                text = segment.get('text', "")
                speaker = segment.get('speaker', "Unknown")
                if start is not None and end is not None:
                    segment_time = f"{seconds_to_hms(start)} - {seconds_to_hms(end)}"
                    segments_to_insert.append((episode_id, segment_time, text, speaker, language))
                else:
                    logging.warning(f"Invalid translated segment data for Episode ID {episode_id}: {segment}")
            if segments_to_insert:
                cursor.executemany(insert_query, segments_to_insert)
                logging.info(f"Inserted {len(segments_to_insert)} translated segments for Episode ID {episode_id}")
            else:
                logging.warning(f"No valid translated segments to insert for Episode ID {episode_id}")
            conn.commit()
            return len(segments_to_insert)
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert translated segments for Episode ID {episode_id}: {e}")
        raise


def validate_json_structure(data):
    required_keys = ['metadata', 'transcript', 'chapters']
    for key in required_keys:
        if key not in data:
            logging.warning(f"Missing key '{key}' in JSON data.")
            return False
    return True

def master_episode_exists(conn, guid):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM Episodes WHERE guid = %s;", (guid,))
        return cursor.fetchone() is not None


def populate_database(conn, root_directory):
    root_path = Path(root_directory)
    if not root_path.exists() or not root_path.is_dir():
        logging.critical(f"Root directory '{root_directory}' does not exist or is not a directory.")
        return

    json_files = list(root_path.rglob("*.json"))
    # Sort files so that those in 'en' or 'en-us' directories come first.
    json_files = sorted(
        json_files, 
        key=lambda p: 0 if any(part.lower() in ("en", "en-us") for part in p.parts) else 1
    )
    logging.info(f"Found {len(json_files)} JSON files in '{root_directory}' and its subdirectories.")

    for json_file in json_files:
        logging.info(f"Processing file: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not validate_json_structure(data):
                logging.warning(f"Invalid JSON structure in file: {json_file}. Skipping.")
                continue

            metadata = data.get('metadata', {})
            if not metadata:
                logging.warning(f"Metadata missing in file: {json_file}")
                continue

            # Get essential metadata.
            guid = metadata.get('guid')
            if not guid:
                logging.warning(f"GUID missing in file: {json_file}. Skipping.")
                continue

            # (Optional) Skip file if master episode already exists.
            # Uncomment the following lines if you want to skip old episodes entirely:
            # if master_episode_exists(conn, guid):
            #     logging.info(f"Episode with GUID {guid} already exists. Skipping file: {json_file}")
            #     continue

            channel_title = metadata.get('channel_title')
            if not channel_title:
                logging.warning(f"Channel title missing in metadata for file: {json_file}")
                continue
            sanitized_channel_title = sanitize_filename(channel_title)
            channel_image_url = metadata.get('channel_image_url', None)
            channel_author = metadata.get('author', None)
            channel_summary = clean_html(metadata.get('summary', None))

            channel_id = insert_channel(conn, channel_title, sanitized_channel_title,
                                        channel_image_url=channel_image_url,
                                        channel_author=channel_author,
                                        channel_summary=channel_summary)
            if not channel_id:
                logging.warning(f"Skipping episode due to missing channel ID for channel '{channel_title}'.")
                continue

            episode_title = metadata.get('episode_title')
            if not episode_title:
                logging.warning(f"Episode title missing in metadata for file: {json_file}")
                continue
            sanitized_episode_title = sanitize_filename(episode_title)
            publication_date = safe_parse_date(metadata.get('publication_date'))
            duration = metadata.get('duration', '00:00:00')
            episode_number = safe_parse_int(metadata.get('episode_number'))
            explicit = metadata.get('explicit')
            audio_url = metadata.get('audio_url', '')
            image_url = metadata.get('image_url', '')
            description = clean_html(metadata.get('description', 'No Description Available'))
            categories = metadata.get('categories') or []
            if isinstance(categories, str):
                categories = [categories.strip('{}').replace('""', '').replace('"', '')]
            elif isinstance(categories, dict):
                categories = list(categories.values())
            elif isinstance(categories, list):
                categories = [str(cat).strip() for cat in categories]
            else:
                categories = []
            language = metadata.get('language') or 'Unknown'
            link = metadata.get('link') or ''
            transcript = data.get('transcript', '')

            # Get the translated flag; default to False.
            translated = metadata.get('translated', False)

            essential_fields = ['episode_title', 'guid']
            missing_fields = [field for field in ['episode_title', 'guid'] if not metadata.get(field)]
            if not transcript.strip():
                missing_fields.append('transcript')
            if missing_fields:
                logging.warning(f"Missing essential fields {missing_fields} in file: {json_file}")
                continue

            episode_metadata = {
                'episode_title': episode_title,
                'sanitized_episode_title': sanitized_episode_title,
                'publication_date': publication_date,
                'duration': duration,
                'episode_number': episode_number,
                'explicit': explicit,
                'guid': guid,
                'audio_url': audio_url,
                'image_url': image_url,
                'channel_image_url': channel_image_url,
                'description': description,
                'categories': categories,
                'language': language,
                'link': link,
                'transcript': transcript,
                'translated': translated,
            }

            if translated is False:
                # Original record: insert into master tables.
                episode_id, inserted = insert_episode(conn, channel_id, episode_metadata)
                if not episode_id:
                    logging.warning(f"Skipping episode '{episode_title}' due to missing episode ID.")
                    continue
                if inserted:
                    segments = data.get('segments', [])
                    if segments:
                        count_inserted = insert_transcript(conn, episode_id, segments)
                        if count_inserted == 0:
                            logging.warning(f"No segments inserted for episode '{episode_title}'.")
                    else:
                        logging.warning(f"No segments found in file: {json_file}")
                    chapters = data.get('chapters', [])
                    if chapters:
                        insert_chapters(conn, episode_id, chapters)
                    else:
                        logging.warning(f"No chapters found in file: {json_file}")
                else:
                    logging.info(f"Episode '{episode_title}' already exists in master. Skipping transcripts and chapters insertion.")
            else:
                # Translated record: do not insert into master.
                # Instead, find the master episode using its GUID.
                master_episode_id = get_master_episode_id(conn, guid)
                if not master_episode_id:
                    logging.warning(f"Master episode not found for translated episode '{episode_title}' (GUID: {guid}). Skipping.")
                    continue
                # Optionally update the channel translation as well.
                channel_metadata = {
                    'channel_title': channel_title,
                    'sanitized_channel_title': sanitized_channel_title,
                    'channel_image_url': channel_image_url,
                    'channel_author': channel_author,
                    'channel_summary': channel_summary,
                    'language': language,
                    'translated': True,
                }
                insert_channel_translation(conn, channel_id, channel_metadata)
                # Insert or update the episode translation.
                insert_episode_translation(conn, master_episode_id, episode_metadata)
                chapters = data.get('chapters', [])
                if chapters:
                    insert_chapters_translation(conn, master_episode_id, chapters, language)
                else:
                    logging.warning(f"No chapters found in file: {json_file}")
                segments = data.get('segments', [])
                if segments:
                    insert_transcript_translation(conn, master_episode_id, segments, language)
                else:
                    logging.warning(f"No segments found in file: {json_file}")

        except json.JSONDecodeError as jde:
            logging.error(f"JSON decoding failed for file '{json_file}': {jde}")
        except Exception as e:
            logging.error(f"Failed to process file '{json_file}': {e}")

def get_master_episode_id(conn, guid):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM Episodes WHERE guid = %s;", (guid,))
        result = cursor.fetchone()
        return result[0] if result else None

def main():
    DB_NAME = "podcast_db"
    DB_USER = "postgres"
    DB_PASSWORD = "root"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    ROOT_DIRECTORY = "transcripts"
    try:
        connection = connect_db(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
        populate_database(connection, ROOT_DIRECTORY)
    except Exception as main_e:
        logging.critical(f"An unhandled exception occurred: {main_e}")
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
