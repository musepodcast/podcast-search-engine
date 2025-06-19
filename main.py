# main.py

from dateutil import parser
import time
from datetime import datetime, timezone
import subprocess
import os
import yaml  # For handling YAML files
from transformers import pipeline as transformers_pipeline  # Renamed for clarity
from downloader import parse_feed, download_audio
from transcriber import transcribe_and_diarize, initialize_diarization_pipeline, validate_audio
from utils import sanitize_filename
from pydub import AudioSegment
import logging
import json
import re
import math
import warnings
import spacy
import nltk  # For NLTK data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch  # For tensor operations
import podcastparser  # Imported for parsing feeds
import requests  # Imported for HTTP requests
import io  # Imported for handling IO operations
from bs4 import BeautifulSoup  # Imported for clean_html function
import torch.nn.functional as F
from pathlib import Path

BASE = Path(__file__).parent            # …\podcast_news
# where all your artifacts now live
DATABASE_ROOT = BASE.parent / "podcast_data"

# Suppress specific Pyannote Audio warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote.audio')

# If ffmpeg is not in PATH, set the path explicitly
# Uncomment and set the correct paths if necessary
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# Configure logging (centralized configuration)
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs during troubleshooting
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler(str(DATABASE_ROOT / "log" / "transcription.log"), encoding='utf-8')
    ]
)

# Automatic download of NLTK stopwords and punkt if not present
try:
    nltk.data.find('corpora/stopwords')
    logging.info("NLTK stopwords corpus already exists.")
except LookupError:
    logging.info("NLTK stopwords corpus not found. Downloading...")
    nltk.download('stopwords')
    logging.info("NLTK stopwords corpus downloaded successfully.")

try:
    nltk.data.find('tokenizers/punkt')
    logging.info("NLTK punkt tokenizer already exists.")
except LookupError:
    logging.info("NLTK punkt tokenizer not found. Downloading...")
    nltk.download('punkt')
    logging.info("NLTK punkt tokenizer downloaded successfully.")

# Initialize NLP models once
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load spaCy model: {e}")
    nlp = None

# Optionally, load the SentenceTransformer with FP16 precision.
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    sentence_model.eval()

    #sentence_model.half()  # Convert the model's parameters to FP16
    logging.info("Sentence-BERT model loaded successfully on CUDA.")
except Exception as e:
    logging.critical(f"Failed to load Sentence-BERT model: {e}")
    sentence_model = None

def convert_to_wav(input_path, wav_path):
    ext = os.path.splitext(input_path)[1].lower()
    try:
        # Let pydub auto-detect, or force format for known containers:
        if ext in ['.mp4', '.m4a', '.aac']:
            audio = AudioSegment.from_file(input_path, format='mp4')
        else:
            audio = AudioSegment.from_file(input_path)  # works for .mp3, .wav, etc.

        audio.export(wav_path, format='wav')
        return True
    except Exception as e:
        logging.error(f"Conversion error for {input_path}: {e}", exc_info=True)
        return False

# A helper function to split long texts and then average the embeddings.
def get_embedding(model, text, max_tokens=256):
    # Ensure the text is non-empty and strip whitespace.
    if not text or not text.strip():
        logging.warning("Empty or whitespace-only text provided to get_embedding.")
        return None
    text = text.strip()

    try:
        tokenizer = getattr(model, "tokenizer", None)
    except Exception as e:
        logging.error("Failed to retrieve tokenizer from model", exc_info=True)
        tokenizer = None

    # If a tokenizer is available, use it to check token count and split if necessary.
    if tokenizer:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True)
        except Exception as e:
            logging.error("Tokenization failed", exc_info=True)
            return None

        if len(tokens) > max_tokens:
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence.strip()
                try:
                    candidate_tokens = tokenizer.encode(candidate, add_special_tokens=True)
                except Exception as e:
                    logging.error("Tokenization failed for candidate text", exc_info=True)
                    continue

                if len(candidate_tokens) > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence.strip()
                else:
                    current_chunk = candidate
            if current_chunk:
                chunks.append(current_chunk)

            embeddings = []
            for chunk in chunks:
                try:
                    start_time = time.perf_counter()
                    embedding = model.encode(chunk, convert_to_tensor=True, show_progress_bar=False)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start_time
                    logging.info(f"Computed embedding for chunk (len {len(chunk)} characters) in {elapsed:.3f} seconds")
                    embeddings.append(embedding)
                except Exception as e:
                    logging.error(f"Failed to compute embedding for chunk: {chunk}", exc_info=True)
            if embeddings:
                return torch.stack(embeddings, dim=0).mean(dim=0)
            else:
                logging.error("No valid embeddings computed from chunks.")
                return None
        else:
            try:
                start_time = time.perf_counter()
                embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                logging.info(f"Computed embedding for text in {elapsed:.3f} seconds")
                return embedding
            except Exception as e:
                logging.error("Failed to compute embedding for text", exc_info=True)
                return None
    else:
        # If no tokenizer is available, proceed directly.
        try:
            start_time = time.perf_counter()
            embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            logging.info(f"Computed embedding for text (no tokenizer) in {elapsed:.3f} seconds")
            return embedding
        except Exception as e:
            logging.error("Failed to compute embedding for text (no tokenizer)", exc_info=True)
            return None


# Function to load configuration
def load_config(config_path='config.yaml'):
    """
    Load YAML configuration file.
    
    Parameters:
    - config_path (str): Path to the YAML configuration file.
    
    Returns:
    - dict: Configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {config_path}")
        # Validate required sections and parameters
        required_sections = ['summarizer', 'chapter_generation']
        for section in required_sections:
            if section not in config:
                logging.critical(f"Missing section '{section}' in configuration file. Exiting.")
                exit(1)
        logging.info(f"All required configuration sections are present.")
        return config
    except FileNotFoundError:
        logging.critical(f"Configuration file {config_path} not found. Exiting.")
        exit(1)
    except yaml.YAMLError as e:
        logging.critical(f"Error parsing YAML file: {e}")
        exit(1)

# Load the configuration
config = load_config()

# Initialize Summarization pipeline using config parameters
try:
    summarizer = transformers_pipeline(
        "summarization",
        model=config['summarizer']['model'],
        tokenizer=config['summarizer']['model'],
        framework="pt"  # Ensure PyTorch framework is used
    )
    logging.info("Summarization pipeline initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize summarization pipeline: {e}", exc_info=True)
    summarizer = None

# --------------------------- Integrated Metadata Extraction Functions ---------------------------

def safe_parse_boolean(value):
    """
    Safely parse a value to a boolean.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ['yes', 'true', '1', 'explicit']:
            return True
        elif value_lower in ['no', 'false', '0', 'clean']:
            return False
        else:
            logging.warning(f"Unrecognized explicit value '{value}'. Setting to None.")
            return None
    elif isinstance(value, (int, float)):
        return bool(value)
    else:
        logging.warning(f"Unsupported type for explicit value '{value}'. Setting to None.")
        return None

def normalize_duration(duration_seconds):
    """
    Normalize duration in seconds to HH:MM:SS format.
    """
    if duration_seconds is None:
        return None
    try:
        # Some feeds may provide duration in 'HH:MM:SS' or 'MM:SS' format as strings
        if isinstance(duration_seconds, str):
            parts = duration_seconds.split(':')
            parts = [int(part) for part in parts]
            while len(parts) < 3:
                parts.insert(0, 0)  # Prepend zeros for missing hours or minutes
            h, m, s = parts[-3:]
            total_seconds = h * 3600 + m * 60 + s
        else:
            total_seconds = int(duration_seconds)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02}:{m:02}:{s:02}"
    except Exception as e:
        logging.warning(f"Failed to normalize duration '{duration_seconds}': {e}")
        return None

def safe_parse_int(value):
    """
    Safely parse a value to an integer.
    
    Parameters:
    - value: The value to parse.
    
    Returns:
    - int or None: Parsed integer or None if parsing fails.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_parse_date(date_input):
    """
    Safely parse a date string or struct_time to a datetime object.

    Parameters:
    - date_input (str or struct_time): The date to parse.

    Returns:
    - datetime or None: Parsed datetime object or None if parsing fails.
    """
    try:
        if isinstance(date_input, time.struct_time):
            # Convert struct_time to datetime
            return datetime.fromtimestamp(time.mktime(date_input), timezone.utc)
        elif isinstance(date_input, str):
            return parser.parse(date_input)
        else:
            return None
    except Exception as e:
        logging.warning(f"Failed to parse date '{date_input}': {e}")
        return None

def parse_podcast_feed(feed_url):
    """
    Parse the podcast feed using podcastparser and extract metadata.
    
    If the URL is an Apple Podcasts page, use iTunes Lookup to get the real RSS feed URL.
    Otherwise fall back to normal requests + HTML/RSS parsing.
    """
    try:
        # 1) Apple page detection → iTunes Lookup
        if 'podcasts.apple.com' in feed_url:
            m = re.search(r'/id(\d+)', feed_url)
            if not m:
                logging.error(f"Can't find Apple ID in URL: {feed_url}")
                return None
            pid = m.group(1)
            lookup = requests.get(
                'https://itunes.apple.com/lookup',
                params={'id': pid},
                timeout=10
            )
            lookup.raise_for_status()
            results = lookup.json().get('results', [])
            if not results or not results[0].get('feedUrl'):
                logging.error(f"iTunes lookup failed for ID {pid}")
                return None

            real_feed = results[0]['feedUrl']
            logging.info(f"iTunes lookup: {feed_url} → {real_feed}")
            feed_url = real_feed

        # 2) Fetch the RSS/XML (either the original URL, or the feedUrl from lookup)
        resp = requests.get(feed_url, allow_redirects=True, timeout=10)
        resp.raise_for_status()
        content = resp.text

        # 3) Parse as RSS/XML
        feed_io   = io.StringIO(content)
        feed_data = podcastparser.parse(feed_url, feed_io)
        logging.debug(f"Parsed feed successfully: {feed_url}")

        # 4) (Optional) grab <itunes:author>
        m2 = re.search(r'<itunes:author>(.*?)</itunes:author>', content, re.IGNORECASE)
        if m2:
            feed_data['manual_author'] = m2.group(1).strip()

        return feed_data

    except Exception as e:
        logging.error(f"Error parsing podcast feed '{feed_url}': {e}", exc_info=True)
        return None


def extract_feed_categories(feed_data):
    """
    Extract categories from the feed data.

    Parameters:
    - feed_data: dict, parsed feed data.

    Returns:
    - categories: list of str, list of category names.
    """
    categories = []
    itunes_categories = feed_data.get('itunes_categories', [])

    if not itunes_categories:
        logging.debug("No itunes_categories found in feed data.")
        return categories

    # Log the structure of 'itunes_categories' for debugging
    logging.debug(f"itunes_categories: {itunes_categories}")

    # Flatten the list if it's a list of lists
    for category in itunes_categories:
        if isinstance(category, list):
            for sub_cat in category:
                if isinstance(sub_cat, str):
                    categories.append(sub_cat)
                    logging.debug(f"Extracted category: {sub_cat}")
                elif isinstance(sub_cat, dict):
                    text = sub_cat.get('text')
                    if text:
                        categories.append(text)
                        logging.debug(f"Extracted category: {text}")
                    else:
                        logging.debug("Sub-category dictionary does not contain 'text' key.")
                else:
                    logging.debug(f"Unrecognized sub-category format: {sub_cat}")
        elif isinstance(category, dict):
            # Handle dictionary format
            text = category.get('text')
            if text:
                categories.append(text)
                logging.debug(f"Extracted category: {text}")
            else:
                logging.debug("Category dictionary does not contain 'text' key.")
        elif isinstance(category, str):
            # Handle string format
            categories.append(category)
            logging.debug(f"Extracted category: {category}")
        else:
            logging.debug(f"Unrecognized category format: {category}")

    return categories

def extract_feed_language(feed_data):
    """
    Extract the language from the feed data.

    Parameters:
    - feed_data: dict, parsed feed data.

    Returns:
    - language: str or None, language of the podcast feed.
    """
    language = feed_data.get('language')
    if language:
        logging.debug(f"Extracted language: {language}")
    else:
        logging.debug("No language found in feed data.")
    return language

def extract_channel_image_url(feed_data):
    """
    Extracts the channel image URL from the feed data.

    Parameters:
    - feed_data: dict, parsed feed data.

    Returns:
    - channel_image_url: str or None, URL of the channel image.
    """
    channel_image_url = None
    # First, check 'cover_url' in feed_data
    if 'cover_url' in feed_data and feed_data.get('cover_url'):
        channel_image_url = feed_data.get('cover_url')
        logging.debug(f"Channel Image URL from 'cover_url': {channel_image_url}")
    # Then, check 'image' in feed_data
    elif 'image' in feed_data:
        image = feed_data.get('image')
        if isinstance(image, dict):
            channel_image_url = image.get('href')
            if channel_image_url:
                logging.debug(f"Channel Image URL from 'image' dict href: {channel_image_url}")
        elif isinstance(image, str):
            channel_image_url = image
            logging.debug(f"Channel Image URL from 'image' str: {channel_image_url}")
    else:
        logging.debug("No channel image URL found in feed data.")
    return channel_image_url

def extract_episode_image_url(episode):
    """
    Extracts the episode image URL from the episode data.

    Parameters:
    - episode: dict, parsed episode data.

    Returns:
    - episode_image_url: str or None, URL of the episode image.
    """
    episode_image_url = None
    # Check 'itunes_image' in episode
    if 'itunes_image' in episode:
        itunes_image = episode.get('itunes_image')
        if isinstance(itunes_image, dict):
            episode_image_url = itunes_image.get('href')
            if episode_image_url:
                logging.debug(f"Episode Image URL from 'itunes_image' dict href: {episode_image_url}")
        elif isinstance(itunes_image, str):
            episode_image_url = itunes_image
            logging.debug(f"Episode Image URL from 'itunes_image' str: {episode_image_url}")
    # Fallback: Check 'image' in episode
    if not episode_image_url and 'image' in episode:
        image = episode.get('image')
        if isinstance(image, dict):
            episode_image_url = image.get('href')
            if episode_image_url:
                logging.debug(f"Episode Image URL from 'image' dict href: {episode_image_url}")
        elif isinstance(image, str):
            episode_image_url = image
            logging.debug(f"Episode Image URL from 'image' str: {episode_image_url}")
    # Additional Fallback: Check 'episode_art_url' in episode
    if not episode_image_url and 'episode_art_url' in episode:
        episode_image_url = episode.get('episode_art_url')
        if episode_image_url:
            logging.debug(f"Episode Image URL from 'episode_art_url': {episode_image_url}")
    # If still not found, set to None
    if not episode_image_url:
        logging.debug("No episode image URL found in episode data.")
    return episode_image_url

def clean_html(raw_html):
    """
    Remove HTML tags from a string using BeautifulSoup.

    Parameters:
    - raw_html: str, the raw HTML string.

    Returns:
    - str, the cleaned text without HTML tags.
    """
    if not isinstance(raw_html, str):
        logging.warning(f"Description is not a string: {raw_html}")
        return "No Description Available"

    # Simple regex to check for HTML tags
    if not re.search(r'<[^>]+>', raw_html):
        logging.debug(f"Description does not contain HTML tags: {raw_html}")
        return raw_html  # Return as is or handle accordingly

    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def split_audio(file_path, chunk_length_ms=300000):  # 5 minutes
    """
    Split a large audio file into smaller chunks of specified length.

    Parameters:
    - file_path: str, path to the large audio file.
    - chunk_length_ms: int, length of each chunk in milliseconds (default is 5 minutes).

    Returns:
    - List of chunk file paths.
    """
    try:
        audio = AudioSegment.from_wav(file_path)
    except Exception as e:
        logging.error(f"Failed to load WAV file {file_path}: {e}", exc_info=True)
        return []

    total_length = len(audio)
    num_chunks = math.ceil(total_length / chunk_length_ms)
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, total_length)
        chunk = audio[start:end]
        chunk_filename = f"{os.path.splitext(file_path)[0]}_chunk{i+1}.wav"
        try:
            chunk.export(chunk_filename, format="wav")
            chunks.append(chunk_filename)
            logging.info(f"Created chunk: {chunk_filename}")
        except Exception as e:
            logging.error(f"Failed to export chunk {chunk_filename}: {e}", exc_info=True)

    return chunks

# Global cache for embeddings
_embedding_cache = {}

def cached_get_embedding(model, text, max_tokens=256):
    if text in _embedding_cache:
        logging.info("Cache hit for text.")
        return _embedding_cache[text]
    else:
        logging.info("Cache miss for text; computing embedding.")
        embedding = get_embedding(model, text, max_tokens)
        _embedding_cache[text] = embedding
        return embedding



# Updated compute_similarity that uses get_embedding().
def compute_similarity(model, text1, text2):
    if not model:
        logging.error("Sentence-BERT model is not loaded.")
        return 0.0
    try:
        # Get (possibly averaged) embeddings for each text using the cache.
        emb1 = cached_get_embedding(model, text1, max_tokens=256)
        emb2 = cached_get_embedding(model, text2, max_tokens=256)
        # Normalize and compute cosine similarity.
        emb1 = F.normalize(emb1, p=2, dim=0)
        emb2 = F.normalize(emb2, p=2, dim=0)
        similarity = torch.dot(emb1, emb2).item()
        return similarity
    except Exception as e:
        logging.error(f"Error computing similarity: {e}", exc_info=True)
        return 0.0

# Optionally update is_title_unique so that it uses get_embedding() as well.
def is_title_unique(new_title, chapters, model, similarity_threshold=0.6):
    if not model:
        logging.error("Sentence-BERT model is not loaded.")
        return False
    try:
        if not chapters:
            return True

        new_emb = get_embedding(model, new_title, max_tokens=256)
        new_emb = F.normalize(new_emb, p=2, dim=0)

        # Cache the embeddings for existing chapter titles.
        existing_texts = [chapter['title'] for chapter in chapters]
        # Compute embeddings for all titles in one batch
        existing_embeddings = model.encode(existing_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
        if existing_embeddings.dim() == 1:
            existing_embeddings = existing_embeddings.unsqueeze(0)
        existing_embeddings = F.normalize(existing_embeddings, p=2, dim=1)
        similarities = F.cosine_similarity(new_emb.unsqueeze(0), existing_embeddings)
        max_sim = similarities.max().item() if similarities.numel() > 0 else 0.0
        logging.debug(f"Max similarity of '{new_title}' with existing titles: {max_sim}")
        return max_sim < similarity_threshold
    except Exception as e:
        logging.error(f"Error in uniqueness check: {e}", exc_info=True)
        return False

def clean_title(title):
    """
    Clean the chapter title by removing unwanted characters,
    ensuring proper punctuation and capitalization.
    
    Parameters:
    - title (str): The chapter title to clean.
    
    Returns:
    - str: Cleaned chapter title.
    """
    try:
        # Remove unwanted characters except for periods, hyphens, and apostrophes
        title = re.sub(r'[^\w\s\.-]', '', title)
        
        # Ensure the title ends with a period if not already punctuated
        if not title.endswith('.'):
            title += '.'
        
        # Capitalize the first letter of the title
        title = title.capitalize()
        
        return title
    except Exception as e:
        logging.error(f"Error cleaning title '{title}': {e}", exc_info=True)
        return title  # Return the original title if cleaning fails

def is_title_valid(title, config):
    """
    Validate the generated chapter title based on predefined criteria.
    
    Parameters:
    - title (str): The chapter title to validate.
    - config (dict): Configuration parameters from config.yaml.
    
    Returns:
    - bool: True if the title is valid, False otherwise.
    """
    try:
        # Check for minimum word count
        word_count = len(title.split())
        if word_count < config['summarizer']['min_length']:
            logging.debug(f"Title '{title}' rejected for insufficient word count: {word_count} words.")
            return False
        
        # Check for excessive punctuation (allowing periods, hyphens, and apostrophes)
        if re.search(r'[^\w\s\.-]', title):
            logging.debug(f"Title '{title}' rejected for containing unwanted characters.")
            return False
        
        # Check for presence of at least one noun or named entity
        # This requires spaCy's NLP pipeline to be initialized
        if nlp:
            doc = nlp(title)
            has_noun = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
            if not has_noun:
                logging.debug(f"Title '{title}' rejected for lacking nouns or proper nouns.")
                return False
        else:
            logging.warning("spaCy model is not available for POS tagging.")
        
        return True
    except Exception as e:
        logging.error(f"Error validating title '{title}': {e}", exc_info=True)
        return False  # Reject the title if validation fails

def preprocess_text(text):
    """
    Preprocess the input text by removing filler words and unnecessary spaces.
    
    Parameters:
    - text (str): The text to preprocess.
    
    Returns:
    - str: Cleaned text.
    """
    try:
        # Remove filler words like "uh", "um", "like", "you know"
        fillers = ['uh', 'um', 'like', 'you know']
        pattern = re.compile(r'\b(' + '|'.join(fillers) + r')\b', flags=re.I)
        text = pattern.sub('', text)
        
        # Remove extra spaces
        text = re.sub(' +', ' ', text)
        
        return text.strip()
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}", exc_info=True)
        return text  # Return the original text if preprocessing fails

def verify_entities(title, segment_text):
    """
    Verify that the entities in the title are present in the segment text.
    
    Parameters:
    - title (str): The chapter title.
    - segment_text (str): The original transcript segment.
    
    Returns:
    - bool: True if entities are verified, False otherwise.
    """
    if not nlp:
        logging.warning("spaCy model is not available for entity verification.")
        return True  # Skip verification if spaCy is not available
    try:
        title_doc = nlp(title)
        segment_doc = nlp(segment_text)
        title_entities = set([ent.text.lower() for ent in title_doc.ents])
        segment_entities = set([ent.text.lower() for ent in segment_doc.ents])
        if not title_entities:
            return True  # No entities to verify
        verified = bool(title_entities & segment_entities)
        logging.debug(f"Entity verification: {verified} for title '{title}'")
        return verified
    except Exception as e:
        logging.error(f"Error during entity verification: {e}", exc_info=True)
        return False

def generate_chapter_title(segment_text, config=None):
    """
    Generate a descriptive chapter title from a segment's text using the summarizer.
    
    Parameters:
    - segment_text (str): The text of the transcript segment.
    - config (dict): Configuration parameters from config.yaml.
    
    Returns:
    - str or None: Generated chapter title if valid, None otherwise.
    """
    try:
        if not summarizer:
            logging.error("Summarization pipeline is not available.")
            return None

        # Preprocess text
        segment_text = preprocess_text(segment_text)

        # Truncate text if too long (adjust max_length as needed)
        max_input_length = 1024  # or use summarizer.model.config.max_position_embeddings if available
        if len(segment_text) > max_input_length:
            segment_text = segment_text[:max_input_length]

        # Generate summary using the summarizer with enhanced parameters to reduce hallucinations
        summary = summarizer(
            segment_text,
            max_length=config['summarizer']['max_length'],
            min_length=config['summarizer']['min_length'],
            do_sample=False,
            num_beams=6,  # Increased from 4 to allow more beams for better accuracy
            no_repeat_ngram_size=3,  # Increased from 2 to reduce repetition
            length_penalty=2.0,  # Encourages shorter summaries
            early_stopping=True
        )[0]['summary_text']
        
        # Clean and format title
        title = clean_title(summary)
        
        # Validate the title
        if not is_title_valid(title, config):
            logging.debug(f"Invalid chapter title generated: '{title}'. Skipping.")
            return None
        
        # Additional similarity check between title and segment to ensure relevance
        title_similarity = compute_similarity(sentence_model, title, segment_text)
        if title_similarity < 0.2:  # Threshold can be adjusted based on experimentation
            logging.debug(f"Title '{title}' has low similarity ({title_similarity}) with segment. Skipping.")
            return None
        
        # Verify entities to reduce hallucinations
        if not verify_entities(title, segment_text):
            logging.debug(f"Entities in title '{title}' are not present in segment. Skipping.")
            return None
        
        logging.debug(f"Generated Chapter Title: {title} with similarity {title_similarity}")
        return title
    except Exception as e:
        logging.error(f"Error generating chapter title: {e}", exc_info=True)
        return None  # Return None to indicate failure

def aggregate_segments_non_overlapping(segments, window_size=5):
    """
    Aggregate segments into non-overlapping windows.
    
    Parameters:
    - segments (list): List of transcript segments.
    - window_size (int): Number of segments to aggregate.
    
    Returns:
    - list of str: Aggregated texts.
    """
    aggregated_texts = []
    for i in range(0, len(segments), window_size):
        window = segments[i:i + window_size]
        aggregated_text = ' '.join([segment.get('text', '') for segment in window])
        aggregated_texts.append(aggregated_text)
    return aggregated_texts

def add_chapters_to_transcript(transcript_json_path, config):
    """
    Add chapters to the transcript JSON based on summarization.
    
    Parameters:
    - transcript_json_path (str): Path to the transcript JSON file.
    - config (dict): Configuration parameters.
    """
    similarity_threshold = config['chapter_generation']['similarity_threshold']
    max_chapters = config['chapter_generation']['max_chapters']
    aggregation_window_size = config['chapter_generation']['aggregation_window_size']

    try:
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data.get('segments', [])
        chapters = []

        if not segments:
            logging.warning("No segments found in transcript.")
            return

        # Add "Intro" as the first chapter
        chapters.append({
            'title': 'Intro',
            'time': '0:00'
        })
        logging.info(f"Added chapter: 'Intro' at 0:00")

        # Aggregate texts using non-overlapping windows
        aggregated_texts = aggregate_segments_non_overlapping(segments, window_size=aggregation_window_size)

        for idx, aggregated_text in enumerate(aggregated_texts):
            start_segment_idx = idx * aggregation_window_size
            if start_segment_idx >= len(segments):
                break
            start_time = segments[start_segment_idx].get('start', 0)  # Start time in seconds

            if idx == 0:
                # The first aggregation window corresponds to the beginning, already covered by "Intro"
                # Depending on your preference, you can choose to skip or include it
                # Here, we'll process it normally to generate a chapter
                pass  # No action needed

            else:
                # Compute similarity with the previous aggregation window
                similarity = compute_similarity(sentence_model, aggregated_texts[idx], aggregated_texts[idx -1])

                logging.debug(f"Aggregated Text {idx}: Similarity with previous: {similarity}")

                if similarity < similarity_threshold:
                    # Generate chapter title using summarizer
                    title = generate_chapter_title(aggregated_text, config=config)

                    if not title:
                        # Title was invalid or generation failed
                        logging.debug(f"Skipped adding chapter due to invalid title.")
                        continue

                    # Check for uniqueness against all existing titles
                    if not is_title_unique(title, chapters, sentence_model, similarity_threshold=0.6):
                        logging.debug(f"Skipped adding chapter due to similarity with existing titles: '{title}'")
                        continue

                    # Convert start_time to mm:ss
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    timestamp = f"{minutes}:{seconds:02d}"

                    chapters.append({
                        'title': title,
                        'time': timestamp
                    })

                    logging.info(f"Added chapter: '{title}' at {timestamp}")

                    # Check if max_chapters limit is reached
                    if len(chapters) >= max_chapters:
                        logging.info(f"Reached maximum number of chapters: {max_chapters}")
                        break

        # Add chapters to data
        data['chapters'] = chapters

        # Save updated JSON
        with open(transcript_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Chapters added successfully to {transcript_json_path}")

    except Exception as e:
        logging.error(f"Failed to add chapters: {e}", exc_info=True)

def process_chunk(chunk, pipeline):
    """
    Process a single audio chunk:
    - Transcribe audio
    - Perform speaker diarization
    - Return transcription data
    """
    logging.info(f"Processing chunk: {chunk}")
    try:
        transcription_data = transcribe_and_diarize(
            chunk,          # Pass as positional argument
            pipeline        # Pass as positional argument
        )
        return transcription_data
    except TypeError as e:
        logging.error(f"Error processing chunk {chunk}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing chunk {chunk}: {e}")
        return None

def process_entry(entry, channel_transcript_dir, download_dir, channel_title, pipeline, config, feed_data, channel_summary, channel_author):
    """
    Process a single podcast episode entry:
    - Download audio
    - Convert MP3 to WAV
    - Split WAV into chunks
    - Transcribe each chunk with speaker diarization
    - Adjust segment times based on chunk offsets
    - Generate chapter titles from the transcript segments
    - Save a combined JSON with all transcripts, segments, and chapters
    """
    try:
        # Extract episode metadata
        raw_title = entry.get('title', 'Unknown Title')
        sanitized_title = sanitize_filename(raw_title)
        logging.debug(f"Sanitized Episode Title: {sanitized_title}")

        mp3_filename = f"{sanitized_title}.mp3"
        wav_filename = f"{sanitized_title}.wav"
        mp3_file_path = os.path.join(download_dir, mp3_filename)
        wav_file_path = os.path.join(download_dir, wav_filename)
        transcript_filename = os.path.join(channel_transcript_dir, f"{sanitized_title}.json")

        # Check if transcript already exists
        if os.path.exists(transcript_filename):
            logging.info(f"Transcript already exists: {transcript_filename}")
            return  # Skip to the next entry

        # Download audio file if it doesn't exist
        if not os.path.exists(mp3_file_path):
            logging.info(f"Downloading audio for: {sanitized_title}")
            audio_file = download_audio(entry, download_dir, mp3_filename)
            if not audio_file:
                logging.error(f"Failed to download audio for entry: {sanitized_title}")
                return  # Skip to the next entry
        else:
            logging.info(f"MP3 file already exists: {mp3_file_path}")

        # Convert MP3 to WAV
        if not os.path.exists(wav_file_path):
            try:
                logging.info(f"Converting {mp3_file_path} to WAV...")
                
                # Added try-except block to handle conversion errors for mp4 files
                if not convert_to_wav(mp3_file_path, wav_file_path):
                    logging.error(f"Failed to convert {mp3_file_path} to WAV. Skipping.")
                    return

                logging.info(f"Successfully converted to {wav_file_path}")
            except Exception as e:
                logging.error(f"Failed to convert {mp3_file_path} to WAV: {e}", exc_info=True)
                return  # Skip to the next entry
        else:
            logging.info(f"WAV file already exists: {wav_file_path}")

        # Split WAV into 5-minute chunks
        try:
            audio = AudioSegment.from_wav(wav_file_path)
        except Exception as e:
            logging.error(f"Failed to load WAV file for splitting: {e}", exc_info=True)
            return  # Skip to the next entry

        if len(audio) > 300000:  # 5 minutes in ms
            logging.info(f"Splitting {wav_file_path} into 5-minute chunks...")
            chunks = split_audio(wav_file_path)
        else:
            chunks = [wav_file_path]

        if not chunks:
            logging.error(f"No chunks created for {wav_file_path}. Skipping transcription.")
            return

        combined_data = {
            'transcript': '',
            'segments': [],
            'metadata': {}
            # Removed 'summary' as per user request
        }

        

        cumulative_time = 0  # To adjust segment times based on chunk offsets
        chunk_length_ms = 300000  # 5 minutes in milliseconds

        # Process chunks sequentially
        for idx, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {idx + 1}/{len(chunks)}: {chunk}")

            # Validate audio file
            if not validate_audio(chunk):
                logging.error(f"Invalid audio file: {chunk}. Skipping.")
                cumulative_time += chunk_length_ms / 1000.0  # Still increment time
                continue

            # Transcribe and diarize
            transcription_data = process_chunk(chunk, pipeline)  # Corrected call
            if transcription_data:
                # Append transcript
                combined_data['transcript'] += transcription_data.get('transcript', '') + ' '

                # Adjust and append segments
                for segment in transcription_data.get('segments', []):
                    adjusted_segment = {
                        'start': segment['start'] + cumulative_time,
                        'end': segment['end'] + cumulative_time,
                        'text': segment['text'],
                        'speaker': segment['speaker']
                    }
                    combined_data['segments'].append(adjusted_segment)
            else:
                logging.error(f"Failed to retrieve transcription data from chunk: {chunk}")

            cumulative_time += chunk_length_ms / 1000.0  # Increment by 5 minutes

        # Extract and clean description from 'description' field
        raw_description = entry.get('description', 'No Description Available')
        logging.debug(f"Raw Description for '{sanitized_title}': {raw_description}")

        if raw_description != 'No Description Available':
            clean_description = clean_html(raw_description)
            logging.debug(f"Cleaned Description for '{sanitized_title}': {clean_description}")
        else:
            logging.warning(f"No description found for episode: {sanitized_title}")
            clean_description = 'No Description Available'

        # Extract publication date
        publication_date = safe_parse_date(
            entry.get('published_parsed') or
            entry.get('published') or
            entry.get('updated') or
            entry.get('pubDate') or
            entry.get('date')
        )

        # Convert publication_date to ISO format string if not None
        publication_date_str = publication_date.isoformat() if publication_date else None

        # Extract episode image URL using the integrated function
        episode_image_url = extract_episode_image_url(entry)
        logging.info(f"Episode Image URL: {episode_image_url if episode_image_url else 'None'}")

        # Extract channel image URL using the integrated function
        channel_image_url = extract_channel_image_url(feed_data)
        logging.info(f"Channel Image URL: {channel_image_url if channel_image_url else 'None'}")

        

        # Extract explicit value
        explicit_value = entry.get('itunes_explicit')
        if explicit_value is None:
            # Fallback to feed-level itunes_explicit using integrated metadata extraction
            if feed_data:
                explicit_value = feed_data.get('explicit')
                logging.debug(f"Feed-level explicit value found: {explicit_value}")
            else:
                explicit_value = None
        # Parse the explicit value using safe_parse_boolean
        explicit = safe_parse_boolean(explicit_value)

        

        # Extract categories and language using the integrated metadata extraction
        categories = extract_feed_categories(feed_data) if feed_data else None
        language = extract_feed_language(feed_data) if feed_data else None

        authors_list = entry.get("authors")
        episode_author = None

        # 1) Check 'authors' array (like feedparser can store for some feeds)
        if authors_list and len(authors_list) > 0:
            # e.g. authors_list = [{"name": "Shawn Ryan"}]
            episode_author = authors_list[0].get("name")

        # 2) Fallback to 'author' or 'itunes_author'
        if not episode_author:
            episode_author = entry.get("author") or entry.get("itunes_author")

        # 3) Fallback to channel_author
        if not episode_author:
            episode_author = channel_author




        # Extract duration from multiple possible fields
        duration = normalize_duration(
            entry.get('itunes_duration') or
            entry.get('total_time') or
            entry.get('duration') or
            entry.get('length') or
            entry.get('time')  # Add more as needed
        )

        # Extract episode number from multiple possible fields
        episode_number = safe_parse_int(
            entry.get('itunes_episode') or
            entry.get('episode_number') or
            entry.get('number') or
            entry.get('episode')  # Add more as needed
        )

        #Translated is False unless the .json has been modified by translated.py then the value is equal to true
        translated = False

        # Compile episode metadata
        metadata = {
            'channel_title': channel_title,
            'episode_title': raw_title,
            'sanitized_episode_title': sanitized_title,
            'publication_date': publication_date_str,
            'duration': duration,
            'episode_number': episode_number,
            'explicit': explicit,
            "author": episode_author,
            'summary': channel_summary,
            'guid': entry.get('guid'),
            'audio_url': entry.enclosures[0].get('url') if entry.enclosures else None,
            'image_url': episode_image_url,         # Episode image
            'channel_image_url': channel_image_url, # Channel image
            'description': clean_description, 
            'categories': categories,  # Included feed-level categories
            'language': language,      # Included feed-level language
            'link': entry.get('link'),
            'translated': translated,
        }

        # Log the extracted metadata
        logging.info("\n--- Extracted Episode Metadata ---")
        for key, value in metadata.items():
            # Format key for better readability
            formatted_key = key.replace('_', ' ').capitalize()
            logging.info(f"{formatted_key}: {value}")

        # Log feed-level categories separately
        logging.info("\n--- Feed-Level Categories ---")
        if categories:
            for idx, category in enumerate(categories, start=1):
                logging.info(f"Category {idx}: {category}")
        else:
            logging.info("No categories found.")

        # Combine metadata with transcript and segments
        combined_data['metadata'] = metadata

        # Save combined data as a single JSON file
        logging.info(f"Saving combined transcription, metadata, and segments to {transcript_filename}")
        try:
            os.makedirs(os.path.dirname(transcript_filename), exist_ok=True)  # Ensure directory exists
            with open(transcript_filename, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=4, ensure_ascii=False)
            logging.info(f"Combined data saved successfully: {transcript_filename}")
        except Exception as e:
            logging.error(f"Failed to save combined data: {e}", exc_info=True)
            return

        # Add chapters to the transcript with summarization
        add_chapters_to_transcript(transcript_filename, config)

        # Delete chunk WAV files first
        for chunk in chunks:
            try:
                os.remove(chunk)
                logging.info(f"Deleted chunk file: {chunk}")
            except FileNotFoundError:
                logging.warning(f"Chunk file not found, could not delete: {chunk}")
            except Exception as e:
                logging.warning(f"Could not delete chunk file '{chunk}': {e}")

        # Delete the original MP3 file after successful processing
        try:
            os.remove(mp3_file_path)
            logging.info(f"Deleted MP3 file: {mp3_file_path}")
        except Exception as e:
            logging.warning(f"Could not delete MP3 file: {e}")

        # Delete the main WAV file only if it wasn't part of the chunks
        if wav_file_path not in chunks:
            try:
                os.remove(wav_file_path)
                logging.info(f"Deleted main WAV file: {wav_file_path}")
            except Exception as e:
                logging.warning(f"Could not delete main WAV file: {e}")

        return combined_data
    except Exception as e:
        logging.error(f"An error occurred while processing entry '{sanitized_title}': {e}", exc_info=True)      

def main():
    """
    podcast_feeds = [
        'https://feeds.megaphone.fm/jessemichels',              # American Alchemy
        'https://audioboom.com/channels/5033205.rss',           # Redacted News
        'https://feeds.megaphone.fm/candace',                   # Candace Owens
        'https://feeds.megaphone.fm/GLT1412515089',             # Joe Rogan Experience
        'https://feeds.megaphone.fm/RSV1597324942',             # The Tucker Carlson Show
        'https://feeds.megaphone.fm/WWO7410387571',             # Shawn Ryan Show
        'https://anchor.fm/s/2fa50a94/podcast/rss',             # PBD
        'http://feeds.feedburner.com/gangster-capitalism',      # Campus Files
        'https://tschimandher.libsyn.com/rss',                  # The Skinny Confidential Him & Her Podcast
        'https://feeds.megaphone.fm/APPI6857213837',            # Andrew Schulz's Flagrant with Akaash Singh
        'https://www.spreaker.com/show/5975113/episodes/feed',  # Total Disclosure: UFOs-CoverUps & Conspiracy
        'https://feeds.simplecast.com/ob9OSBIN',                # The Economics of Everyday Things
        'https://feeds.megaphone.fm/search-engine',             # Search Engine

        # Add more RSS feed URLs here
    ]
    """
    
    # --- load your feeds parameters ---
    #feeds_cfg = yaml.safe_load(open('feeds_config.yaml', 'r'))['feeds']
    master_file = DATABASE_ROOT / "watcher_json" / "master_rss.json"

     # read your master list of { name, url }
    try:
        with open(master_file, 'r', encoding='utf-8') as f:
            master = json.load(f)
    except FileNotFoundError:
        logging.critical(f"Could not find {master_file}; run update_master_rss.py first.")
        return

    # just take the URL field for each entry
    podcast_feeds = [ entry['url'] for entry in master ]
    # initialize list for feeds needing manual review
    failed_feeds = []
    download_dir        = DATABASE_ROOT / "podcasts"
    base_transcript_dir = DATABASE_ROOT / "transcripts"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(base_transcript_dir, exist_ok=True)

    # Initialize the diarization pipeline once
    diarization_pipeline = initialize_diarization_pipeline()
    if not diarization_pipeline:
        logging.critical("Diarization pipeline failed to initialize. Exiting.")
        return

    # Ensure summarizer is initialized (Retained for chapter titles)
    if not summarizer:
        logging.critical("Summarization pipeline is not available. Exiting.")
        return

    for feed_url in podcast_feeds:
        logging.info(f"Processing feed: {feed_url}")

        feed = parse_feed(feed_url)
        if not feed:
            failed_feeds.append(feed_url)
            continue

        # Check for entries; treat zero entries as a failure to review
        entries = getattr(feed, 'entries', [])
        if not entries:
            logging.warning(f"No entries to process for feed: {feed_url}")
            failed_feeds.append(feed_url)
            continue

        # Parse the feed using the integrated metadata extraction
        feed_data = parse_podcast_feed(feed_url)
        #logging.debug("Full feed_data:")
        #logging.debug(json.dumps(feed_data, indent=2, ensure_ascii=False))
        if not feed_data:
            logging.error(f"Failed to parse feed data for: {feed_url}")
            continue

        # Extract the channel name from the feed
        channel_title = feed_data.get('title', 'Unknown_Channel')

        sanitized_channel_title = sanitize_filename(channel_title)
        logging.info(f"Channel Title: {channel_title}")
        logging.info(f"Sanitized Channel Title: {sanitized_channel_title}")

        channel_author = None

        # 1) If 'authors' is populated:
        if 'authors' in feed_data and feed_data['authors']:
            # Usually a list of strings
            channel_author = feed_data['authors'][0]

        # 2) Or if 'itunes_tags' has an 'author' key:
        elif 'itunes_tags' in feed_data and 'author' in feed_data['itunes_tags']:
            channel_author = feed_data['itunes_tags']['author']

        # 3) Or if feed_data['author'] is present
        elif 'author' in feed_data:
            channel_author = feed_data['author']

        # 4) Or if feed_data['manual_author'] is present
        elif 'manual_author' in feed_data:
            channel_author = feed_data['manual_author']

        logging.info(f"Channel Author: {channel_author}")

        channel_summary = None

       
        # For the summary, some feeds place it in 'description', others in 'summary'
        channel_summary = feed_data.get('summary') or feed_data.get('description')
        logging.info(f"Channel Summary: {channel_summary}")
        

        # Determine source language (use feed language if available, otherwise default to "eng")
        source_lang = feed_data.get('language', 'eng').lower()
        # Create subdirectory: transcripts/<channel_name>/<source_lang>/
        channel_transcript_dir = os.path.join(base_transcript_dir, sanitized_channel_title, source_lang)
        os.makedirs(channel_transcript_dir, exist_ok=True)
        logging.info(f"Transcript directory for channel '{sanitized_channel_title}': {channel_transcript_dir}")

        entries = feed.entries
        logging.info(f"Number of entries found: {len(entries)}")

        if not entries:
            logging.warning(f"No entries to process for feed: {feed_url}")
            continue
        
        for entry in entries[:10]:  # Process only the latest 5 entries for testing
            logging.debug(f"Starting processing for entry: {entry.get('title', 'No Title')}")
            process_entry(entry, channel_transcript_dir, download_dir, channel_title, diarization_pipeline, config, feed_data, channel_summary, channel_author)

    # Print a list of failed feeds
    with open('failed_feeds.json', 'w', encoding='utf-8') as f:
        json.dump(failed_feeds, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(failed_feeds)} failed feeds to failed_feeds.json")

    # After processing all feeds, run the translation script.
    logging.info("All feeds processed. Running translation script...")
    try:
        subprocess.run(["python", "translate.py"], check=True)
        logging.info("Translation script completed successfully.")
    except Exception as e:
        logging.error(f"Translation script failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
