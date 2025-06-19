# enhanced_podcastparser_test.py
import podcastparser
import requests
import logging
import io
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
    ]
)

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
        total_seconds = int(duration_seconds)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02}:{m:02}:{s:02}"
    except Exception as e:
        logging.warning(f"Failed to normalize duration '{duration_seconds}': {e}")
        return None

def parse_podcast_feed(feed_url):
    """
    Parse the podcast feed using podcastparser and extract metadata.

    Parameters:
    - feed_url: str, the URL of the podcast RSS feed.

    Returns:
    - feed_data: dict, parsed feed data.
    """
    try:
        logging.debug(f"Fetching RSS feed from URL: {feed_url}")
        response = requests.get(feed_url)
        response.raise_for_status()  # Ensure we got a valid response
        feed_content = response.text  # Get the content as a decoded string

        # Wrap the feed content in a StringIO object
        feed_content_io = io.StringIO(feed_content)

        # Parse the feed content
        feed_data = podcastparser.parse(feed_url, feed_content_io)
        logging.debug(f"Successfully parsed feed with podcastparser: {feed_url}")
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

def process_feed(feed_url):
    """
    Process a single RSS feed: parse, extract metadata from the first episode, and log it.

    Parameters:
    - feed_url: str, the URL of the podcast RSS feed.
    """
    logging.info(f"\nProcessing feed: {feed_url}")
    feed_data = parse_podcast_feed(feed_url)
    if not feed_data:
        logging.error(f"Failed to parse feed: {feed_url}")
        return

    # Extract feed-level metadata
    feed_explicit = feed_data.get('explicit')
    logging.info(f"Feed-level explicit: {feed_explicit}")

    channel_title = feed_data.get('title', 'Unknown Channel')
    logging.info(f"Channel Title: {channel_title}")

    # Extract language
    language = extract_feed_language(feed_data)
    logging.info(f"Feed Language: {language if language else 'None'}")

    # Extract feed-level categories
    categories = extract_feed_categories(feed_data)
    logging.info(f"Feed Categories: {categories if categories else 'None'}")

    # Extract channel image
    channel_image_url = extract_channel_image_url(feed_data)
    logging.info(f"Channel Image URL: {channel_image_url if channel_image_url else 'None'}")

    # Extract episodes
    episodes = feed_data.get('episodes', [])
    logging.info(f"Number of episodes found: {len(episodes)}")

    if not episodes:
        logging.warning("No episodes found in the feed.")
        return

    # Process only the first episode
    episode = episodes[0]
    logging.info(f"Processing episode: {episode.get('title', 'No Title')}")

    # Log all keys in the episode dictionary
    logging.debug(f"Episode keys: {list(episode.keys())}")

    # Extract episode-level metadata
    episode_title = episode.get('title', 'Unknown Title')
    episode_explicit = episode.get('explicit')

    # Handle publication date with multiple possible keys
    publication_date_str = None  # Initialize as None
    if 'published' in episode and episode.get('published'):
        episode_pub_date = episode.get('published')
        # Check if published is a timestamp (integer or float) or string
        if isinstance(episode_pub_date, (int, float)):
            try:
                publication_date = datetime.fromtimestamp(episode_pub_date, timezone.utc)
                publication_date_str = publication_date.isoformat()
                logging.debug(f"Published (timestamp converted): {publication_date_str}")
            except Exception as e:
                logging.warning(f"Failed to convert published timestamp '{episode_pub_date}': {e}")
                publication_date_str = episode_pub_date  # Keep original if conversion fails
        else:
            publication_date_str = episode_pub_date
            logging.debug(f"Published (string): {episode_pub_date}")
    elif 'pubDate' in episode and episode.get('pubDate'):
        episode_pub_date = episode.get('pubDate')
        publication_date_str = episode_pub_date
        logging.debug(f"PubDate (string): {episode_pub_date}")
    elif 'published_parsed' in episode and episode.get('published_parsed'):
        pub_date = episode.get('published_parsed')
        if isinstance(pub_date, tuple):
            try:
                publication_date = datetime(*pub_date[:6], tzinfo=timezone.utc)
                publication_date_str = publication_date.isoformat()
                logging.debug(f"Published Parsed (tuple): {pub_date} converted to {publication_date_str}")
            except Exception as e:
                logging.warning(f"Failed to convert published_parsed '{pub_date}': {e}")
                publication_date_str = None
        else:
            publication_date_str = None
            logging.debug(f"Published Parsed (non-tuple): {pub_date}")
    else:
        episode_pub_date = None
        publication_date_str = None
        logging.debug("No publication date found.")

    # Normalize duration
    episode_duration_seconds = episode.get('total_time')
    episode_duration = normalize_duration(episode_duration_seconds)
    logging.debug(f"Duration Seconds: {episode_duration_seconds}")

    # Extract GUID
    episode_guid = episode.get('guid')
    logging.debug(f"GUID: {episode_guid}")

    # Extract Link
    episode_link = episode.get('link')
    logging.debug(f"Link: {episode_link}")

    # Extract Description
    episode_description = episode.get('description')
    logging.debug(f"Description: {episode_description}")

    # Extract Enclosures (for Audio URL)
    episode_enclosures = episode.get('enclosures', [])
    if episode_enclosures:
        audio_url = episode_enclosures[0].get('url')
        logging.debug(f"Audio URL: {audio_url}")
    else:
        audio_url = None
        logging.debug("No enclosures found for audio URL.")

    # Extract episode image URL using podcastparser's 'episode_art_url'
    episode_image_url = extract_episode_image_url(episode)
    logging.info(f"Episode Image URL: {episode_image_url if episode_image_url else 'None'}")

    # Handle episode number with multiple possible keys
    episode_episode_number = None
    if 'itunes_episode' in episode and episode.get('itunes_episode'):
        episode_episode_number = episode.get('itunes_episode')
        logging.debug(f"Episode Number from 'itunes_episode': {episode_episode_number}")
    elif 'episode_number' in episode and episode.get('episode_number'):
        episode_episode_number = episode.get('episode_number')
        logging.debug(f"Episode Number from 'episode_number': {episode_episode_number}")
    elif 'number' in episode and episode.get('number'):
        episode_episode_number = episode.get('number')
        logging.debug(f"Episode Number from 'number': {episode_episode_number}")
    elif 'episode' in episode and episode.get('episode'):
        episode_episode_number = episode.get('episode')
        logging.debug(f"Episode Number from 'episode': {episode_episode_number}")
    else:
        logging.debug("No episode number found.")

    # Compile episode metadata
    episode_metadata = {
        'title': episode_title,
        'explicit': safe_parse_boolean(episode_explicit),
        'publication_date': publication_date_str,
        'duration': episode_duration,
        'guid': episode_guid,
        'link': episode_link,
        'description': episode_description,
        'audio_url': audio_url,
        'episode_image_url': episode_image_url,  # Separate field for episode image
        'channel_image_url': channel_image_url,  # Separate field for channel image
        'episode_number': episode_episode_number,
        'categories': categories,  # Included feed-level categories
        'language': language,      # Included feed-level language
        # You can add more fields here if needed
    }

    # Log the extracted metadata
    logging.info("\n--- Extracted Episode Metadata ---")
    for key, value in episode_metadata.items():
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

def main():
    # List of RSS feed URLs to process
    rss_feed_urls = [
        'https://feeds.megaphone.fm/GLT1412515089',     # Joe Rogan Experience
        'https://anchor.fm/s/2fa50a94/podcast/rss',    # PBD
        'https://tschimandher.libsyn.com/rss',         # The Skinny Confidential Him & Her Podcast
        'https://feeds.megaphone.fm/WWO7410387571',    # Shawn Ryan Show grab 68
        'https://feeds.megaphone.fm/APPI6857213837',    # Andrew Schulz's Flagrant with Akaash Singh
    ]

    # Iterate through each RSS feed and process it
    for feed_url in rss_feed_urls:
        process_feed(feed_url)

if __name__ == "__main__":
    main()
