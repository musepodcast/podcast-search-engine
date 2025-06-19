# podcast_project/settings/production.py

from .base import *
import warnings
import os
from pathlib import Path
import environ    # if you installed django-environ
from elastic_transport import SecurityWarning
from urllib3.exceptions import InsecureRequestWarning

# BASE_DIR should point to the folder containing manage.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Tell django-environ where to find your .env
env = environ.Env(
    # you can declare casting/defaults here if you like:
    DEBUG=(bool, False),
)
environ.Env.read_env(env_file=BASE_DIR / ".env")


# Silence ES’s SecurityWarning about TLS+verify_certs=False
warnings.filterwarnings("ignore", category=SecurityWarning)

# And also silence urllib3’s InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

DEBUG = False
ALLOWED_HOSTS = [
    "musepodcast.com",
    "www.musepodcast.com",
    "192.168.1.228",
    "host.docker.internal",
]

ELASTICSEARCH_DSL = {
    "default": {
        "hosts": os.environ.get("ES_HOST", "https://localhost:9200"),
        "http_auth": (
            os.environ.get("ES_USER", "elastic"),
            os.environ.get("ES_PASSWORD", ""),
        ),
        "verify_certs": os.environ.get("ES_VERIFY_CERTS", "False") == "True",
    },
    "signal_processor": "django_elasticsearch_dsl.signals.RealtimeSignalProcessor",
}

# Use environment variables for database settings, e.g.:
# Use one DATABASE_URL instead of separate DB_NAME etc.
DATABASES = {
    "default": {
        "ENGINE": os.environ.get("DB_ENGINE", "django.db.backends.postgresql"),
        "NAME":   os.environ.get("DB_NAME",   "podcast_db"),
        "USER":   os.environ.get("DB_USER",   "postgres"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST":   os.environ.get("DB_HOST",   "localhost"),
        "PORT":   os.environ.get("DB_PORT",   "5432"),
    }
}

# Tell Django where to collect static to, and configure WhiteNoise
STATIC_ROOT = BASE_DIR / "staticfiles"

# (Optional) enable compression and long-term caching
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"


# Additional production settings:
# Right after your other security settings:
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
