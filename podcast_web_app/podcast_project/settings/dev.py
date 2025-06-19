# podcast_project/settings/dev.py
from .base import *
import warnings
from elastic_transport import SecurityWarning
from urllib3.exceptions import InsecureRequestWarning
import os
import environ    # if you installed django-environ

# Silence ES’s SecurityWarning about TLS+verify_certs=False
warnings.filterwarnings("ignore", category=SecurityWarning)

# And also silence urllib3’s InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
env = environ.Env()
environ.Env.read_env()   # reads the .env file


DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

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


# Optionally, override the database settings for development if needed:
# Configure Postgres via environment variables
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "podcast_db"),
        "USER": os.environ.get("DB_USER", "postgres"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}

