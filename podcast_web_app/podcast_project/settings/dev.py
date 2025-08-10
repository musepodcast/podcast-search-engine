# podcast_project/settings/dev.py
from .base import *
import warnings
from elastic_transport import SecurityWarning
from urllib3.exceptions import InsecureRequestWarning
import os
import environ    # if you installed django-environ

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
env = environ.Env()
environ.Env.read_env()   # reads the .env file

SITE_ID = 1
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
ACCOUNT_DEFAULT_HTTP_PROTOCOL = "http"

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
        "ENGINE":   env("DB_ENGINE"),
        "NAME":     env("DB_NAME"),
        "USER":     env("DB_USER"),
        "PASSWORD": env("DB_PASSWORD"),
        "HOST":     env("DB_HOST"),
        "PORT":     env("DB_PORT"),
    }
}

SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "APP": {
            "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
            "key": "",
        },
        "SCOPE": ["profile", "email"],
        "AUTH_PARAMS": {"access_type": "offline", "prompt": "consent"},
        },
    "twitter": {
        "APP": {
            "client_id": os.environ["TWITTER_CLIENT_ID"],
            "secret":    os.environ["TWITTER_SECRET"],
            "key":       "",
        },
        "SCOPE": ["email"],
        "AUTH_PARAMS": {"include_email": "true"},
    },
    "github": {
        "APP": {
            "client_id": env("GITHUB_CLIENT_ID_DEV", default=""),
            "secret":    env("GITHUB_CLIENT_SECRET_DEV", default=""),
            "key":       "",  
        },
        # optional scopes
        "SCOPE": ["read:user", "user:email"],
        "AUTH_PARAMS": {"allow_signup": "true"},
    },
}