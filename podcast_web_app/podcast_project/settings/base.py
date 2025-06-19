# podcast_project/settings/base.py

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-default-key")

# Common settings
DEBUG = False  # default here, override in dev.py

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    "django_filters",
    "django_elasticsearch_dsl",
    "podcasts.search",
    "podcasts",
    "phonenumber_field",
    "widget_tweaks",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_otp",
    "django_otp.plugins.otp_totp",
    "django_otp.plugins.otp_hotp",
    "django_otp.plugins.otp_static",
    "two_factor",
    "two_factor.plugins.phonenumber",
    "axes",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "podcasts.middleware.PageVisitMiddleware",  # your custom middleware
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django_otp.middleware.OTPMiddleware",
    "axes.middleware.AxesMiddleware",
]

ROOT_URLCONF = "podcast_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "podcasts.context_processors.unseen_replies",
                "django.template.context_processors.i18n",
            ],
        },
    },
]

WSGI_APPLICATION = "podcast_project.wsgi.application"

ELASTICSEARCH_DSL = {
  'default': {
    'hosts': 'https://localhost:9200',
    'http_auth': ('elastic', 'Mz9O_U5WDu6rvHsEBRa6'),
    'verify_certs': False,
  },
  'signal_processor': 'django_elasticsearch_dsl.signals.RealtimeSignalProcessor',
}



AUTH_USER_MODEL = 'podcasts.CustomUser'
LOGIN_REDIRECT_URL = '/channels'
LOGOUT_REDIRECT_URL = '/channels/'
ADMIN_LOGIN_URL = '/admin/login/'

# Axes settings, email backend, and other security settings go here...
AXES_FAILURE_LIMIT = 5
AXES_COOLOFF_TIME = 1  # in hours
AXES_LOCK_OUT_AT_FAILURE = True
AXES_RESET_ON_SUCCESS = True
AXES_LOCKOUT_PARAMETERS = ['username', 'ip_address']

AUTHENTICATION_BACKENDS = [
    'axes.backends.AxesStandaloneBackend',
    'podcasts.authentication.EmailOrUsernameModelBackend',
    'django.contrib.auth.backends.ModelBackend',  # optional fallback
]

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
DEFAULT_FROM_EMAIL = 'webmaster@localhost'
# Internationalization, static files, etc.

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LANGUAGE_CODE = "en"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

LANGUAGES = [
    ('en', 'English'),  # English
    ('pt', 'Português'),    # Portuguese 
    ('es', 'Español'),  # Spanish  
    ('it', 'Italiano'),  # Italian
    ('fr', 'Français'),  # French
    ('ru', 'Русский'),  # Russian
    ('uk', 'українська'),   # Ukrainian
    ('cn', '中文 (简体)'),  # Simplified Chinese
    ('tw', '中文 (繁體)'),  # Traditional Chinese
    ('ko', '한국어'),  # Korean
    ('ja', '日本語'),  # Japanese
    ('tr', 'Türkçe'),  # Turkish
    ('de', 'Deutsch'),  # German
    ('ar', 'العربية'),  # Arabic
    ('hi', 'हिन्दी'),  # Hindi
    ('vi', 'Tiếng Việt'),  # Vietnamese
    ('tl', 'Tagalog'),  # Tagalog
    # add more as needed
]

LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale'),
]

STATIC_URL = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Logging (example)
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        # Capture debug from our podcasts.views module:
        "podcasts.views": {
            "handlers": ["console"],
            "level": "DEBUG",
        },
    },
}
