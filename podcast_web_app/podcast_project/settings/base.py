# podcast_project/settings/base.py

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-default-key")

_env = os.environ.get("EPISODE_JSON_BASE")  # optional override

if _env:
    # Supports "~", "~user", absolute or relative paths from env
    EPISODE_JSON_BASE = Path(os.path.expanduser(_env))
else:
    # Robust default: home/podcast_data/transcripts
    EPISODE_JSON_BASE = Path.home() / "podcast_data" / "transcripts"

# Optional: normalize (doesn't fail if it doesn't exist)
EPISODE_JSON_BASE = EPISODE_JSON_BASE.resolve(strict=False)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "60"))
EPISODE_ASSISTANT_MAX_SEGMENTS = int(os.environ.get("EPISODE_ASSISTANT_MAX_SEGMENTS", "8"))

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
    "django.contrib.sitemaps",
    "django.contrib.sites",
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
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
    "allauth.socialaccount.providers.twitter", 
    "allauth.socialaccount.providers.github",
    "allauth.socialaccount.providers.apple",
    
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "allauth.account.middleware.AccountMiddleware",
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
                "podcasts.context_processors.admin_ticket_counts",
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
#AXES_LOCKOUT_TEMPLATE = "security/locked_out.html"
AXES_LOCKOUT_URL = "/account/locked/"



# == Allauth core ==
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_LOGIN_METHODS = {"email", "username"}  # both allowed
ACCOUNT_SIGNUP_FIELDS = ["email*", "username*", "password1*", "password2*"]


# Require email verification + 24h expiry
ACCOUNT_EMAIL_VERIFICATION = "mandatory"
ACCOUNT_EMAIL_CONFIRMATION_EXPIRE_DAYS = 1  # 24 hours
# Show a page with a button instead of auto-confirming on GET
ACCOUNT_CONFIRM_EMAIL_ON_GET = False

# After successful confirmation, send anonymous users to login
LOGIN_URL = "/login/"
ACCOUNT_EMAIL_CONFIRMATION_ANONYMOUS_REDIRECT_URL = "/login/"

# If a logged-in user hits a confirm link, send them somewhere sensible:
ACCOUNT_EMAIL_CONFIRMATION_AUTHENTICATED_REDIRECT_URL = "/channels"



# Use our adapter (activates user on confirm)
ACCOUNT_ADAPTER = "podcasts.adapters.AccountAdapter"

# Use our custom manual-signup form
ACCOUNT_FORMS = {
    "signup": "podcasts.forms.CustomSignupForm",
}

# Optional: don’t auto-login immediately on confirm (pick your preference)
ACCOUNT_LOGIN_ON_EMAIL_CONFIRMATION = False


AUTHENTICATION_BACKENDS = [
    'axes.backends.AxesStandaloneBackend',
    'podcasts.authentication.EmailOrUsernameModelBackend',
    'django.contrib.auth.backends.ModelBackend',  # optional fallback
    'allauth.account.auth_backends.AuthenticationBackend',
]

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
DEFAULT_FROM_EMAIL = 'webmaster@localhost'
# Internationalization, static files, etc.

SOCIALACCOUNT_ADAPTER = "podcasts.adapters.ConditionalSocialAdapter"
SOCIALACCOUNT_AUTO_SIGNUP = False  # force the “extra data” step
SOCIALACCOUNT_FORMS = {
    "signup": "podcasts.forms.CustomSocialSignupForm",
}




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
TIME_ZONE = 'America/Chicago'
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

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

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
