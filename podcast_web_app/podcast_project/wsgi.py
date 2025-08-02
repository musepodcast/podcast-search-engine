import os
from pathlib import Path
from django.core.wsgi import get_wsgi_application
from whitenoise import WhiteNoise

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "podcast_project.settings")

# 2a) Get the core Django WSGI app
application = get_wsgi_application()

# 2b) Let WhiteNoise serve your normal STATIC_ROOT at /static/
application = WhiteNoise(
    application,
    root=str(Path(__file__).resolve().parent.parent / "staticfiles"),
    prefix='static/'
)

# 2c) Also serve attachments from ~/podcast_data/support_attachments
application.add_files(
    str(Path.home() / 'podcast_data' / 'support_attachments'),
    prefix='media/support_attachments/'
)
