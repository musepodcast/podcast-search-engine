from django.contrib import admin
from django.urls import path, include
from django.contrib.sitemaps.views import sitemap as sitemap_view, index as sitemap_index
from django.views.decorators.cache import cache_page
from podcasts.sitemaps import (
    StaticViewSitemap,
    ChannelSitemap,
    EpisodeSitemap,
)

from django.conf.urls.i18n import i18n_patterns
from two_factor import urls as two_factor_urls
from django.http import HttpResponse
from podcasts.adapters import CancelledRedirectView
from django.conf import settings
from django.conf.urls.static import static
from pathlib import Path
from django.views.generic import TemplateView

# Point at ~/podcast_data/support_attachments
SUPPORT_ATTACHMENTS_ROOT = Path.home() / 'podcast_data' / 'support_attachments'

def favicon_view(request):
    # Return an empty response with the correct content type.
    return HttpResponse("", content_type="image/x-icon")

def filter_valid_patterns(patterns):
    from django.urls.resolvers import URLPattern, URLResolver
    valid = []
    for p in patterns:
        if isinstance(p, (URLPattern, URLResolver)):
            valid.append(p)
        elif isinstance(p, list):
            valid.extend(filter_valid_patterns(p))
    return valid

def index_no_xrobots(request, *args, **kwargs):
    resp = sitemap_index(request, *args, **kwargs)
    resp.headers.pop("X-Robots-Tag", None)
    return resp

def sitemap_no_xrobots(request, *args, **kwargs):
    resp = sitemap_view(request, *args, **kwargs)
    resp.headers.pop("X-Robots-Tag", None)
    return resp

two_factor_patterns = filter_valid_patterns(two_factor_urls.urlpatterns)

sitemaps = {
    "static": StaticViewSitemap,
    "channels": ChannelSitemap,
    "episodes": EpisodeSitemap,
}



urlpatterns = [
    path('admin/', admin.site.urls),
    path("accounts/<str:provider>/login/cancelled/", CancelledRedirectView.as_view(), name="socialaccount_login_cancelled"),
    path("accounts/", include("allauth.urls")), 
    path('auth/', include('django.contrib.auth.urls')),
    # Include podcasts URLs at the root (so home page is available at "/")
    path('i18n/', include('django.conf.urls.i18n')),
    path('favicon.ico', favicon_view),
    path('', include((two_factor_patterns, 'two_factor'), namespace='two_factor')),
    path('', include('podcasts.urls', namespace='podcasts')),
    path("sitemap.xml", cache_page(60 * 60)(index_no_xrobots), {"sitemaps": sitemaps, "sitemap_url_name": "sitemap-sections"}, name="sitemap-index",),
    path("sitemap-<section>.xml", cache_page(60 * 60)(sitemap_no_xrobots), {"sitemaps": sitemaps}, name="sitemap-sections",),
    path("robots.txt", TemplateView.as_view(template_name="podcasts/robots.txt", content_type="text/plain")),
    
]
#path('', include((two_factor_patterns, 'two_factor'), namespace='two_factor')),

if settings.DEBUG:
    # 1) Serve attachments *first*:
    urlpatterns += static(
        '/media/support_attachments/',
        document_root=SUPPORT_ATTACHMENTS_ROOT
    )
    # 2) Then fall back to other MEDIA:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT 
    )