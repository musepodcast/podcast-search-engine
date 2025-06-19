from django.contrib import admin
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns
from two_factor import urls as two_factor_urls
from django.http import HttpResponse

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

two_factor_patterns = filter_valid_patterns(two_factor_urls.urlpatterns)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    # Include podcasts URLs at the root (so home page is available at "/")
    path('i18n/', include('django.conf.urls.i18n')),
    path('favicon.ico', favicon_view),
    path('', include((two_factor_patterns, 'two_factor'), namespace='two_factor')),
    path('', include('podcasts.urls', namespace='podcasts')),
    
]
#path('', include((two_factor_patterns, 'two_factor'), namespace='two_factor')),