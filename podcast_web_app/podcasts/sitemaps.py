# podcasts/sitemaps.py

from django.contrib.sitemaps import Sitemap
from django.urls import reverse

class StaticViewSitemap(Sitemap):
    priority = 1.0
    changefreq = "daily"
    protocol = "https"

    def items(self):
        # return the names of the URL patterns you want in the sitemap
        return ["podcasts:home"]

    def location(self, item):
        # reverse() that URL name
        return reverse(item)
