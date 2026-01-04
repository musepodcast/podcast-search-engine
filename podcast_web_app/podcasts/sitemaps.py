# podcasts/sitemaps.py
from django.contrib.sitemaps import Sitemap
from django.urls import reverse
from django.db.models import Max
from .models import Channel, Episode

class StaticViewSitemap(Sitemap):
    changefreq = "daily"
    priority = 1.0
    protocol = "https"

    def items(self):
        # these are your public, non-auth views
        return [
            "podcasts:home",
            "podcasts:channel_list",
            "podcasts:episode_list",
        ]

    def location(self, item):
        return reverse(item)


class ChannelSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.8
    protocol = "https"
    limit = 50000

    def items(self):
        return (
            Channel.objects
            .exclude(sanitized_channel_title__isnull=True)
            .exclude(sanitized_channel_title__exact="")
            .annotate(latest_pub=Max("episodes__publication_date"))
            .only("id", "sanitized_channel_title")   # keeps it light
            .order_by("id")                          # stable paging
        )

    def location(self, obj):
        return reverse("podcasts:channel_detail",
                       kwargs={"sanitized_channel_title": obj.sanitized_channel_title})

    def lastmod(self, obj):
        return obj.latest_pub


class EpisodeSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.6
    protocol = "https"
    limit = 50000

    def items(self):
        return (
            Episode.objects
            .select_related("channel")
            .exclude(sanitized_episode_title__isnull=True)
            .exclude(sanitized_episode_title__exact="")
            .exclude(channel__sanitized_channel_title__isnull=True)
            .exclude(channel__sanitized_channel_title__exact="")
            .exclude(publication_date__isnull=True)
            .only(
                "id",
                "publication_date",
                "sanitized_episode_title",
                "channel__sanitized_channel_title",
            )
            .order_by("id")
        )

    def location(self, obj):
        return reverse("podcasts:episode_detail", kwargs={
            "sanitized_channel_title": obj.channel.sanitized_channel_title,
            "sanitized_episode_title": obj.sanitized_episode_title,
        })

    def lastmod(self, obj):
        return obj.publication_date

