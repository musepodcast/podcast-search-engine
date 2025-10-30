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

    def items(self):
        # avoid channels with no slug — your detail view needs sanitized_channel_title
        return (
            Channel.objects
            .exclude(sanitized_channel_title__isnull=True)
            .exclude(sanitized_channel_title__exact="")
        )

    def location(self, obj):
        return reverse(
            "podcasts:channel_detail",
            kwargs={"sanitized_channel_title": obj.sanitized_channel_title},
        )

    def lastmod(self, obj):
        # use newest episode in that channel as "lastmod"
        latest_ep = (
            Episode.objects
            .filter(channel=obj)
            .aggregate(m=Max("publication_date"))["m"]
        )
        return latest_ep


class EpisodeSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.6
    protocol = "https"

    def items(self):
        # only episodes that actually have a slug + channel slug
        return (
            Episode.objects
            .select_related("channel")
            .exclude(sanitized_episode_title__isnull=True)
            .exclude(sanitized_episode_title__exact="")
            .exclude(channel__sanitized_channel_title__isnull=True)
            .exclude(channel__sanitized_channel_title__exact="")
            # optional: only publish those with a real date
            .exclude(publication_date__isnull=True)
        )

    def location(self, obj):
        return reverse(
            "podcasts:episode_detail",
            kwargs={
                "sanitized_channel_title": obj.channel.sanitized_channel_title,
                "sanitized_episode_title": obj.sanitized_episode_title,
            },
        )

    def lastmod(self, obj):
        return obj.publication_date
