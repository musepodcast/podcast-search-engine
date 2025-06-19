# podcasts/search/documents.py

from django_elasticsearch_dsl import Document, Index, fields
from django_elasticsearch_dsl.registries import registry

from podcasts.models import (
    Channel,
    ChannelTranslations,
    Episode,
    EpisodeTranslations,
    Transcript,
    TranscriptTranslations,
)

# ────── CHANNEL ──────
channels_index = Index('channels')
channels_index.settings(number_of_shards=1, number_of_replicas=0)

@registry.register_document
@channels_index.document
class ChannelDocument(Document):
    channel_title   = fields.TextField()
    channel_summary = fields.TextField()

    class Django:
        model = Channel
        # The two explicit fields above are all we need.

    def get_queryset(self):
        return Channel.objects.all()


# ────── CHANNEL TRANSLATIONS ──────
channel_translations_index = Index('channel_translations')
channel_translations_index.settings(number_of_shards=1, number_of_replicas=0)

@registry.register_document
@channel_translations_index.document
class ChannelTranslationsDocument(Document):
    language        = fields.KeywordField()
    channel_title   = fields.TextField()
    channel_summary = fields.TextField()

    class Django:
        model = ChannelTranslations

    def get_queryset(self):
        # only index ones that have actually been translated
        return ChannelTranslations.objects.filter(translated=True)


# ────── EPISODE ──────
episodes_index = Index('episodes')
episodes_index.settings(number_of_shards=1, number_of_replicas=0)

@registry.register_document
@episodes_index.document
class EpisodeDocument(Document):
    episode_title     = fields.TextField()
    publication_date  = fields.DateField()      # will accept your timestamp-without-time-zone
    channel_id        = fields.IntegerField()   # so you can link back to Channels

    class Django:
        model = Episode

    def get_queryset(self):
        return Episode.objects.all()


# ────── EPISODE TRANSLATIONS ──────
episode_translations_index = Index('episode_translations')
episode_translations_index.settings(number_of_shards=1, number_of_replicas=0)

@registry.register_document
@episode_translations_index.document
class EpisodeTranslationsDocument(Document):
    language         = fields.KeywordField()
    episode_title    = fields.TextField()
    publication_date = fields.DateField()
    episode_id       = fields.IntegerField(attr='episode_id')
    channel_id       = fields.IntegerField(attr='episode.channel.id')

    class Django:
        model = EpisodeTranslations

    def get_queryset(self):
        return EpisodeTranslations.objects.filter(translated=True)


# ────── TRANSCRIPT ──────
transcripts_index = Index('transcripts')
transcripts_index.settings(number_of_shards=1, number_of_replicas=0)

@registry.register_document
@transcripts_index.document
class TranscriptDocument(Document):
    segment_text = fields.TextField()
    segment_time = fields.KeywordField()
    episode_id   = fields.IntegerField(attr='episode_id')

    class Django:
        model = Transcript

    def get_queryset(self):
        return Transcript.objects.all()


# ────── TRANSCRIPT TRANSLATIONS ──────
transcript_translations_index = Index('transcript_translations')
transcript_translations_index.settings(number_of_shards=1, number_of_replicas=0)

@registry.register_document
@transcript_translations_index.document
class TranscriptTranslationsDocument(Document):
    language      = fields.KeywordField()
    segment_text  = fields.TextField()
    segment_time  = fields.KeywordField()
    episode_id    = fields.IntegerField(attr='episode_id')

    class Django:
        model = TranscriptTranslations

    def get_queryset(self):
        return TranscriptTranslations.objects.all()
