# podcasts/models.py

from django.db import models
from django.contrib.postgres.search import SearchVectorField
from django.contrib.postgres.indexes import GinIndex
from django.core.exceptions import ValidationError
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.utils import timezone
from phonenumber_field.modelfields import PhoneNumberField
from django_countries.fields import CountryField
from allauth.socialaccount.models import SocialAccount
from allauth.account.models import EmailAddress
from django.contrib.auth.models import AbstractUser
from django.core.files.storage import FileSystemStorage
from pathlib import Path
import string
import random
import os

SUPPORT_ATTACHMENTS_ROOT = Path.home() / 'podcast_data' / 'support_attachments'
# if you need a string path:
SUPPORT_ATTACHMENTS_ROOT = str(SUPPORT_ATTACHMENTS_ROOT)

support_attachment_storage = FileSystemStorage(
    location=str(Path.home() / 'podcast_data' / 'support_attachments'),
    base_url='/media/support_attachments/',
)

class Channel(models.Model):
    channel_title = models.TextField(unique=True)
    sanitized_channel_title = models.TextField()

    # ADD THESE FIELDS:
    channel_author = models.TextField(blank=True, null=True)
    channel_summary = models.TextField(blank=True, null=True)
    channel_image_url = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'channels'

    def __str__(self):
        return self.channel_title

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    enforce_2fa = models.BooleanField(
        default=False,
        help_text="If enabled, you will be required to enter an OTP every time you log in."
    )

    birthdate = models.DateField(null=True, blank=True)
    phone_number = PhoneNumberField(null=False, blank=False)
    
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    gender = models.CharField(
        max_length=1,
        choices=GENDER_CHOICES,
        default='O'
    )

    country = CountryField(blank_label='(select country)', default='US')
    # now that Channel is defined, this will resolve
        # “Am I currently running the community agent?”
    is_contributing = models.BooleanField(
        default=False,
        help_text="Whether this user is currently running the community agent."
    )

    # Which channels they’ve opted into
    contribute_channels = models.ManyToManyField(
        'podcasts.Channel',
        blank=True,
        help_text="Which channels this user has opted to contribute to."
    )
    support_ticket_limit = models.PositiveIntegerField(
        default=5,
        help_text="Max open (pending/in-progress) tickets this user can have."
    )
    def __str__(self):
        return self.username        

    def delete(self, *args, **kwargs):
        # Clean up allauth relations


        SocialAccount.objects.filter(user=self).delete()
        EmailAddress.objects.filter(user=self).delete()

        # Then delete the user itself
        super().delete(*args, **kwargs)

class Episode(models.Model):
    channel = models.ForeignKey(
        Channel,
        on_delete=models.CASCADE,
        related_name='episodes',
        blank=True,
        null=True
    )
    episode_title = models.TextField()
    sanitized_episode_title = models.TextField()
    publication_date = models.DateTimeField(blank=True, null=True)
    duration = models.TextField(blank=True, null=True)
    episode_number = models.IntegerField(blank=True, null=True)
    explicit = models.BooleanField(blank=True, null=True)
    guid = models.TextField(unique=True)
    audio_url = models.TextField(blank=True, null=True)
    image_url = models.TextField(blank=True, null=True)  # Renamed from 'episode_image'
    description = models.TextField(blank=True, null=True)
    categories = models.JSONField(blank=True, null=True)  
    language = models.CharField(max_length=50, blank=True, null=True)  # Added 'language'
    tsv_transcript = SearchVectorField(null=True, blank=True)  # Changed to SearchVectorField
    search_vector = SearchVectorField(null=True, editable=False) # Full-text search vector for the episode

    class Meta:
        managed = True
        db_table = 'episodes'
        unique_together = ('channel', 'episode_title')
        ordering = ['-publication_date']
        indexes = [
            GinIndex(fields=['search_vector'], name='episodes_search_vector_gin'),
            GinIndex(fields=['episode_title'], name='episodes_title_trgm_gin', opclasses=['gin_trgm_ops']),
            GinIndex(fields=['description'],    name='episodes_desc_trgm_gin', opclasses=['gin_trgm_ops']),
        ]
        
    @property
    def full_transcript(self) -> str:
        # join all transcript.segment_text for this episode:
        return " ".join(t.segment_text for t in self.transcripts.all())
    
    @property
    def thumbnail_url(self):
        """
        Return the episode’s image_url if set,
        otherwise the channel’s image_url,
        otherwise our static placeholder.
        """
        if self.image_url:
            return self.image_url
        if self.channel and self.channel.channel_image_url:
            return self.channel.channel_image_url
        return static('podcasts/images/no-image.jpg')
    
    def __str__(self):
        return self.episode_title
    
    def clean(self):
        super().clean()
        if self.categories:
            if not isinstance(self.categories, (list, dict)):
                raise ValidationError({'categories': 'Categories must be a list or dictionary.'})
            if isinstance(self.categories, list):
                for category in self.categories:
                    if not isinstance(category, str):
                        raise ValidationError({'categories': 'Each category must be a string.'})

class ChannelVisit(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    channel = models.ForeignKey(Channel, on_delete=models.CASCADE)
    count = models.PositiveIntegerField(default=0)
    last_visited = models.DateTimeField(auto_now=True)
    first_visited = models.DateTimeField(auto_now_add=True)
    last_ip_address = models.GenericIPAddressField(null=True, blank=True)  # new field

    def __str__(self):
        return f"{self.user.username} - {self.channel.channel_title}"

class EpisodeVisit(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    episode = models.ForeignKey(Episode, on_delete=models.CASCADE)
    count = models.PositiveIntegerField(default=0)
    last_visited = models.DateTimeField(auto_now=True)
    first_visited = models.DateTimeField(auto_now_add=True)
    last_ip_address = models.GenericIPAddressField(null=True, blank=True)  # new field
    
    def __str__(self):
        return f"{self.user.username} - {self.episode.episode_title}"
    
class SearchQuery(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        null=True, 
        blank=True, 
        on_delete=models.SET_NULL
    )
    query = models.CharField(max_length=255)
    search_in = models.TextField(blank=True)  # e.g., a comma-separated list of fields
    search_date = models.CharField(max_length=50, blank=True)  # e.g., 'anytime' or a specific value
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    count = models.PositiveIntegerField(default=1)
    first_searched = models.DateTimeField(auto_now_add=True)
    last_searched = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.query} by {self.user or 'anonymous'}"



class Transcript(models.Model):
    episode = models.ForeignKey(
        Episode,
        on_delete=models.CASCADE,
        related_name='transcripts',
        blank=True,
        null=True
    )
    segment_time = models.TextField()
    segment_text = models.TextField()
    speaker = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'transcripts'

    def __str__(self):
        return f"{self.speaker} at {self.segment_time}"


class Chapter(models.Model):
    episode = models.ForeignKey(
        Episode,
        on_delete=models.CASCADE,
        related_name='chapters',
        blank=True,
        null=True
    )
    chapter_title = models.TextField()
    chapter_time = models.CharField(max_length=10)

    class Meta:
        managed = False
        db_table = 'chapters'

    def __str__(self):
        return f"{self.chapter_title} at {self.chapter_time}"

"""
TRANSLATED SECTION TABLES
"""

class ChannelTranslations(models.Model):
    channel_title = models.TextField(unique=True)
    sanitized_channel_title = models.TextField()

    # ADD THESE FIELDS:
    channel_author = models.TextField(blank=True, null=True)
    channel_summary = models.TextField(blank=True, null=True)
    channel_image_url = models.TextField(blank=True, null=True)
    language = models.CharField(max_length=50, blank=True, null=True)
    translated = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'channelstranslations'

    def __str__(self):
        return self.channel_title
        


class EpisodeTranslations(models.Model):
    episode = models.ForeignKey(
        Episode,
        on_delete=models.CASCADE,
        db_column='episode_id',
        related_name='episode_translations',
        blank=True,
        null=True
    )
    episode_title = models.TextField()
    sanitized_episode_title = models.TextField()
    publication_date = models.DateTimeField(blank=True, null=True)
    duration = models.TextField(blank=True, null=True)
    episode_number = models.IntegerField(blank=True, null=True)
    explicit = models.BooleanField(blank=True, null=True)
    guid = models.TextField(unique=True)
    audio_url = models.TextField(blank=True, null=True)
    image_url = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    categories = models.JSONField(blank=True, null=True)
    language = models.CharField(max_length=50, blank=True, null=True)
    tsv_transcript = SearchVectorField(null=True, blank=True)
    translated = models.BooleanField(blank=True, null=True)
    search_vector = SearchVectorField(null=True, editable=False)  # Full-text search vector for translations

    class Meta:
        managed = True
        db_table = 'episodestranslations'
        unique_together = ('episode', 'episode_title')
        ordering = ['-publication_date']
        indexes = [
            GinIndex(fields=['search_vector'], name='eptrans_search_vector_gin'),
            GinIndex(fields=['episode_title'], name='eptrans_title_trgm_gin', opclasses=['gin_trgm_ops']),
            GinIndex(fields=['description'],    name='eptrans_desc_trgm_gin', opclasses=['gin_trgm_ops']),
        ]

    def __str__(self):
        return self.episode_title

    
    def clean(self):
        super().clean()
        if self.categories:
            if not isinstance(self.categories, (list, dict)):
                raise ValidationError({'categories': 'Categories must be a list or dictionary.'})
            if isinstance(self.categories, list):
                for category in self.categories:
                    if not isinstance(category, str):
                        raise ValidationError({'categories': 'Each category must be a string.'})


class TranscriptTranslations(models.Model):
    episodetranslations = models.ForeignKey(
        EpisodeTranslations,
        on_delete=models.CASCADE,
        db_column='episode_id',  # This tells Django to use the "episode_id" column
        related_name='transcriptstranslations',
        blank=True,
        null=True
    )
    segment_time = models.TextField()
    segment_text = models.TextField()
    speaker = models.CharField(max_length=50, blank=True, null=True)
    language = models.CharField(max_length=50, blank=True, null=True)
    
    class Meta:
        managed = False
        db_table = 'transcriptstranslations'

    def __str__(self):
        return f"{self.speaker} at {self.segment_time}"




class ChapterTranslations(models.Model):
    episodetranslations = models.ForeignKey(
        EpisodeTranslations,
        on_delete=models.CASCADE,
        db_column='episode_id',  # Use the same column as defined in your DB
        related_name='chapterstranslations',
        blank=True,
        null=True
    )
    chapter_title = models.TextField()
    chapter_time = models.CharField(max_length=10)
    language = models.CharField(max_length=50, blank=True, null=True)
    
    class Meta:
        managed = False
        db_table = 'chapterstranslations'

    def __str__(self):
        return f"{self.chapter_title} at {self.chapter_time}"
    
class ChannelInteraction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name="channel_interactions"
    )
    channel = models.ForeignKey(
        Channel, 
        on_delete=models.CASCADE, 
        related_name="channel_interactions"
    )
    followed = models.BooleanField(default=False)
    notifications_enabled = models.BooleanField(default=False)
    rating = models.PositiveSmallIntegerField(
        null=True, 
        blank=True, 
        help_text="Rating from 1 (lowest) to 5 (highest)"
    )

    class Meta:
        unique_together = ('user', 'channel')
        verbose_name = "Channel Interaction"
        verbose_name_plural = "Channel Interactions"

    def __str__(self):
        return f"{self.user.username} - {self.channel.channel_title}"

class EpisodeInteraction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="episode_interactions"
    )
    episode = models.ForeignKey(
        'Episode',  # use a string if Episode is defined later or just Episode if already imported
        on_delete=models.CASCADE,
        related_name="episode_interactions"
    )
    bookmarked = models.BooleanField(default=False)
    rating = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        help_text="Rating from 1 (lowest) to 5 (highest)"
    )

    class Meta:
        unique_together = ('user', 'episode')
        verbose_name = "Episode Interaction"
        verbose_name_plural = "Episode Interactions"

    def __str__(self):
        return f"{self.user.username} - {self.episode.episode_title}"
    
class Comment(models.Model):
    episode = models.ForeignKey('Episode', on_delete=models.CASCADE, related_name='comments')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    # For replies; leave blank for top-level comments.
    parent = models.ForeignKey('self', null=True, blank=True, related_name='replies', on_delete=models.CASCADE)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    seen_by = models.ManyToManyField(settings.AUTH_USER_MODEL, blank=True, related_name="seen_comments")

    def __str__(self):
        return f"{self.user.username}: {self.text[:50]}"

    def reaction_counts(self):
        # Returns a dictionary with counts for each reaction type.
        counts = self.reactions.values('reaction').annotate(count=models.Count('reaction'))
        reaction_data = {key: 0 for key in ['like', 'dislike', 'heart', 'laugh']}
        for item in counts:
            reaction_data[item['reaction']] = item['count']
        return reaction_data
    
# Proxy model for replies.s
class Reply(Comment):
    class Meta:
        proxy = True
        verbose_name = "Reply"
        verbose_name_plural = "Replies"

    def __str__(self):
        return f"Reply by {self.user.username} on {self.episode}"

class CommentReaction(models.Model):
    REACTION_CHOICES = (
        ('like', 'Like'),
        ('dislike', 'Dislike'),
        ('heart', 'Heart'),
        ('laugh', 'Laugh'),
    )
    comment = models.ForeignKey(Comment, on_delete=models.CASCADE, related_name='reactions')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    reaction = models.CharField(max_length=10, choices=REACTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # A user may react only once per comment per reaction type.
        unique_together = (('comment', 'user', 'reaction'),)

    def __str__(self):
        return f"{self.user.username} {self.reaction} on comment {self.comment.id}"    
    


def generate_ticket_number(length=8):
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choices(chars, k=length))
        if not SupportTicket.objects.filter(ticket_number=code).exists():
            return code

# Store attachments under ~/podcast_data/support_attachments
SUPPORT_ATTACHMENTS_ROOT = str(Path.home() / 'podcast_data' / 'support_attachments')
SUPPORT_ATTACHMENTS_URL  = '/media/support_attachments/'

support_attachment_storage = FileSystemStorage(
    location=SUPPORT_ATTACHMENTS_ROOT,
    base_url=SUPPORT_ATTACHMENTS_URL,
)

class SupportTicket(models.Model):
    ticket_number      = models.CharField(
                             max_length=8,
                             unique=True,
                             null=True,
                             blank=True,
                             editable=False,
                         )
    user               = models.ForeignKey(
                             settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE,
                             related_name='support_tickets'
                         )
    subject            = models.CharField(max_length=255)
    message            = models.TextField(max_length=3000)
    submission_date    = models.DateTimeField(auto_now_add=True)
    last_reviewed_date = models.DateTimeField(null=True, blank=True)
    status             = models.CharField(
                             max_length=20,
                             choices=[
                                 ('pending',     'Pending'),
                                 ('in_progress', 'In Progress'),
                                 ('completed',   'Completed'),
                                 ('cancelled',   'Cancelled'),
                             ],
                             default='pending'
                         )
    

    def save(self, *args, **kwargs):
        if not self.ticket_number:
            self.ticket_number = generate_ticket_number()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.ticket_number} {self.subject}"

class SupportTicketAttachment(models.Model):
    ticket = models.ForeignKey(
                 SupportTicket,
                 on_delete=models.CASCADE,
                 related_name='attachments'
             )
    file   = models.FileField(storage=support_attachment_storage)

    def __str__(self):
        return os.path.basename(self.file.name)