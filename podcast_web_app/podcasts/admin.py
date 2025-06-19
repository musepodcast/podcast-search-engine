# podcasts/admin.py

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.safestring import mark_safe
from .models import Channel, Episode, Transcript, Chapter, CustomUser, ChannelVisit, EpisodeVisit, SearchQuery, ChannelInteraction, EpisodeInteraction, Comment, Reply

@admin.register(Channel)
class ChannelAdmin(admin.ModelAdmin):
    list_display = ('id', 'channel_title', 'sanitized_channel_title')
    search_fields = ('channel_title',)

@admin.register(Episode)
class EpisodeAdmin(admin.ModelAdmin):
    list_display = ('id', 'episode_title', 'channel', 'publication_date', 'guid')
    search_fields = ('episode_title', 'channel__channel_title', 'guid')
    list_filter = ('channel', 'publication_date', 'explicit')

@admin.register(Transcript)
class TranscriptAdmin(admin.ModelAdmin):
    list_display = ('id', 'episode', 'segment_time', 'speaker')
    search_fields = ('episode__episode_title', 'speaker', 'segment_text')
    list_filter = ('speaker',)

@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ('id', 'episode', 'chapter_title', 'chapter_time')
    search_fields = ('episode__episode_title', 'chapter_title')
    list_filter = ('episode',)

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = (
        'id', 'username', 'email', 'first_name', 'last_name',
        'birthdate', 'country', 'phone_number', 'gender', 'is_staff'
    )
    search_fields = ('username', 'email', 'first_name', 'last_name')
    
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {
            'fields': (
                'first_name', 'last_name', 'email', 
                'birthdate', 'country', 'phone_number', 'gender'
            )
        }),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser',
                                       'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': (
                'username', 'email', 'first_name', 'last_name',
                'birthdate', 'country', 'phone_number', 'gender',
                'password1', 'password2'
            ),
        }),
    )


@admin.register(ChannelVisit)
class ChannelVisitAdmin(admin.ModelAdmin):
    list_display = ('user', 'channel', 'count', 'first_visited', 'last_visited', 'last_ip_address')
    search_fields = ('user__username', 'channel__channel_title')

@admin.register(EpisodeVisit)
class EpisodeVisitAdmin(admin.ModelAdmin):
    list_display = ('user', 'episode', 'count', 'first_visited', 'last_visited', 'last_ip_address')
    search_fields = ('user__username', 'episode__episode_title')

@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'query', 'user', 'search_in', 'search_date',
        'count', 'first_searched', 'last_searched', 'ip_address'
    )
    search_fields = ('query', 'user__username', 'search_in')

@admin.register(ChannelInteraction)
class ChannelInteractionAdmin(admin.ModelAdmin):
    list_display = ('user', 'channel', 'followed', 'notifications_enabled', 'rating')
    search_fields = ('user__username', 'channel__channel_title')
    list_filter = ('followed', 'notifications_enabled', 'rating')

@admin.register(EpisodeInteraction)
class EpisodeInteractionAdmin(admin.ModelAdmin):
    list_display = ('user', 'episode', 'bookmarked', 'rating')
    search_fields = ('user__username', 'episode__episode_title')
    list_filter = ('bookmarked', 'rating')

@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ('id', 'episode', 'user', 'created_at', 'parent', 'formatted_text')
    search_fields = ('user__username', 'text')
    list_filter = ('created_at',)
    
    def formatted_text(self, obj):
        # Use the highlight_mentions filter (make sure your custom filters are loaded in admin templates)
        #from django.template.loader import render_to_string
        # Alternatively, you can use the custom filter directly:
        from podcasts.templatetags import custom_filters
        return custom_filters.highlight_mentions(obj.text)
    formatted_text.short_description = "Text"

# Admin for Replies (proxy model)
@admin.register(Reply)
class ReplyAdmin(admin.ModelAdmin):
    list_display = ('id', 'episode', 'user', 'created_at', 'formatted_text', 'tagged_users')
    search_fields = ('user__username', 'text')
    list_filter = ('created_at',)
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.filter(text__icontains='@')
    
    def formatted_text(self, obj):
        return mark_safe(obj.text)
    formatted_text.short_description = "Text"
    
    def tagged_users(self, obj):
        import re
        tagged = re.findall(r'@(\w+)', obj.text)
        return ", ".join(tagged) if tagged else "-"
    tagged_users.short_description = "Tagged Users"
