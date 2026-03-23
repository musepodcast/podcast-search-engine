# podcasts/admin.py

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.safestring import mark_safe
from django.db.models import Sum, Count, Q, F
from .models import (
    Channel, Episode, Transcript,
    Chapter, CustomUser, ChannelVisit,
    EpisodeVisit, SearchQuery, ChannelInteraction, 
    EpisodeInteraction, Comment, Reply, 
    SupportTicket, SupportTicketAttachment, ChannelSearchQuery, EpisodeAssistantQuery,
    EpisodeDownload, EpisodeShare
)
from django.utils import timezone
from axes.handlers.proxy import AxesProxyHandler  
from django.db.models import OuterRef, Exists
from datetime import timedelta
from allauth.account.models import EmailAddress
from django.utils.html import format_html_join

@admin.register(Channel)
class ChannelAdmin(admin.ModelAdmin):
    list_display = ('id', 'channel_title', 'sanitized_channel_title')
    search_fields = ('channel_title',)

@admin.register(Episode)
class EpisodeAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'episode_title', 'channel', 'publication_date', 'guid',
        'total_views', 'total_downloads', 'total_shares',
    )
    search_fields = ('episode_title', 'channel__channel_title', 'guid')
    list_filter = ('channel', 'publication_date', 'explicit')
    ordering = ('-publication_date',)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        # annotate from EpisodeVisit, EpisodeDownload, EpisodeShare
        qs = qs.annotate(
            _views=Sum('episodevisit__count'),
            _downloads=Sum('downloads__count'),
            _shares=Sum('shares__count'),
        )
        return qs

    @admin.display(ordering='_views', description='Views')
    def total_views(self, obj):
        return obj._views or 0

    @admin.display(ordering='_downloads', description='Downloads')
    def total_downloads(self, obj):
        return obj._downloads or 0

    @admin.display(ordering='_shares', description='Shares')
    def total_shares(self, obj):
        return obj._shares or 0


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

@admin.action(description="Purge unverified, inactive users older than 1 day")
def purge_unverified_action(modeladmin, request, queryset):
    cutoff = timezone.now() - timedelta(days=1)
    qs = (
        queryset.filter(
            is_active=False,
            is_staff=False,
            is_superuser=False,
            last_login__isnull=True,
            date_joined__lt=cutoff,
        )
        .annotate(
            has_verified_email=Exists(
                EmailAddress.objects.filter(user=OuterRef("pk"), verified=True)
            )
        )
        .filter(has_verified_email=False)
    )
    count = qs.count()
    qs.delete()
    modeladmin.message_user(request, f"Deleted {count} unverified account(s).")

@admin.action(description="Preview purge: list unverified, inactive users older than 1 day (no delete)")
def preview_purge_unverified_action(modeladmin, request, queryset):
    cutoff = timezone.now() - timedelta(days=1)
    qs = (
        queryset.filter(
            is_active=False,
            is_staff=False,
            is_superuser=False,
            last_login__isnull=True,
            date_joined__lt=cutoff,
        )
        .annotate(
            has_verified_email=Exists(
                EmailAddress.objects.filter(user=OuterRef("pk"), verified=True)
            )
        )
        .filter(has_verified_email=False)
    )

    count = qs.count()
    # Show up to 20 usernames/emails in the admin message, then “+N more…”
    sample = list(qs.values_list("username", "email")[:20])
    if sample:
        lines = format_html_join("", "<li>{} ({})</li>", ((u or "—", e or "—") for u, e in sample))
        more = "" if count <= 20 else f"<br>…and {count - 20} more."
        modeladmin.message_user(
            request,
            mark_safe(f"<p><strong>{count}</strong> account(s) would be purged:</p><ul>{lines}</ul>{more}"),
            level="WARNING",
        )
    else:
        modeladmin.message_user(request, "No accounts would be purged.", level="INFO")

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    actions = [preview_purge_unverified_action, purge_unverified_action, "unlock_login_attempts"]

    #actions = [purge_unverified_action, "unlock_login_attempts"]

    list_display = (
        'id', 'username', 'email', 'first_name', 'last_name',
        'birthdate', 'country', 'phone_number', 'gender', 'is_staff', 'support_ticket_limit'
    )
    search_fields = ('username', 'email', 'first_name', 'last_name')
    
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {
            'fields': (
                'first_name', 'last_name', 'email', 
                'birthdate', 'country', 'phone_number', 'gender', 'support_ticket_limit'
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

    def unlock_login_attempts(self, request, queryset):
        """
        Clear Axes attempts for selected users so they can log in immediately.
        Works whether they log in by username or email.
        """
        count = 0
        for user in queryset:
            # Reset by username
            AxesProxyHandler.reset_attempts(username=user.username)
            # Reset by email (in case they log in with email)
            if user.email:
                AxesProxyHandler.reset_attempts(username=user.email)
            count += 1
        self.message_user(request, f"Unlocked Axes attempts for {count} user(s).")
    unlock_login_attempts.short_description = "Unlock login attempts (Axes)"

    def delete_queryset(self, request, queryset):
        # Call instance.delete() on each to trigger your override
        for user in queryset:
            user.delete()


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

@admin.register(ChannelSearchQuery)
class ChannelSearchQueryAdmin(admin.ModelAdmin):
    list_display = ('channel', 'query', 'who', 'count', 'last_searched', 'language', 'ip_address')
    list_filter  = ('channel', 'language')
    search_fields = ('query', 'user__username', 'channel__channel_title', 'ip_address')

    @admin.display(description='User')
    def who(self, obj):
        return obj.user.username if obj.user_id else 'guest'

@admin.register(EpisodeAssistantQuery)
class EpisodeAssistantQueryAdmin(admin.ModelAdmin):
    list_display = ('episode', 'query', 'who', 'count', 'last_asked', 'language', 'model_name', 'ip_address')
    list_filter = ('user', 'language', 'model_name', 'episode__channel')
    search_fields = ('query', 'user__username', 'episode__episode_title', 'ip_address')
    autocomplete_fields = ('user', 'episode')
    ordering = ('-last_asked',)

    @admin.display(description='User')
    def who(self, obj):
        return obj.user.username if obj.user_id else 'guest'

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

class SupportTicketAttachmentInline(admin.TabularInline):
    model = SupportTicketAttachment
    extra = 0

@admin.register(SupportTicket)
class SupportTicketAdmin(admin.ModelAdmin):
    list_display       = (
        'ticket_number',   # ← use this
        'user','user_email','subject','status',
        'submission_date','last_reviewed_date'
    )
    list_display_links = ('ticket_number','subject')
    list_editable      = ('status',)
    list_filter        = ('status',)
    search_fields      = ('subject','message','user__username','user__email')
    inlines            = [SupportTicketAttachmentInline]

    def user_email(self, obj):
        return obj.user.email
    user_email.admin_order_field = 'user__email'
    user_email.short_description = 'Email'

    def save_model(self, request, obj, form, change):
        # if editing an existing ticket AND status was changed…
        if change and 'status' in form.changed_data:
            obj.last_reviewed_date = timezone.now()
        super().save_model(request, obj, form, change)

@admin.register(EpisodeDownload)
class EpisodeDownloadAdmin(admin.ModelAdmin):
    list_display  = (
        'user', 'episode', 'language', 'count',
        'last_downloaded', 'last_ip_address',
    )
    list_filter   = ('language', 'episode__channel',)
    search_fields = ('user__username', 'episode__episode_title', 'last_ip_address', 'last_user_agent', 'last_filename')
    autocomplete_fields = ('user', 'episode')
    ordering = ('-last_downloaded',)

@admin.register(EpisodeShare)
class EpisodeShareAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'episode', 'count',
        'last_shared', 'last_ip_address',
    )
    list_filter = ('episode__channel',)
    search_fields = (
        'user__username',
        'episode__episode_title',
        'last_ip_address',
        'last_user_agent',
    )
    ordering = ('-last_shared',)
    autocomplete_fields = ('user', 'episode')
