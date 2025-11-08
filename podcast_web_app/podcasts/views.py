# podcast_web_app/podcasts/views.py

from django.views.generic import ListView, DetailView, TemplateView
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import (
    Q, Prefetch, Avg, 
    Count, Sum, F, 
    OuterRef, Subquery, Value, 
    IntegerField, FloatField
)
from pathlib import Path
from django.db.models.functions import Coalesce
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.timesince import timesince
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy as _lazy
from .forms import CustomUserCreationForm
from .models import CustomUser
from allauth.account.models import EmailAddress
from django.conf import settings
import logging, time
from collections import Counter
import re, difflib, unicodedata
import sys
import json
import itertools
from django.http import Http404, HttpResponse, HttpResponseForbidden, JsonResponse, FileResponse
from django.utils.encoding import iri_to_uri
from .models import (
    Channel, ChannelTranslations, ChannelVisit,
    Episode, EpisodeTranslations, EpisodeVisit,
    Transcript, TranscriptTranslations,
    Chapter, ChapterTranslations,
    SearchQuery, ChannelInteraction, EpisodeInteraction,
    Comment, CommentReaction, Reply,
    SupportTicket, SupportTicketAttachment, ChannelSearchQuery,
    EpisodeDownload, EpisodeShare

)
from .filters import EpisodeFilter
from django.core.paginator import Paginator, EmptyPage

from .forms import CustomAuthenticationForm, SupportTicketForm
from .models import SupportTicketAttachment
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic.edit import UpdateView, CreateView
from .forms import UserProfileForm, CustomUserCreationForm, Disable2FAForm, OTPChallengeForm
from django.contrib.auth import get_user_model

from two_factor.views.core import LoginView as BaseTwoFactorLoginView

from django.contrib.auth import login
from django.views import View

from django.contrib import messages
from django.urls import reverse
from django_otp.plugins.otp_totp.models import TOTPDevice

from two_factor.views.core import LoginView as TwoFactorLoginView

from django.contrib.auth.views import LoginView
from django.contrib.postgres.search import SearchVector, SearchQuery as PgSearchQuery, SearchRank, TrigramSimilarity
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST, require_GET, require_http_methods

from podcasts.search.documents import EpisodeDocument, TranscriptDocument
from elasticsearch_dsl import Q as ES_Q
from elasticsearch_dsl.connections import connections

from django.contrib.postgres.search import TrigramSimilarity
from datetime import datetime, timedelta
from axes.models import AccessAttempt
from axes.conf import settings as axes_settings
from django.utils import timezone as tz
from django.utils.timezone import now

# podcasts/views.py
from datetime import timedelta
from django.shortcuts import render
from django.utils import timezone
from axes.models import AccessAttempt
from axes.conf import settings as axes_settings
from django.db import transaction

_slug_non_alnum = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

def slug_norm(s: str) -> str:
    """Diacritic-insensitive, lower, non-alnum→_ normalizer for episode slugs."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _slug_non_alnum.sub("_", s)
    return re.sub(r"_+", "_", s).strip("_")

@require_POST
def episode_share_ping(request, episode_id):
    episode = get_object_or_404(Episode, pk=episode_id)

    user = request.user if request.user.is_authenticated else None
    ip   = request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip() or request.META.get('REMOTE_ADDR')
    ua   = (request.META.get('HTTP_USER_AGENT') or '')[:1000]

    # aggregate per (user, episode); guests all go to the same row (user=None)
    share, _ = EpisodeShare.objects.get_or_create(
        user=user,
        episode=episode,
        defaults={'count': 0}
    )
    EpisodeShare.objects.filter(pk=share.pk).update(
        count=F('count') + 1,
        last_shared=timezone.now(),
        last_ip_address=ip,
        last_user_agent=ua,
    )

    # return total shares for this episode (all users)
    total_shares = (
        EpisodeShare.objects
        .filter(episode=episode)
        .aggregate(total=Coalesce(Sum('count'), 0))['total']
    )

    return JsonResponse({
        "ok": True,
        "total_shares": total_shares,
    })

def episode_download_json_file(request, sanitized_channel_title, sanitized_episode_title):
    """
    Streams the episode JSON from disk and records a download row.
    Handles locale aliases (pt-br -> pt), filename suffixes (<title>_<lang>.json),
    and diacritic-insensitive matching (e.g., Português vs Portugues).
    Returns JSON 404 (not HTML) when missing so browsers don't save .htm files.
    """
    # -------------------- helpers --------------------
    _ALIAS_TO_CODE = {
        "en": "en", "en-us": "en", "en_gb": "en", "en-gb": "en",
        "pt": "pt", "pt-br": "pt", "pt_br": "pt",
        "es": "es", "es-mx": "es", "es_es": "es",
        "it": "it", "fr": "fr", "ru": "ru", "uk": "uk",
        "zh": "cn", "zh-cn": "cn", "zh_hans": "cn", "zh-hans": "cn",
        "zh-tw": "tw", "zh_hant": "tw", "zh-hant": "tw",
        "cn": "cn", "tw": "tw",
        "ko": "ko", "ja": "ja", "tr": "tr", "de": "de",
        "ar": "ar", "hi": "hi", "vi": "vi", "tl": "tl",
    }
    def _canon_lang(lang: str) -> str:
        if not lang:
            return "en"
        s = lang.strip().lower().replace("_", "-")
        return _ALIAS_TO_CODE.get(s, _ALIAS_TO_CODE.get(s.split("-")[0], s.split("-")[0]))

    _norm_non_alnum = re.compile(r"[^a-z0-9]+")
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "")
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.lower()
        s = _norm_non_alnum.sub("_", s)
        return re.sub(r"_+", "_", s).strip("_")

    def _truncate_tokens_starts(candidate: str, max_drop: int = 2):
        parts = candidate.split("_")
        yield candidate
        for k in range(1, min(max_drop, max(0, len(parts)-1)) + 1):
            yield "_".join(parts[:len(parts)-k])

    def _client_ip(req):
        xff = req.META.get('HTTP_X_FORWARDED_FOR')
        return (xff.split(',')[0].strip() if xff else req.META.get('REMOTE_ADDR'))

    # -------------------- language & base dirs --------------------
    req_lang = (request.GET.get("lang")
                or getattr(request, "LANGUAGE_CODE", None)
                or "en")
    code = _canon_lang(req_lang)  # e.g., en-us -> en, pt-br -> pt

    base_dir = Path(settings.EPISODE_JSON_BASE)  # e.g., C:\Users\isaac\podcast_data\transcripts
    chan_dir = base_dir / sanitized_channel_title

    tried = []
    if not chan_dir.exists():
        return JsonResponse({"error": "Channel folder not found",
                             "tried": [str(chan_dir)]},
                            status=404)

    # Build candidate language dirs:
    #  - canonical code (e.g. "en")
    #  - any subdir that equals code or starts with "code-" or "code_"
    lang_dirs = []
    primary = chan_dir / code
    if primary.exists():
        lang_dirs.append(primary)

    for p in chan_dir.iterdir():
        if p.is_dir() and (p.name == code or p.name.startswith(f"{code}-") or p.name.startswith(f"{code}_")):
            if p not in lang_dirs:
                lang_dirs.append(p)

    # if still empty and code != 'en', fallback to plain 'en'
    if not lang_dirs and code != "en":
        en_dir = chan_dir / "en"
        if en_dir.exists():
            lang_dirs.append(en_dir)

    # helper: try to locate a file inside a directory
    def _try_dir(d: Path, served_code: str):
        # 1) exact
        exact = d / f"{sanitized_episode_title}.json"
        tried.append(str(exact))
        if exact.exists():
            return exact, served_code

        # 2) suffixed with canonical code
        suff = d / f"{sanitized_episode_title}_{served_code}.json"
        tried.append(str(suff))
        if suff.exists():
            return suff, served_code

        # 3) heuristic scan (accent-/punctuation-insensitive)
        if d.exists():
            target_norm = _norm(sanitized_episode_title)
            targets = list(_truncate_tokens_starts(target_norm, max_drop=2))
            for f in d.glob("*.json"):
                stem_norm = _norm(f.stem)
                m = re.match(r"^(.*)_(\w{2,3})$", stem_norm)
                # If filename has a language suffix, strip it ONLY if it matches served_code canonically
                f_base = m.group(1) if (m and _canon_lang(m.group(2)) == served_code) else stem_norm
                if any(f_base == t or f_base.startswith(t) or t.startswith(f_base) for t in targets):
                    return f, served_code
            tried.append(f"[scanned] {str(d)}\\*.json")
        return None, served_code

    # -------------------- search candidate dirs --------------------
    for d in lang_dirs:
        # served_code is canonical of req_lang regardless of folder's exact name
        hit, served_code = _try_dir(d, code)
        if hit:
            return _serve_and_log(request, hit, served_code,
                                  sanitized_channel_title, sanitized_episode_title)

    # last fallback: plain 'en' if not already tried
    if code != "en":
        en_dir = chan_dir / "en"
        if en_dir.exists() and en_dir not in lang_dirs:
            hit, served_code = _try_dir(en_dir, "en")
            if hit:
                return _serve_and_log(request, hit, served_code,
                                      sanitized_channel_title, sanitized_episode_title)

    return JsonResponse({"error": "Episode JSON not found", "tried": tried}, status=404)


def _serve_and_log(request, file_path: Path, served_code: str,
                   sanitized_channel_title: str, sanitized_episode_title: str):
    """
    Internal helper: logs the download (EpisodeDownload) and streams the file.
    """
    # Pick a friendly download name
    download_name = f"{sanitized_channel_title}__{served_code}__{sanitized_episode_title}.json"

    # Resolve Episode for logging; if not found by base slug, try translations
    episode_obj = Episode.objects.filter(
        channel__sanitized_channel_title=sanitized_channel_title,
        sanitized_episode_title=sanitized_episode_title
    ).select_related('channel').first()

    if not episode_obj:
        tr = EpisodeTranslations.objects.select_related('episode', 'episode__channel').filter(
            episode__channel__sanitized_channel_title=sanitized_channel_title,
            sanitized_episode_title=sanitized_episode_title,
            translated=True
        ).first()
        if tr:
            episode_obj = tr.episode

    # Gather request metadata
    try:
        file_size = file_path.stat().st_size
    except Exception:
        file_size = None

    user = request.user if getattr(request, "user", None) and request.user.is_authenticated else None
    ip   = request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip() or request.META.get('REMOTE_ADDR')
    ua   = (request.META.get('HTTP_USER_AGENT') or '')[:1000]

    # Aggregate/update a single row per (user, episode, language)
    if episode_obj:
        dl, _ = EpisodeDownload.objects.get_or_create(
            user=user,
            episode=episode_obj,
            language=served_code,
            defaults={"count": 0}
        )
        EpisodeDownload.objects.filter(pk=dl.pk).update(
            count=F('count') + 1,
            last_downloaded=now(),
            last_ip_address=ip,
            last_user_agent=ua,
            last_filename=download_name,
            last_file_path=str(file_path),
            bytes_served=file_size,
        )

    # Stream the file
    return FileResponse(
        open(file_path, "rb"),
        as_attachment=True,
        filename=iri_to_uri(download_name),
        content_type="application/json",
    )

def _get_client_ip(request):
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    return xff.split(",")[0].strip() if xff else request.META.get("REMOTE_ADDR", "")

def _cooloff_td():
    c = axes_settings.AXES_COOLOFF_TIME
    if isinstance(c, timedelta):
        return c
    try:
        return timedelta(hours=float(c))  # Axes treats int as hours
    except Exception:
        return timedelta(hours=1)

def locked_out_view(request):
    ip = _get_client_ip(request)
    username = request.POST.get("username") or request.GET.get("username") or ""

    qs = AccessAttempt.objects.all()
    # Prefer exact matches; fall back progressively
    candidates = (
        qs.filter(ip_address=ip, username=username)
        or qs.filter(ip_address=ip)
        or qs
    )

    limit = axes_settings.AXES_FAILURE_LIMIT

    # Pick the attempt that actually triggered lock; fall back to most recent
    attempt = (
        candidates.filter(failures_since_start__gte=limit)
                  .order_by("-attempt_time")
                  .first()
        or candidates.order_by("-attempt_time").first()
    )

    remaining_seconds = 0
    unlock_at = None

    if attempt:
        base_time = attempt.attempt_time  # updated on each failure in your version
        # Make sure it's timezone-aware
        if timezone.is_naive(base_time):
            base_time = timezone.make_aware(base_time, timezone.get_current_timezone())
        unlock_at = base_time + _cooloff_td()
        diff = int((unlock_at - timezone.now()).total_seconds())
        if diff > 0:
            remaining_seconds = diff

    def fmt(seconds: int) -> str:
        m, s = divmod(max(0, seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    context = {
        "remaining_seconds": remaining_seconds,
        "eta_readable": fmt(remaining_seconds),
        "unlock_at": timezone.localtime(unlock_at) if unlock_at else None,
        "support_email": "musepodcasthelp@gmail.com",
    }
    return render(request, "security/locked_out.html", context, status=429)


@require_http_methods(["GET", "POST"])
@login_required
def toggle_contribute(request):
    """
    GET  → return whether the user is currently contributing
    POST → flip the flag and set which channels they should support
    """
    user = request.user  # your CustomUser

    if request.method == "GET":
        return JsonResponse({"is_contributing": user.is_contributing})

    # POST: toggle the flag
    user.is_contributing = not user.is_contributing

    # read the channels query param
    raw = request.GET.get("channels", "")
    if "all" in raw or not raw.strip():
        # subscribe to every channel
        user.contribute_channels.set(Channel.objects.all())
    else:
        # parse comma‑separated IDs
        ids = [int(pk) for pk in raw.split(",") if pk.isdigit()]
        user.contribute_channels.set(Channel.objects.filter(pk__in=ids))

    user.save()
    return JsonResponse({"is_contributing": user.is_contributing})

GUEST_USERNAME = "guest"

def get_guest_user():
    User = get_user_model()
    user = User.objects.filter(username=GUEST_USERNAME).first()
    if user:
        return user
    # Create a disabled guest account the first time we need it
    with transaction.atomic():
        user, created = User.objects.get_or_create(
            username=GUEST_USERNAME,
            defaults={
                "email": "guest@example.invalid",  # non-routable TLD
                "is_active": False,
            }
        )
        if created:
            try:
                user.set_unusable_password()
                user.save(update_fields=["password"])
            except Exception:
                pass
    return user

User = get_user_model()

class RepliesListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/replies_list.html'
    context_object_name = 'replies'
    paginate_by = 10

    def get_queryset(self):
        username = self.request.user.username
        # Filter replies that mention the user. Adjust as needed (e.g., you might also require replies to have a non-null parent)
        qs = Reply.objects.filter(text__icontains='@' + username).order_by('-created_at')
        return qs

    def get(self, request, *args, **kwargs):
        # First, get the queryset of replies (those that mention the user)
        qs = self.get_queryset()
        # Mark each reply as seen by adding the current user into the seen_by ManyToManyField.
        for reply in qs:
            reply.seen_by.add(request.user)
        return super().get(request, *args, **kwargs)
    
@require_GET
def search_users(request):
    query = request.GET.get('q', '').strip()
    User = get_user_model()
    if query:
        # For instance, filter usernames that start with the query (case-insensitive)
        users = User.objects.filter(username__istartswith=query)[:10]
        usernames = [user.username for user in users]
    else:
        usernames = []
    return JsonResponse({'usernames': usernames})

def process_mentions(text, comment):
    """
    Finds all @mentions in text, and for each valid username,
    wraps it in a span (or triggers a notification).
    """
    # Find all occurrences after an '@' consisting of one or more word characters.
    mentioned_usernames = re.findall(r'@(\w+)', text)
    for username in set(mentioned_usernames):
        try:
            user_obj = User.objects.get(username=username)
            # Optional: Create a notification for the mentioned user here.
            # Replace plain @username with a highlighted version.
            text = re.sub(r'@' + re.escape(username) + r'\b', 
                          f'<span class="mention-highlight">@{username}</span>', text)
        except User.DoesNotExist:
            # If no user exists, leave the text as is.
            pass
    return text


@login_required
def post_comment(request, episode_id):
    """
    Create a new top-level comment or reply for an episode.
    Expects JSON: { "text": "Your comment", "parent_id": optional }
    Processes @mentions in the comment text.
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        parent_id = data.get('parent_id')
        if not text:
            return JsonResponse({'error': 'Empty comment text.'}, status=400)
        episode = get_object_or_404(Episode, id=episode_id)
        comment = Comment.objects.create(
            episode=episode,
            user=request.user,
            text=text,
            parent_id=parent_id  # Will be None for top-level comments.
        )
        # Process @mentions in the comment text.
        processed_text = process_mentions(comment.text, comment)
        comment.text = processed_text
        comment.save()

        response_data = {
            'comment_id': comment.id,
            'username': comment.user.username,
            'text': comment.text,
            'created_at': timesince(comment.created_at) + " ago",
            'reactions': comment.reaction_counts(),
            'replies': []  # Replies can be filled by further AJAX calls.
        }
        return JsonResponse(response_data)
    return JsonResponse({'error': 'Invalid request.'}, status=400)

@login_required
def comment_reaction(request, comment_id):
    """
    Toggle a reaction (like, dislike, heart, laugh) for a given comment.
    Expects JSON: { "reaction": "like" }
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        reaction_type = data.get('reaction')
        if reaction_type not in ['like', 'dislike', 'heart', 'laugh']:
            return JsonResponse({'error': 'Invalid reaction.'}, status=400)
        comment = get_object_or_404(Comment, id=comment_id)
        reaction, created = CommentReaction.objects.get_or_create(
            user=request.user,
            comment=comment,
            reaction=reaction_type
        )
        if not created:
            # If reaction exists, remove it (toggle off).
            reaction.delete()
        # Return updated reaction counts.
        return JsonResponse({'reactions': comment.reaction_counts()})
    return JsonResponse({'error': 'Invalid request.'}, status=400)

@login_required
def get_comments(request, episode_id):
    """
    Return a JSON list of comments (top-level) for an episode.
    For each comment, include basic info and reaction counts.
    """
    episode = get_object_or_404(Episode, id=episode_id)
    comments = episode.comments.filter(parent__isnull=True).order_by('-created_at')
    comments_list = []
    for comment in comments:
        processed = process_mentions(comment.text, comment)
        comments_list.append({
            'comment_id': comment.id,
            'username': comment.user.username,
            'text': processed,
            'created_at': timesince(comment.created_at) + " ago",
            'reactions': comment.reaction_counts(),
            # For now, replies can be handled later.
            'replies': []
        })
    return JsonResponse({'comments': comments_list})

# Toggle Episode Bookmark
@login_required
@require_POST
def toggle_episode_bookmark(request, episode_id):
    episode = get_object_or_404(Episode, id=episode_id)
    interaction, created = EpisodeInteraction.objects.get_or_create(
        user=request.user, episode=episode
    )
    interaction.bookmarked = not interaction.bookmarked
    interaction.save()
    return JsonResponse({'bookmarked': interaction.bookmarked})

# Update Episode Rating (expects a 'rating' parameter between 1 and 5)
@login_required
@require_POST
def update_episode_rating(request, episode_id):
    episode = get_object_or_404(Episode, id=episode_id)
    try:
        rating_value = int(request.POST.get('rating', 0))
    except ValueError:
        return JsonResponse({'error': 'Invalid rating value'}, status=400)
    if rating_value < 1 or rating_value > 5:
        return JsonResponse({'error': 'Rating must be between 1 and 5'}, status=400)
    interaction, created = EpisodeInteraction.objects.get_or_create(
        user=request.user, episode=episode
    )
    interaction.rating = rating_value
    interaction.save()
    return JsonResponse({'rating': interaction.rating})

# Toggle Follow Status
@login_required
@require_POST
def toggle_follow(request, channel_id):
    channel = get_object_or_404(Channel, id=channel_id)
    interaction, created = ChannelInteraction.objects.get_or_create(user=request.user, channel=channel)
    interaction.followed = not interaction.followed
    interaction.save()
    return JsonResponse({'followed': interaction.followed})

# Toggle Notifications
@login_required
@require_POST
def toggle_notifications(request, channel_id):
    channel = get_object_or_404(Channel, id=channel_id)
    interaction, created = ChannelInteraction.objects.get_or_create(user=request.user, channel=channel)
    interaction.notifications_enabled = not interaction.notifications_enabled
    interaction.save()
    return JsonResponse({'notifications_enabled': interaction.notifications_enabled})

# Update Rating (expects a 'rating' parameter 1-5)
@login_required
@require_POST
def update_rating(request, channel_id):
    channel = get_object_or_404(Channel, id=channel_id)
    try:
        rating_value = int(request.POST.get('rating', 0))
    except ValueError:
        return JsonResponse({'error': 'Invalid rating value'}, status=400)
    if rating_value < 1 or rating_value > 5:
        return JsonResponse({'error': 'Rating must be between 1 and 5'}, status=400)
    interaction, created = ChannelInteraction.objects.get_or_create(user=request.user, channel=channel)
    interaction.rating = rating_value
    interaction.save()
    return JsonResponse({'rating': interaction.rating})


class TwoFactorChallengeView(View):
    template_name = "registration/login_2fa.html"
    form_class = OTPChallengeForm

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {"form": form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            token = form.cleaned_data["token"]
            user_id = request.session.get("pre_2fa_user_id")
            if not user_id:
                messages.error(request, "Session expired. Please log in again.")
                return redirect("podcasts:login")
            User = get_user_model()
            try:
                user = User.objects.get(pk=user_id)
            except User.DoesNotExist:
                messages.error(request, "User not found.")
                return redirect("podcasts:login")
            devices = TOTPDevice.objects.filter(user=user, confirmed=True)
            if any(device.verify_token(token) for device in devices):
                # Set the authentication backend explicitly
                user.backend = 'django.contrib.auth.backends.ModelBackend'
                login(request, user)
                request.session.pop("pre_2fa_user_id", None)
                return redirect("podcasts:channel_list")
            else:
                form.add_error("token", "Invalid OTP token.")
        return render(request, self.template_name, {"form": form})

class CustomLoginView(LoginView):
    template_name = "registration/login.html"  # Use your existing login template
    authentication_form = CustomAuthenticationForm

    def form_valid(self, form):
        user = form.get_user()
        # Check if the user has a confirmed TOTP device.
        if user.totpdevice_set.filter(confirmed=True).exists():
            self.request.session["pre_2fa_user_id"] = user.pk
            # If the user has chosen to enforce 2FA on every login,
            # redirect to the OTP challenge view unconditionally.
            if user.enforce_2fa:
                # Clear any remember cookie logic if needed
                return redirect("podcasts:two_factor_challenge")
            else:
                # Otherwise, proceed with the normal two-factor flow,
                # which may bypass OTP if the device is recognized.
                return super().form_valid(form)
        else:
            # If no TOTP device exists, log in normally.
            login(self.request, user)
            return redirect(self.get_success_url())

class SecureDisable2FAView(LoginRequiredMixin, View):
    template_name = "podcasts/disable_2fa.html"
    form_class = Disable2FAForm

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {"form": form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            token = form.cleaned_data["token"]
            # Get all confirmed TOTP devices for the user
            devices = TOTPDevice.objects.filter(user=request.user, confirmed=True)
            verified = False
            for device in devices:
                if device.verify_token(token):
                    verified = True
                    break
            if verified:
                # If token is valid, disable 2FA by deleting devices
                devices.delete()
                messages.success(request, "Two‑Factor Authentication has been disabled.")
                return redirect("podcasts:profile")
            else:
                form.add_error("token", "Invalid OTP token. Please try again.")
        return render(request, self.template_name, {"form": form})

class CustomDisable2FAView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        # Remove all TOTP devices for this user to disable 2FA.
        TOTPDevice.objects.filter(user=request.user).delete()
        messages.success(request, "Two‑Factor Authentication has been disabled.")
        return redirect(reverse('podcasts:profile'))

    def get(self, request, *args, **kwargs):
        # Optionally, you can render a confirmation page or simply forbid GET requests.
        return HttpResponseForbidden("GET not allowed. Please use POST.")


class CustomTwoFactorLoginView(BaseTwoFactorLoginView):
    def form_valid(self, form):
        """
        After validating the primary credentials, check if the user has a TOTP device.
        If so, force OTP entry with a custom template; if not, proceed as normal.
        """
        user = form.get_user()
        if user.totpdevice_set.exists():
            # Optionally, you can set a custom template or redirect to a custom OTP view.
            # For example, redirect to a URL that renders a custom otp_totp.html.
            self.template_name = 'two_factor/custom_otp_totp.html'
        else:
            # No OTP device: skip the OTP step.
            return self.login_success(form)
        return super().form_valid(form)

User = get_user_model()

class ProfileUpdateView(LoginRequiredMixin, UpdateView):
    model = User
    form_class = UserProfileForm
    template_name = 'podcasts/profile_edit.html'
    success_url = reverse_lazy('podcasts:profile')  # Redirect back to profile page after success.

    def get_object(self):
        # Ensure the user can only update their own profile.
        return self.request.user

class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = "podcasts/profile.html"

class SignUpView(CreateView):
    form_class = CustomUserCreationForm
    template_name = "podcasts/signup.html"
    # Send the user to your customized “verification sent” page
    success_url = reverse_lazy("account_email_verification_sent")

    def form_valid(self, form):
        # Create the user as inactive until email is confirmed
        user = form.save(commit=False)
        user.is_active = False
        user.save()

        try:
            # Create EmailAddress + send confirmation email
            EmailAddress.objects.add_email(
                request=self.request,
                user=user,
                email=user.email,
                confirm=True,  # send email
                signup=True,   # use the *signup* templates
            )
        except Exception as e:
            # Clean up and show a friendly error
            log.exception("Signup confirmation email failed")
            user.delete()
            msg = "We couldn’t send a confirmation email. Please try again."
            if getattr(settings, "DEBUG", False):
                msg = f"We couldn’t send a confirmation email: {e.__class__.__name__}: {e}"
            form.add_error(None, msg)
            return self.form_invalid(form)

        # Go to the “verification sent” page
        return redirect(self.success_url)


def validate_username(request):
        username = request.GET.get('username', None)
        is_taken = False
        if username:
            is_taken = User.objects.filter(username__iexact=username).exists()
        return JsonResponse({'is_taken': is_taken})
    
logger = logging.getLogger(__name__)





def get_selected_language(request):
    # Use the GET parameter "lang" if provided; otherwise, fall back to request.LANGUAGE_CODE
    return request.GET.get('lang', getattr(request, 'LANGUAGE_CODE', 'en')).lower()



# Function to handle legacy URL redirection
def channel_redirect(request, pk):
    channel = get_object_or_404(Channel, pk=pk)
    return redirect('podcasts:channel_detail', sanitized_channel_title=channel.sanitized_channel_title)


class HomeView(TemplateView):
    template_name = 'podcasts/home.html'


class ChannelListView(ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/channel_list.html'
    context_object_name = 'channels'
    paginate_by = 10

    # Map query params to annotated fields
    ALLOWED_SORTS = {'views', 'favorites', 'notifications', 'stars', 'title', 'episodes'}
    ALLOWED_DIRS = {'asc', 'desc'}

    LABELS = {
        ('views','desc'): _lazy("Most watched"),
        ('views','asc'):  _lazy("Least watched"),
        ('favorites','desc'): _lazy("Most favorited"),
        ('favorites','asc'):  _lazy("Least favorited"),
        ('notifications','desc'): _lazy("Most notified"),
        ('notifications','asc'):  _lazy("Least notified"),
        ('stars','desc'): _lazy("Most stars"),
        ('stars','asc'):  _lazy("Least stars"),
        ('title','asc'):  _lazy("A → Z"),
        ('title','desc'): _lazy("Z → A"),
        ('episodes','desc'): _lazy("Most episodes"),
        ('episodes','asc'):  _lazy("Least episodes"),
    }

    def _parse_sort(self):
        sort = self.request.GET.get('sort', 'views').lower()
        direction = self.request.GET.get('dir', 'desc').lower()
        if sort not in self.ALLOWED_SORTS:
            sort = 'views'
        if direction not in self.ALLOWED_DIRS:
            direction = 'desc'
        return sort, direction

    def get_queryset(self):
        sort, direction = self._parse_sort()
        dir_prefix = '' if direction == 'asc' else '-'

        # Subquery: total views per channel (no duplication from other joins)
        visits_sq = (
            ChannelVisit.objects
            .filter(channel=OuterRef('pk'))
            .values('channel')            # group by channel
            .annotate(total=Sum('count')) # sum the counts
            .values('total')[:1]          # select summed value
        )
        # NEW: total episodes per channel via subquery
        episodes_sq = (
            Episode.objects
            .filter(channel=OuterRef('pk'))
            .values('channel')
            .annotate(c=Count('*'))
            .values('c')[:1]
        )

        qs = (
            Channel.objects
            .annotate(
                total_views=Coalesce(Subquery(visits_sq, output_field=IntegerField()), 0),
                episode_count=Coalesce(Subquery(episodes_sq, output_field=IntegerField()), 0),
                favorites_count=Count(
                    'channel_interactions',
                    filter=Q(channel_interactions__followed=True),
                    distinct=True,
                ),
                notifications_count=Count(
                    'channel_interactions',
                    filter=Q(channel_interactions__notifications_enabled=True),
                    distinct=True,
                ),
                avg_rating=Avg('channel_interactions__rating'),
                rating_count=Count(
                    'channel_interactions__user',
                    filter=Q(channel_interactions__rating__isnull=False),
                    distinct=True,
                ),
            )
        )
        # (optional) sorting by episodes

        if sort == 'views':
            ordering = [f'{dir_prefix}total_views', 'channel_title']
        elif sort == 'favorites':
            ordering = [f'{dir_prefix}favorites_count', 'channel_title']
        elif sort == 'notifications':
            ordering = [f'{dir_prefix}notifications_count', 'channel_title']
        elif sort == 'stars':
            if direction == 'desc':
                ordering = [
                    F('avg_rating').desc(nulls_last=True),
                    F('rating_count').desc(),
                    'channel_title',
                ]
            else:
                ordering = [
                    F('avg_rating').asc(nulls_last=True),
                    F('rating_count').asc(),
                    'channel_title',
                ]
        elif sort == 'episodes':
            ordering = [f'{dir_prefix}episode_count', 'channel_title']
        elif sort == 'title':
            ordering = [f'{dir_prefix}channel_title']
        else:
            ordering = ['-total_views', 'channel_title']

        return qs.order_by(*ordering)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        lang = get_selected_language(self.request)

        # expose sort state to template
        sort, direction = self._parse_sort()
        context['sort'] = sort
        context['dir'] = direction
            # Use the lazy labels so they render in the active language
        context['current_sort_label'] = self.LABELS.get(
            (sort, direction),
            self.LABELS[('views', 'desc')]  # fallback aligns with your default
        )
        
        if lang not in ('en', 'en-us'):
            translations = ChannelTranslations.objects.filter(
                language=lang,
                translated=True,
                sanitized_channel_title__in=[ch.sanitized_channel_title for ch in context['channels']]
            )
            trans_map = {tr.sanitized_channel_title: tr for tr in translations}
            for ch in context['channels']:
                tr = trans_map.get(ch.sanitized_channel_title)
                if not tr:
                    continue
                ch.channel_title   = tr.channel_title
                ch.channel_summary = tr.channel_summary

        context['selected_language'] = lang
        return context

    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax') == '1':
            return render(self.request, 'podcasts/channel_list_items.html', context)
        return super().render_to_response(context, **response_kwargs)

def _norm_lang(request):
    raw = (get_selected_language(request) or 'en').lower()
    # treat all English as base
    if raw.startswith('en'):
        return 'en'
    # collapse es-ES -> es, pt-BR -> pt, etc.
    return raw.split('-', 1)[0]



class ChannelDetailView(DetailView):
    login_url           = reverse_lazy('podcasts:home')
    template_name       = 'podcasts/channel_detail.html'
    context_object_name = 'channel'

    # ---- config for ranking ----
    TITLE_BONUS   = 100
    PER_OCC_BONUS = 10
    STOP_WORDS    = {'the','a','an','of','in','and','or','to','so','for','on','at','by'}

    # ---------- helpers ----------
    def _tokenize(self, q: str):
        toks = [t for t in re.findall(r"\w+", (q or "").lower()) if t and t not in self.STOP_WORDS]
        return toks or ([q.lower()] if q else [])

    def _es_index_name(self):
        try:
            return TranscriptDocument._index._name
        except Exception:
            return getattr(TranscriptDocument, 'Index', object()).name

    def _bool_filter_episode_ids(self, ids):
        """Filter that works if episode_id is keyword in some shards and long in others."""
        ints = [int(i) for i in ids]
        strs = [str(i) for i in ints]
        return {
            "bool": {
                "should": [
                    {"terms": {"episode_id": ints}},
                    {"terms": {"episode_id": strs}},
                ],
                "minimum_should_match": 1,
            }
        }

    def _occurrence_counts(self, candidate_ids, tokens):
        """
        Return {episode_id: count} using ES.
        1) Try scripted_metric for true word-boundary counts across tokens.
        2) Fallback: terms agg of matching segments per episode (under-count but preserves ordering signal).
        """
        if not candidate_ids:
            return {}

        idx    = self._es_index_name()
        client = connections.get_connection()

        filt = self._bool_filter_episode_ids(candidate_ids)
        size_limit = min(len(candidate_ids), int(getattr(settings, "ES_TERMS_AGG_SIZE_LIMIT", 65535)))

        # ---- 1) scripted_metric (exact word-boundary occurrences across tokens) ----
        script = {
            "scripted_metric": {
                "init_script": "state.count = 0;",
                "map_script": """
                    def txt = params._source.segment_text;
                    if (txt == null) return;
                    def lower = txt.toLowerCase();
                    int total = 0;
                    for (t in params.tokens) {
                        def needle = t;
                        int i = 0;
                        int n = needle.length();
                        while (true) {
                            i = lower.indexOf(needle, i);
                            if (i == -1) break;
                            boolean beforeLetter = (i > 0 && Character.isLetter(lower.charAt(i-1)));
                            int j = i + n;
                            boolean afterLetter = (j < lower.length() && Character.isLetter(lower.charAt(j)));
                            if (!beforeLetter && !afterLetter) total += 1;
                            i = i + n;
                        }
                    }
                    state.count += total;
                """,
                "combine_script": "return state.count;",
                "reduce_script": """
                    int sum = 0;
                    for (s in states) sum += (int) s;
                    return sum;
                """,
                "params": {"tokens": tokens},
            }
        }

        body = {
            "size": 0,
            "query": {"bool": {"filter": [filt]}},
            "aggs": {
                "by_ep": {
                    "terms": {"field": "episode_id", "size": size_limit},
                    "aggs": {"occ": script}
                }
            }
        }

        try:
            res = client.search(index=idx, body=body)
            buckets = res.get("aggregations", {}).get("by_ep", {}).get("buckets", [])
            out = {}
            for b in buckets:
                try:
                    eid = int(b.get("key"))
                except Exception:
                    continue
                count = int(b.get("occ", {}).get("value", 0))
                if count > 0:
                    out[eid] = count
            return out
        except Exception as e:
            logger.warning("scripted_metric unavailable; fallback to segment-counts: %s", e)

        # ---- 2) fallback: count matching segments per episode ----
        # use a simple match over segment_text for ALL tokens joined with OR
        qtext = " ".join(tokens)
        fb = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [filt],
                    "must": [{"match": {"segment_text": {"query": qtext, "operator": "or"}}}],
                }
            },
            "aggs": {"by_ep": {"terms": {"field": "episode_id", "size": size_limit}}}
        }
        res2 = client.search(index=idx, body=fb)
        out = {}
        for b in res2.get("aggregations", {}).get("by_ep", {}).get("buckets", []):
            try:
                eid = int(b.get("key"))
            except Exception:
                continue
            out[eid] = int(b.get("doc_count", 0))
        return out

    # ---------- resolve base vs translated channel ----------
    def get_object(self):
        slug = self.kwargs['sanitized_channel_title']
        lang = _norm_lang(self.request)

        base = get_object_or_404(Channel, sanitized_channel_title=slug)
        if lang == 'en':
            return base

        tr = ChannelTranslations.objects.filter(
            sanitized_channel_title=slug,
            language__startswith=lang,
            translated=True
        ).first()
        return tr or base

    def dispatch(self, request, *args, **kwargs):
        disp = self.get_object()
        if isinstance(disp, Channel):
            base = disp
        else:
            base = get_object_or_404(Channel, sanitized_channel_title=disp.sanitized_channel_title)
            base.channel_title   = disp.channel_title
            base.channel_summary = disp.channel_summary
            base.channel_author  = getattr(disp, 'channel_author', base.channel_author)

        self.base_channel    = base
        self.display_channel = disp

        # Count a view for both signed-in users and guests (skip AJAX partials)
        if request.GET.get('ajax') != '1':
            xff = request.META.get('HTTP_X_FORWARDED_FOR', '')
            ip  = (xff.split(',')[0].strip() if xff else request.META.get('REMOTE_ADDR'))
            user_obj = request.user if request.user.is_authenticated else get_guest_user()

            visit, _ = ChannelVisit.objects.get_or_create(
                user=user_obj,
                channel=base,
                defaults={'count': 0}
            )
            ChannelVisit.objects.filter(pk=visit.pk).update(
                count=F('count') + 1,
                last_visited=timezone.now(),
                last_ip_address=ip,   # ✅ this field exists in your model
            )

        return super().dispatch(request, *args, **kwargs)

    # ---------- page context ----------
    def get_context_data(self, **kwargs):
        ctx  = super().get_context_data(**kwargs)
        base = self.base_channel
        lang = get_selected_language(self.request)

        ctx['channel'] = self.display_channel if not isinstance(self.display_channel, Channel) else base
        # Always provide a simple flag the template can trust
        ctx['is_authenticated'] = bool(getattr(self.request.user, 'is_authenticated', False))

        # ---- defaults for guests ----
        has_followed_channel   = False
        receive_notifications  = False
        channel_rating         = None

        # only touch ChannelInteraction for authenticated users
        if ctx['is_authenticated']:
            interaction, _ = ChannelInteraction.objects.get_or_create(
                user=self.request.user, channel=base
            )
            has_followed_channel  = interaction.followed
            receive_notifications = interaction.notifications_enabled
            channel_rating        = interaction.rating

        # toggles & aggregates (counts visible to everyone)
        ctx.update({
            'has_followed_channel':  has_followed_channel,
            'receive_notifications': receive_notifications,
            'channel_rating':        channel_rating,
            'star_range':            range(1, 6),
            'favorites_count':       ChannelInteraction.objects.filter(channel=base, followed=True).count(),
            'notifications_count':   ChannelInteraction.objects.filter(channel=base, notifications_enabled=True).count(),
        })
        rating_stats = ChannelInteraction.objects.filter(channel=base, rating__isnull=False)\
                         .aggregate(avg_rating=Avg('rating'), rating_count=Count('rating'))
        ctx['avg_rating']   = rating_stats['avg_rating']   or 0.0
        ctx['rating_count'] = rating_stats['rating_count'] or 0
        ctx['total_views']  = ChannelVisit.objects.filter(channel=base).aggregate(total=Sum('count'))['total'] or 0
                # ✅ total downloads across all episodes in this channel
        ctx['total_downloads'] = (
            EpisodeDownload.objects
            .filter(episode__channel=base)
            .aggregate(total=Coalesce(Sum('count'), 0))['total']
        )

        # ✅ total shares across all episodes in this channel
        ctx['total_shares'] = (
            EpisodeShare.objects
            .filter(episode__channel=base)
            .aggregate(total=Coalesce(Sum('count'), 0))['total']
        )
        
        # --- Total episodes (per current language if translated; otherwise all base episodes) ---
        if lang in ('en', 'en-us'):
            total_episodes = self.base_channel.episodes.count()
        else:
            total_episodes = (
                EpisodeTranslations.objects
                .filter(
                    episode__channel=self.base_channel,
                    language=lang,
                    translated=True
                )
                .count()
            )

        # expose as a plain context var…
        ctx['total_episodes'] = total_episodes

        # …and also attach to both display and base objects so {{ channel.total_episodes }} works
        try:
            setattr(self.base_channel, 'total_episodes', total_episodes)
            setattr(self.display_channel, 'total_episodes', total_episodes)
        except Exception:
            pass
        # ---- CHANNEL-SCOPED SEARCH ----
        q = (self.request.GET.get('q') or '').strip()
        page_num  = int(self.request.GET.get('page', 1))
        page_size = 10

        # Anno subqueries
        lookup = OuterRef('episode') if lang not in ('en','en-us') else OuterRef('pk')
        bookmarks_sq = EpisodeInteraction.objects.filter(episode=lookup, bookmarked=True)\
                        .order_by().values('episode').annotate(c=Count('*')).values('c')
        avg_rating_sq = EpisodeInteraction.objects.filter(episode=lookup)\
                        .order_by().values('episode').annotate(a=Avg('rating')).values('a')
        rating_count_sq = EpisodeInteraction.objects.filter(episode=lookup, rating__isnull=False)\
                            .order_by().values('episode').annotate(c=Count('rating')).values('c')
        views_sq = EpisodeVisit.objects.filter(episode=lookup)\
                    .order_by().values('episode').annotate(s=Sum('count')).values('s')
                # ---- subqueries for per-episode aggregates (used below) ----
        downloads_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        if q:
            # (1) title matches (within this channel)
            if lang in ('en','en-us'):
                title_ids = set(
                    base.episodes.filter(episode_title__icontains=q).values_list('id', flat=True)
                )
            else:
                title_ids = set(
                    EpisodeTranslations.objects.filter(
                        episode__channel=base, language=lang, translated=True,
                        episode_title__icontains=q
                    ).values_list('episode_id', flat=True)
                )

            # (2) transcript/episode ES matches → candidate set (keep your current retrieval)
            es_hits, epi_hits = [], []
            try:
                chan_ids = list(base.episodes.values_list('id', flat=True))

                is_kw = getattr(settings, "ES_EPISODE_ID_IS_KEYWORD", False)
                chan_terms = [str(i) for i in chan_ids] if is_kw else chan_ids

                broad_q  = ES_Q('match',        segment_text={'query': q, 'operator': 'or'})
                phrase_q = ES_Q('match_phrase', segment_text={'query': q})

                MAX_COLLAPSE_WINDOW = getattr(settings, "ES_MAX_COLLAPSE_WINDOW", 10000)
                override = self.request.GET.get('es_window')
                if override and override.isdigit():
                    MAX_COLLAPSE_WINDOW = min(MAX_COLLAPSE_WINDOW, int(override))

                top_k = min(len(chan_terms), MAX_COLLAPSE_WINDOW)

                tsearch = (TranscriptDocument.search()
                           .filter('terms', episode_id=chan_terms)
                           .query('function_score', query=broad_q,
                                  functions=[{'filter': phrase_q, 'weight': 10}],
                                  boost_mode='sum', score_mode='sum')
                           .params(collapse={'field': 'episode_id',
                                             'inner_hits': {'name': 'top_segment', 'size': 1}})
                           .sort({'_score': 'desc'})
                           .extra(track_total_hits=True))
                t_resp = tsearch[0:top_k].execute()
                for h in t_resp:
                    try:
                        es_hits.append((int(getattr(h, 'episode_id')), float(h.meta.score)))
                    except Exception:
                        pass

                ids_filter_vals = [str(i) for i in chan_ids]
                ebroad  = ES_Q('multi_match', query=q, fields=['full_transcript'], type='best_fields', operator='or')
                ephrase = ES_Q('multi_match', query=q, fields=['full_transcript'], type='phrase')

                edoc = (EpisodeDocument.search()
                        .filter('ids', values=ids_filter_vals)
                        .query('function_score', query=ebroad,
                               functions=[{'filter': ephrase, 'weight': 10}],
                               boost_mode='sum', score_mode='sum')
                        .sort({'_score': 'desc'})
                        .source(False))
                e_resp = edoc[0:min(len(ids_filter_vals), MAX_COLLAPSE_WINDOW)].execute()
                for hit in e_resp:
                    try:
                        epi_hits.append((int(hit.meta.id), float(hit.meta.score)))
                    except Exception:
                        pass
            except Exception as e:
                logger.error("ES retrieval failed: %s", e, exc_info=True)

            candidate_ids = {eid for eid, _ in es_hits} | {eid for eid, _ in epi_hits}
            if not candidate_ids:
                paginator = Paginator(range(0), page_size)
                try:
                    page_obj = paginator.page(page_num)
                except EmptyPage:
                    page_obj = paginator.page(1)
                ctx['episodes'] = []
                return ctx

            # (3) NEW RANKING: count occurrences per episode, then 100 + 10*k
            tokens  = self._tokenize(q)
            occ_map = self._occurrence_counts(candidate_ids, tokens)  # {eid: count}

            score_map = {}
            for eid in candidate_ids:
                occ   = int(occ_map.get(eid, 0))
                score = self.PER_OCC_BONUS * occ
                if eid in title_ids:
                    score += self.TITLE_BONUS
                score_map[eid] = score

            # (4) sort: score desc, pub_date desc; paginate; annotate
            if lang in ('en','en-us'):
                objs = Episode.objects.filter(id__in=score_map.keys()).select_related('channel')
                obj_by_id = {o.id: o for o in objs}
                ranked = []
                for eid, score in score_map.items():
                    ep = obj_by_id.get(eid)
                    if not ep:
                        continue
                    ts = ep.publication_date.timestamp() if getattr(ep, 'publication_date', None) else 0
                    ranked.append((ep, score, ts))
                ranked.sort(key=lambda t: (-t[1], -t[2]))

                all_objs = [o for (o, _, _) in ranked]
                total    = len(all_objs)
                start    = (page_num - 1) * page_size
                end      = start + page_size
                page_objs = all_objs[start:end]

                ids_slice = [o.id for o in page_objs]
                ann_qs = Episode.objects.filter(id__in=ids_slice).annotate(
                    bookmarks_count     = Coalesce(Subquery(bookmarks_sq,   output_field=IntegerField()), Value(0)),
                    ep_avg_rating       = Coalesce(Subquery(avg_rating_sq,  output_field=FloatField()),   Value(0.0)),
                    ep_rating_count     = Coalesce(Subquery(rating_count_sq, output_field=IntegerField()), Value(0)),
                    total_episode_views = Coalesce(Subquery(views_sq,       output_field=IntegerField()), Value(0)),
                    total_downloads     = Coalesce(Subquery(downloads_sq,   output_field=IntegerField()), Value(0)),
                    total_shares        = Coalesce(Subquery(shares_sq,      output_field=IntegerField()), Value(0)),
                )
                ann_map = {e.id: e for e in ann_qs}
                page_annotated = [ann_map.get(i, obj_by_id[i]) for i in ids_slice]

                paginator = Paginator(range(total), page_size)
                try:
                    page_obj = paginator.page(page_num)
                except EmptyPage:
                    page_obj = paginator.page(1)
                ctx['episodes'] = page_annotated

            else:
                tr_qs = EpisodeTranslations.objects.filter(
                    episode_id__in=score_map.keys(), language=lang, translated=True
                ).select_related('episode')
                tr_by_eid = {tr.episode_id: tr for tr in tr_qs}

                eps = Episode.objects.filter(id__in=score_map.keys()).values('id', 'publication_date')
                pub_map = {e['id']: e['publication_date'] for e in eps}

                ranked_tr = []
                for eid, score in score_map.items():
                    tr = tr_by_eid.get(eid)
                    if not tr:
                        continue
                    dt = getattr(tr, 'publication_date', None) or pub_map.get(eid)
                    ts = dt.timestamp() if dt else 0
                    ranked_tr.append((tr, score, ts))
                ranked_tr.sort(key=lambda t: (-t[1], -t[2]))
                total = len(ranked_tr)
                start = (page_num - 1) * page_size
                end   = start + page_size
                page_tr = [t[0] for t in ranked_tr[start:end]]

                for tr in page_tr:
                    tr.channel = base
                ids_slice = [tr.episode_id for tr in page_tr]
                ann_qs = EpisodeTranslations.objects.filter(
                    episode_id__in=ids_slice, language=lang, translated=True
                ).annotate(
                    bookmarks_count     = Coalesce(Subquery(bookmarks_sq,   output_field=IntegerField()), Value(0)),
                    ep_avg_rating       = Coalesce(Subquery(avg_rating_sq,  output_field=FloatField()),   Value(0.0)),
                    ep_rating_count     = Coalesce(Subquery(rating_count_sq, output_field=IntegerField()), Value(0)),
                    total_episode_views = Coalesce(Subquery(views_sq,       output_field=IntegerField()), Value(0)),
                    total_downloads     = Coalesce(Subquery(downloads_sq,   output_field=IntegerField()), Value(0)),
                    total_shares        = Coalesce(Subquery(shares_sq,      output_field=IntegerField()), Value(0)),
                )
                ann_map = {tr.episode_id: tr for tr in ann_qs}
                merged = []
                for tr in page_tr:
                    m = ann_map.get(tr.episode_id)
                    if m:
                        m.channel = base
                        merged.append(m)
                    else:
                        merged.append(tr)

                paginator = Paginator(range(total), page_size)
                try:
                    page_obj = paginator.page(page_num)
                except EmptyPage:
                    page_obj = paginator.page(1)
                ctx['episodes'] = merged

            return ctx  # end q branch

        # ---- DEFAULT (no search term) ----
        if lang in ('en', 'en-us'):
            eps_qs = base.episodes.all().order_by('-publication_date').annotate(
                bookmarks_count     = Coalesce(Subquery(bookmarks_sq,   output_field=IntegerField()), Value(0)),
                ep_avg_rating       = Coalesce(Subquery(avg_rating_sq,  output_field=FloatField()),   Value(0.0)),
                ep_rating_count     = Coalesce(Subquery(rating_count_sq, output_field=IntegerField()), Value(0)),
                total_episode_views = Coalesce(Subquery(views_sq,       output_field=IntegerField()), Value(0)),
                total_downloads     = Coalesce(Subquery(downloads_sq,   output_field=IntegerField()), Value(0)),
                total_shares        = Coalesce(Subquery(shares_sq,      output_field=IntegerField()), Value(0)),
            )
            paginator = Paginator(eps_qs, page_size)
            try:
                page_obj = paginator.page(page_num)
            except EmptyPage:
                page_obj = []
            ctx['episodes'] = page_obj
        else:
            tr_qs = EpisodeTranslations.objects.filter(
                episode__channel=base, language=lang, translated=True
            ).order_by('-publication_date').annotate(
                bookmarks_count     = Coalesce(Subquery(bookmarks_sq,   output_field=IntegerField()), Value(0)),
                ep_avg_rating       = Coalesce(Subquery(avg_rating_sq,  output_field=FloatField()),   Value(0.0)),
                ep_rating_count     = Coalesce(Subquery(rating_count_sq, output_field=IntegerField()), Value(0)),
                total_episode_views = Coalesce(Subquery(views_sq,       output_field=IntegerField()), Value(0)),
                total_downloads     = Coalesce(Subquery(downloads_sq,   output_field=IntegerField()), Value(0)),
                total_shares        = Coalesce(Subquery(shares_sq,      output_field=IntegerField()), Value(0)),
            )
            for tr in tr_qs:
                tr.channel = base
            paginator = Paginator(tr_qs, page_size)
            try:
                page_obj = paginator.page(page_num)
            except EmptyPage:
                page_obj = []
            ctx['episodes'] = page_obj

        return ctx
    
    def get(self, request, *args, **kwargs):
        # log both full-page and AJAX searches, but only for page=1 to avoid double counts
        q = (request.GET.get('q') or '').strip()
        if q and (request.GET.get('page', '1') == '1'):
            user = request.user if request.user.is_authenticated else None
            lang = get_selected_language(request)

            xff = request.META.get('HTTP_X_FORWARDED_FOR', '')
            ip  = (xff.split(',')[0].strip() if xff else request.META.get('REMOTE_ADDR'))

            try:
                obj, created = ChannelSearchQuery.objects.get_or_create(
                    user=user,
                    channel=self.base_channel,   # set in dispatch()
                    query=q,
                    defaults={'language': lang, 'ip_address': ip}
                )
                if not created:
                    ChannelSearchQuery.objects.filter(pk=obj.pk).update(
                        count=F('count') + 1,
                        last_searched=timezone.now(),
                        language=lang,
                        ip_address=ip
                    )
            except Exception:
                logging.exception("Failed to record ChannelSearchQuery")

        return super().get(request, *args, **kwargs)



    # ---------- ajax partial vs full template ----------
    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax') == '1':
            if not context['episodes']:
                return HttpResponse('')
            return render(self.request, 'podcasts/channel_detail_item.html', context)
        return super().render_to_response(context, **response_kwargs)




class EpisodeDetailView(DetailView):
    login_url           = reverse_lazy('podcasts:home')
    template_name       = 'podcasts/episode_detail.html'
    context_object_name = 'episode'

    def dispatch(self, request, *args, **kwargs):
        disp = self.get_object()
        if isinstance(disp, Episode):
            base = disp
        else:
            base = disp.episode
            # keep template compatibility
            disp.channel = base.channel

        self.base_episode    = base
        self.display_episode = disp

        # ✅ Count a view for both guests and signed-in users (skip AJAX partials)
        if request.GET.get('ajax') != '1':
            xff = request.META.get('HTTP_X_FORWARDED_FOR', '')
            ip  = (xff.split(',')[0].strip() if xff else request.META.get('REMOTE_ADDR'))
            user_obj = request.user if request.user.is_authenticated else get_guest_user()

            visit, _ = EpisodeVisit.objects.get_or_create(
                user=user_obj,
                episode=base,
                defaults={'count': 0}
            )
            EpisodeVisit.objects.filter(pk=visit.pk).update(
                count=F('count') + 1,
                last_visited=timezone.now(),
                last_ip_address=ip,   # matches your models.py
            )

        return super().dispatch(request, *args, **kwargs)

    def get_object(self):
        slug_ch = self.kwargs['sanitized_channel_title']
        slug_ep = self.kwargs['sanitized_episode_title']
        lang    = (get_selected_language(self.request) or 'en').lower().split('-', 1)[0]

        incoming_norm = slug_norm(slug_ep)

        # 1) exact base
        try:
            base = (
                Episode.objects
                .select_related('channel')
                .get(
                    channel__sanitized_channel_title=slug_ch,
                    sanitized_episode_title=slug_ep,
                )
            )
        except Episode.DoesNotExist:
            # 2) exact translation in this channel
            tr_any = (
                EpisodeTranslations.objects
                .select_related('episode', 'episode__channel')
                .filter(
                    episode__channel__sanitized_channel_title=slug_ch,
                    sanitized_episode_title=slug_ep,
                    translated=True,
                )
                .first()
            )
            if tr_any:
                base = tr_any.episode
            else:
                # 3) AGGRESSIVE FALLBACK
                base = None

                # 3a) scan base episodes for this channel
                for ep in (
                    Episode.objects
                    .filter(channel__sanitized_channel_title=slug_ch)
                    .only('id', 'sanitized_episode_title', 'channel_id')
                ):
                    db_norm = slug_norm(ep.sanitized_episode_title)

                    # accept equal
                    if db_norm == incoming_norm:
                        base = ep
                        break

                    # accept prefix/suffix — incoming longer than DB OR DB longer than incoming
                    if incoming_norm.startswith(db_norm) or db_norm.startswith(incoming_norm):
                        base = ep
                        break

                # 3b) if still nothing, scan translations
                if not base:
                    for tr in (
                        EpisodeTranslations.objects
                        .filter(
                            episode__channel__sanitized_channel_title=slug_ch,
                            translated=True,
                        )
                        .only('id', 'sanitized_episode_title', 'episode_id')
                    ):
                        db_norm = slug_norm(tr.sanitized_episode_title)

                        if db_norm == incoming_norm:
                            base = tr.episode
                            break

                        if incoming_norm.startswith(db_norm) or db_norm.startswith(incoming_norm):
                            base = tr.episode
                            break

                if not base:
                    # nothing matched at all
                    raise Http404("No Episode matches the given query.")

        # at this point we have the base episode
        self.base_episode = base

        # 4) language-specific display
        if lang != 'en':
            tr = (
                EpisodeTranslations.objects
                .select_related('episode', 'episode__channel')
                .filter(episode=base, language__startswith=lang, translated=True)
                .first()
            )
            if tr:
                return tr

        return base



    def get_queryset(self):
        lang = get_selected_language(self.request)
        if lang in ('en','en-us'):
            return Episode.objects.select_related('channel')
        return EpisodeTranslations.objects.filter(language=lang, translated=True)

    def get_context_data(self, **kwargs):
        ctx  = super().get_context_data(**kwargs)
        disp = self.display_episode
        base = self.base_episode
        lang = get_selected_language(self.request)

        # 1) TRANSCRIPTS
        if isinstance(disp, EpisodeTranslations):
            tr_qs = TranscriptTranslations.objects.filter(
                episodetranslations=disp,
                language=lang
            ).order_by('segment_time')
            segments = tr_qs if tr_qs.exists() else Transcript.objects.filter(
                episode=base
            ).order_by('segment_time')
        else:
            segments = Transcript.objects.filter(
                episode=base
            ).order_by('segment_time')

        # 2) CHAPTERS (translated first, fallback to originals)
        if isinstance(disp, EpisodeTranslations):
            ch_qs = ChapterTranslations.objects.filter(
                episodetranslations=disp,
                language=lang
            )
            if not ch_qs.exists():
                ch_qs = Chapter.objects.filter(episode=base)
        else:
            ch_qs = Chapter.objects.filter(episode=base)

        # Helpers to normalize times
        def _to_seconds(ts: str) -> int:
            parts = [int(p) for p in (ts or "0").split(':')]
            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h, m, s = 0, parts[0], parts[1]
            else:
                h, m, s = 0, 0, parts[0]
            return h * 3600 + m * 60 + s

        def _fmt_hms(total_seconds: int) -> str:
            if total_seconds < 0:
                total_seconds = 0
            h = total_seconds // 3600
            m = (total_seconds % 3600) // 60
            s = total_seconds % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        chapters = list(ch_qs)
        chapters.sort(key=lambda c: _to_seconds(c.chapter_time or "0"))
        for c in chapters:
            try:
                c.chapter_time = _fmt_hms(_to_seconds(c.chapter_time or "0"))
            except Exception:
                c.chapter_time = "00:00:00"

        # 3) USER INTERACTION + AGGREGATES (guest-safe)
        # counts for everyone
        ctx['bookmarks_count'] = EpisodeInteraction.objects.filter(
            episode=base, bookmarked=True
        ).count()
        ctx['comments_count'] = base.comments.count()

        stats = EpisodeInteraction.objects.filter(
            episode=base, rating__isnull=False
        ).aggregate(avg=Avg('rating'), cnt=Count('rating'))
        ctx['ep_avg_rating']   = stats['avg'] or 0
        ctx['ep_rating_count'] = stats['cnt'] or 0

        ctx['total_episode_views'] = EpisodeVisit.objects.filter(
            episode=base
        ).aggregate(total=Sum('count'))['total'] or 0

        ctx['total_downloads'] = (
            EpisodeDownload.objects
            .filter(episode=base)
            .aggregate(total=Sum('count'))['total'] or 0
        )

        # per-language downloads
        try:
            canon_lang = _canon_lang(lang)
        except NameError:
            canon_lang = (lang or 'en').split('-', 1)[0].lower()

        ctx['downloads_for_lang'] = (
            EpisodeDownload.objects
            .filter(episode=base, language=canon_lang)
            .aggregate(total=Sum('count'))['total'] or 0
        )

        ctx['total_shares'] = (
            EpisodeShare.objects
            .filter(episode=base)
            .aggregate(total=Coalesce(Sum('count'), 0))['total'] or 0
        )
        
        # per-user toggles only if authenticated
        if self.request.user.is_authenticated:
            ei = EpisodeInteraction.objects.filter(user=self.request.user, episode=base).first()
            if not ei:
                # don't create rows unless needed; default flags
                ctx['is_bookmarked']  = False
                ctx['episode_rating'] = 0
            else:
                ctx['is_bookmarked']  = bool(ei.bookmarked)
                ctx['episode_rating'] = ei.rating or 0
        else:
            ctx['is_bookmarked']  = False
            ctx['episode_rating'] = 0

        ctx['star_range']        = range(1, 6)
        ctx['merged_segments']   = self.merge_consecutive_speakers(segments)
        ctx['chapters']          = chapters
        ctx['post_episode_id']   = self.base_episode.id
        ctx['selected_language'] = lang
        ctx['episode']           = disp
        ctx['base_slug_ch']      = self.base_episode.channel.sanitized_channel_title
        ctx['base_slug_ep']      = self.base_episode.sanitized_episode_title
        return ctx


    def merge_consecutive_speakers(self, segments):
        merged = []
        current = None
        MAX_DURATION = timedelta(minutes=3)

        for seg in segments:
            if current is None:
                # start first segment
                current = {
                    'combined_time': seg.segment_time,
                    'speaker':      seg.speaker,
                    'combined_text': seg.segment_text,
                }
                continue

            # same speaker?
            if seg.speaker == current['speaker']:
                try:
                    # parse the existing start and the new end
                    start_str, _      = current['combined_time'].split(' - ', 1)
                    _, new_end_str   = seg.segment_time.split(' - ', 1)

                    fmt = "%H:%M:%S"
                    start_dt = datetime.strptime(start_str, fmt)
                    end_dt   = datetime.strptime(new_end_str, fmt)

                    # if total duration exceeds 5min, close out current and start a new one
                    if end_dt - start_dt > MAX_DURATION:
                        merged.append(current)
                        current = {
                            'combined_time': seg.segment_time,
                            'speaker':       seg.speaker,
                            'combined_text': seg.segment_text,
                        }
                    else:
                        # otherwise extend the current segment
                        current['combined_time'] = f"{start_str} - {new_end_str}"
                        current['combined_text'] += " " + seg.segment_text

                except ValueError:
                    # if the time format was unexpected, just concatenate
                    current['combined_text'] += " " + seg.segment_text

            else:
                # different speaker → push the old, start fresh
                merged.append(current)
                current = {
                    'combined_time': seg.segment_time,
                    'speaker':       seg.speaker,
                    'combined_text': seg.segment_text,
                }

        # don't forget the last one
        if current:
            merged.append(current)

        return merged



class EpisodeListView(ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/episode_list.html'
    context_object_name = 'episodes'
    paginate_by = 10

    # include "recent"
    ALLOWED_SORTS = {'trending', 'recent', 'views', 'bookmarks', 'comments', 'stars', 'downloaded', 'shared', 'title'}
    ALLOWED_DIRS = {'asc', 'desc'}

    def _parse_sort(self):
        # default: Most recent
        sort = (self.request.GET.get('sort') or 'trending').lower()
        direction = (self.request.GET.get('dir') or 'desc').lower()
        if sort not in self.ALLOWED_SORTS:
            sort = 'trending'
        if direction not in self.ALLOWED_DIRS:
            direction = 'desc'
        return sort, direction

    def get_queryset(self):
        lang = get_selected_language(self.request)
        sort, direction = self._parse_sort()
        dir_prefix = '' if direction == 'asc' else '-'

        # time windows
        now = timezone.now()
        dt_24h = now - timedelta(hours=24)
        dt_7d  = now - timedelta(days=7)

        # --- Subqueries for total downloads ---
        downloads_sq_base = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        downloads_sq_tr = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('episode'))  # note: EpisodeTranslations.episode FK
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_sq_base = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_sq_tr = (
            EpisodeShare.objects
            .filter(episode=OuterRef('episode'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        if lang in ('en', 'en-us'):
            qs = (
                Episode.objects
                .select_related('channel')
                .prefetch_related('transcripts', 'chapters')
                .annotate(
                    # existing aggregates
                    bookmarks_count=Count(
                        'episode_interactions',
                        filter=Q(episode_interactions__bookmarked=True),
                        distinct=True
                    ),
                    comments_count=Count('comments', distinct=True),
                    ep_avg_rating=Avg('episode_interactions__rating'),
                    ep_rating_count=Count(
                        'episode_interactions__user',
                        filter=Q(episode_interactions__rating__isnull=False),
                        distinct=True
                    ),
                    total_episode_views=Coalesce(Sum('episodevisit__count'), 0),
                    total_downloads=Coalesce(Subquery(downloads_sq_base, output_field=IntegerField()), Value(0)),
                    total_shares=Coalesce(Subquery(shares_sq_base, output_field=IntegerField()), Value(0)),

                    # NEW: 24h and 7d windows (velocity)
                    views_24h=Coalesce(
                        Sum('episodevisit__count', filter=Q(episodevisit__last_visited__gte=dt_24h)),
                        0
                    ),
                    views_7d=Coalesce(
                        Sum('episodevisit__count', filter=Q(episodevisit__last_visited__gte=dt_7d)),
                        0
                    ),
                    downloads_7d=Coalesce(
                        Sum('downloads__count', filter=Q(downloads__last_downloaded__gte=dt_7d)),
                        0
                    ),
                    shares_7d=Coalesce(
                        Sum('shares__count', filter=Q(shares__last_shared__gte=dt_7d)),
                        0
                    ),
                    comments_7d=Coalesce(
                        Count('comments', filter=Q(comments__created_at__gte=dt_7d), distinct=True),
                        0
                    ),
                )
                .annotate(
                    # Weighted trending score — tweak weights as you like
                    trending_score=(
                        F('views_7d') * 1.0 +
                        F('downloads_7d') * 3.0 +
                        F('shares_7d') * 4.0 +
                        F('comments_7d') * 2.0 +
                        F('ep_rating_count') * 1.0
                    )
                )
            )

            if sort == 'trending':
                # Desc: most trending. Add tie breakers: 24h velocity, then recency
                ordering = [
                    F('trending_score').desc(nulls_last=True) if direction == 'desc'
                    else F('trending_score').asc(nulls_last=True),
                    F('views_24h').desc(nulls_last=True) if direction == 'desc'
                    else F('views_24h').asc(nulls_last=True),
                    F('publication_date').desc(nulls_last=True) if direction == 'desc'
                    else F('publication_date').asc(nulls_last=True),
                    'episode_title',
                ]
            elif sort == 'recent':
                # publication_date with A–Z tie-break; unrated/NULL dates go last either way
                if direction == 'desc':
                    ordering = [F('publication_date').desc(nulls_last=True), 'episode_title']
                else:
                    ordering = [F('publication_date').asc(nulls_last=True), 'episode_title']

            elif sort == 'views':
                ordering = [f'{dir_prefix}total_episode_views', 'episode_title']

            elif sort == 'bookmarks':
                ordering = [f'{dir_prefix}bookmarks_count', 'episode_title']

            elif sort == 'comments':
                ordering = [f'{dir_prefix}comments_count', 'episode_title']

            elif sort == 'stars':
                if direction == 'desc':
                    ordering = [
                        F('ep_avg_rating').desc(nulls_last=True),
                        F('ep_rating_count').desc(),
                        'episode_title',
                    ]
                else:
                    ordering = [
                        F('ep_avg_rating').asc(nulls_last=True),
                        F('ep_rating_count').asc(),
                        'episode_title',
                    ]

            elif sort == 'downloaded':
                ordering = [f'{dir_prefix}total_downloads', 'episode_title']

            elif sort == 'shared':
                ordering = [f'{dir_prefix}total_shares', 'episode_title']

            elif sort == 'title':
                # pure alphabetical
                ordering = [f'{dir_prefix}episode_title']

            else:
                # safety default
                ordering = [F('publication_date').desc(nulls_last=True), 'episode_title']

            return qs.order_by(*ordering)
        
        # Non-English (EpisodeTranslations)
        qs = (
            EpisodeTranslations.objects
            .filter(language=lang, translated=True)
            .select_related('episode__channel')
            .prefetch_related('transcriptstranslations', 'chapterstranslations')
            .annotate(
                bookmarks_count=Count(
                    'episode__episode_interactions',
                    filter=Q(episode__episode_interactions__bookmarked=True),
                    distinct=True
                ),
                comments_count=Count('episode__comments', distinct=True),
                ep_avg_rating=Avg('episode__episode_interactions__rating'),
                ep_rating_count=Count(
                    'episode__episode_interactions__user',
                    filter=Q(episode__episode_interactions__rating__isnull=False),
                    distinct=True
                ),
                total_episode_views=Coalesce(Sum('episode__episodevisit__count'), 0),
                total_downloads=Coalesce(Subquery(downloads_sq_tr, output_field=IntegerField()), Value(0)),
                total_shares=Coalesce(Subquery(shares_sq_tr, output_field=IntegerField()), Value(0)),

                # Recent windows via base episode relations
                views_24h=Coalesce(
                    Sum('episode__episodevisit__count', filter=Q(episode__episodevisit__last_visited__gte=dt_24h)), 0
                ),
                views_7d=Coalesce(
                    Sum('episode__episodevisit__count', filter=Q(episode__episodevisit__last_visited__gte=dt_7d)), 0
                ),
                downloads_7d=Coalesce(
                    Sum('episode__downloads__count', filter=Q(episode__downloads__last_downloaded__gte=dt_7d)), 0
                ),
                shares_7d=Coalesce(
                    Sum('episode__shares__count', filter=Q(episode__shares__last_shared__gte=dt_7d)), 0
                ),
                comments_7d=Coalesce(
                    Count('episode__comments', filter=Q(episode__comments__created_at__gte=dt_7d), distinct=True), 0
                ),
            )
            .annotate(
                trending_score=(
                    F('views_7d')      * 1.0 +
                    F('downloads_7d')  * 3.0 +
                    F('shares_7d')     * 4.0 +
                    F('comments_7d')   * 2.0 +
                    F('ep_rating_count') * 1.0
                )
            )
        )

        if sort == 'trending':
            ordering = [
                F('trending_score').desc(nulls_last=True) if direction == 'desc'
                else F('trending_score').asc(nulls_last=True),
                F('views_24h').desc(nulls_last=True) if direction == 'desc'
                else F('views_24h').asc(nulls_last=True),
                F('publication_date').desc(nulls_last=True) if direction == 'desc'
                else F('publication_date').asc(nulls_last=True),
                'episode_title',
            ]

        elif sort == 'recent':
            ordering = [
                F('publication_date').desc(nulls_last=True) if direction == 'desc'
                else F('publication_date').asc(nulls_last=True),
                'episode_title',
            ]

        elif sort == 'views':
            ordering = [f'{dir_prefix}total_episode_views', 'episode_title']

        elif sort == 'bookmarks':
            ordering = [f'{dir_prefix}bookmarks_count', 'episode_title']

        elif sort == 'comments':
            ordering = [f'{dir_prefix}comments_count', 'episode_title']

        elif sort == 'stars':
            ordering = [
                F('ep_avg_rating').desc(nulls_last=True) if direction == 'desc'
                else F('ep_avg_rating').asc(nulls_last=True),
                F('ep_rating_count').desc() if direction == 'desc' else F('ep_rating_count').asc(),
                'episode_title',
            ]

        elif sort == 'downloaded':
            ordering = [f'{dir_prefix}total_downloads', 'episode_title']

        elif sort == 'shared':
            ordering = [f'{dir_prefix}total_shares', 'episode_title']

        elif sort == 'title':
            ordering = [f'{dir_prefix}episode_title']

        else:
            ordering = [F('publication_date').desc(nulls_last=True), 'episode_title']

        return qs.order_by(*ordering)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        lang = get_selected_language(self.request)

        if lang not in ('en', 'en-us'):
            patched = []
            for ep in context['episodes']:
                if ep.episode:
                    ep.sanitized_episode_title = ep.episode.sanitized_episode_title
                    ep.channel = ep.episode.channel
                patched.append(ep)
            context['episodes'] = patched

        sort, direction = self._parse_sort()
        context['sort'] = sort
        context['dir'] = direction

        labels = {
            ('trending', 'desc'): "Most trending",
            ('trending', 'asc'):  "Least trending",
            ('recent', 'desc'): "Most recent",
            ('recent', 'asc'):  "Least recent",
            ('views', 'desc'): "Most watched",
            ('views', 'asc'):  "Least watched",
            ('bookmarks', 'desc'): "Most bookmarked",
            ('bookmarks', 'asc'):  "Least bookmarked",
            ('comments', 'desc'): "Most commented",
            ('comments', 'asc'):  "Least commented",
            ('stars', 'desc'): "Most stars",
            ('stars', 'asc'):  "Least stars",
            ('downloaded', 'desc'): "Most downloaded",
            ('downloaded', 'asc'):  "Least downloaded",
            ('shared', 'desc'): "Most shared",
            ('shared', 'asc'):  "Least shared",
            ('title', 'asc'):  "A → Z",
            ('title', 'desc'): "Z → A",
        }
        context['current_sort_label'] = labels.get((sort, direction), "Most trending")
        context['selected_language'] = lang
        return context

    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax') == '1':
            return render(self.request, 'podcasts/episode_list_items.html', context)
        return super().render_to_response(context, **response_kwargs)



class SearchResultsView(LoginRequiredMixin, ListView):
    login_url = 'podcasts:home'

    # ––– Make these match your <input value="…"> in the form –––
    SEGMENT_FIELD        = 'segment_text'
    SEGMENT_ALIAS_FIELD  = 'transcript_text'
    TRANSCRIPTS_FIELD    = 'transcripts'

    context_object_name = 'episodes'
    paginate_by = 10
    STOP_WORDS = {'the','a','an','of','in','and','or','to','so','for','on','at','by'}

    def get_queryset(self):
        # we override pagination entirely, so this is only a placeholder
        from .models import Episode
        return Episode.objects.none()

    # ---------- helper: annotate Episode queryset with aggregates ----------
    def _with_episode_stats(self, qs):
        downloads_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        return (
            qs.annotate(
                bookmarks_count=Count(
                    'episode_interactions',
                    filter=Q(episode_interactions__bookmarked=True),
                    distinct=True
                ),
                comments_count=Count('comments', distinct=True),
                ep_avg_rating=Avg('episode_interactions__rating'),
                ep_rating_count=Count(
                    'episode_interactions__user',
                    filter=Q(episode_interactions__rating__isnull=False),
                    distinct=True
                ),
                total_episode_views=Coalesce(Sum('episodevisit__count'), 0),
                total_downloads=Coalesce(Subquery(downloads_sq, output_field=IntegerField()), Value(0)),
                total_shares=Coalesce(Subquery(shares_sq, output_field=IntegerField()), Value(0)),
            )
        )

    def paginate_queryset(self, qs, page_size):
        from .models import Episode

        q           = (self.request.GET.get('q', '') or '').strip()
        search_type = self.request.GET.get('search_type', 'episodes')

        # 1) No query → default ListView pagination
        if not q:
            return super().paginate_queryset(qs, page_size)

        # 2) CHANNELS branch (unchanged except using correct related name)
        if search_type == 'channels':
            wants = self.request.GET.getlist('search_in')
            filters = []
            if 'channel_title' in wants:
                filters.append(Q(channel_title__icontains=q))
            if 'channel_author' in wants:
                filters.append(Q(channel_author__icontains=q))
            if 'channel_summary' in wants:
                filters.append(Q(channel_summary__icontains=q))

            chans = Channel.objects.all()

            if filters:
                combined = filters.pop()
                for f in filters:
                    combined |= f
                chans = chans.filter(combined)
            else:
                chans = chans.filter(channel_title__icontains=q)

            visits_sq = (
                ChannelVisit.objects
                .filter(channel=OuterRef('pk'))
                .values('channel')
                .annotate(total=Sum('count'))
                .values('total')[:1]
            )
            episodes_sq = (
                Episode.objects
                .filter(channel=OuterRef('pk'))
                .values('channel')
                .annotate(c=Count('*'))
                .values('c')[:1]
            )
            

            chans = (
                chans
                .annotate(
                    total_views=Coalesce(Subquery(visits_sq, output_field=IntegerField()), 0),
                    episode_count=Coalesce(Subquery(episodes_sq, output_field=IntegerField()), 0),
                    favorites_count=Count(
                        'channel_interactions',
                        filter=Q(channel_interactions__followed=True),
                        distinct=True,
                    ),
                    notifications_count=Count(
                        'channel_interactions',
                        filter=Q(channel_interactions__notifications_enabled=True),
                        distinct=True,
                    ),
                    avg_rating=Avg('channel_interactions__rating'),
                    rating_count=Count(
                        'channel_interactions__user',
                        filter=Q(channel_interactions__rating__isnull=False),
                        distinct=True,
                    ),
                )
            ).only(
                'id', 'channel_title', 'channel_author', 'channel_summary',
                'channel_image_url', 'sanitized_channel_title'
            )
            # inside SearchResultsView.paginate_queryset(...), channels branch — just before paginator
            chans = chans.only(
                'id', 'channel_title', 'channel_author', 'channel_summary',
                'channel_image_url', 'sanitized_channel_title'
            )

            # ✅ add a stable ordering to silence UnorderedObjectListWarning
            chans = chans.order_by('channel_title', 'id')

            paginator = Paginator(chans, page_size)
            page_num  = int(self.request.GET.get('page', 1))
            page_obj  = paginator.get_page(page_num)
            return paginator, page_obj, list(page_obj.object_list), page_obj.has_other_pages()

        # 4) FULL-TEXT / ELASTICSEARCH (episodes)
        date_filter = self.request.GET.get('search_date', 'anytime')
        window = None
        if date_filter != 'anytime':
            days = int(date_filter)
            window = (timezone.timedelta(hours=24) if days == 24
                      else timezone.timedelta(days=days))

        selected = set(self.request.GET.getlist('search_in'))
        transcript_only = [
            {self.SEGMENT_FIELD},
            {self.SEGMENT_ALIAS_FIELD},
            {self.TRANSCRIPTS_FIELD},
        ]

        # 4b) Transcript-only ES query with phrase boost, then annotate
        selected = set(self.request.GET.getlist('search_in'))
        transcript_only = [{self.SEGMENT_FIELD}, {self.SEGMENT_ALIAS_FIELD}, {self.TRANSCRIPTS_FIELD}]

        if selected in transcript_only:
            page      = int(self.request.GET.get('page', 1))
            page_size = self.paginate_by
            start     = (page - 1) * page_size
            end       = start + page_size

            broad_q  = ES_Q('match',        segment_text={'query': q, 'operator': 'or'})
            phrase_q = ES_Q('match_phrase', segment_text={'query': q})

            tsearch = (
                TranscriptDocument.search()
                .query(
                    'function_score',
                    query=broad_q,
                    functions=[{'filter': phrase_q, 'weight': 10}],
                    boost_mode='sum',
                    score_mode='sum'
                )
                .params(collapse={
                    'field': 'episode_id',
                    'inner_hits': {'name': 'top_segment', 'size': 1}
                })
                .sort({'_score': 'desc'})
                .extra(track_total_hits=True)
            )

            if window:
                tsearch = tsearch.filter(
                    'nested',
                    path='episode',
                    query={'range': {
                        'episode.publication_date': {
                            'gte': (timezone.now() - window).isoformat()
                        }
                    }}
                )

            resp = tsearch[start:end].execute()
            hits = resp.hits.hits
            scored_ids = [(hit['_source']['episode_id'], hit['_score']) for hit in hits]

            # Pull & annotate, exclude empty channel slugs to avoid NoReverseMatch
            episode_qs = self._with_episode_stats(
                Episode.objects.filter(id__in=[eid for eid, _ in scored_ids])
            ).select_related('channel') \
             .prefetch_related('transcripts') \
             .exclude(channel__sanitized_channel_title__isnull=True) \
             .exclude(channel__sanitized_channel_title='')

            id_map = {e.id: e for e in episode_qs}

            scored_eps = [(id_map.get(eid), score) for eid, score in scored_ids if id_map.get(eid)]
            scored_eps.sort(
                key=lambda pair: (
                    -pair[1],
                    -pair[0].publication_date.timestamp() if pair[0].publication_date else 0
                )
            )
            page_list = [ep for ep, _ in scored_eps]

            total     = resp.hits.total.value if hasattr(resp.hits, "total") else len(scored_eps)
            paginator = Paginator(range(total), page_size)
            try:
                page_obj = paginator.page(page)
            except EmptyPage:
                page_obj = paginator.page(1)

            return paginator, page_obj, page_list, page_obj.has_other_pages()

        # 4c) Multi-match ES across EpisodeDocument, then annotate
        es = EpisodeDocument.search()
        if window:
            es = es.filter('range', publication_date={'gte': timezone.now() - window})

        fields = [
            'episode_title',
            'description',
            'channel.channel_title',
            'translations.episode_title',
            'translations.description',
            'full_transcript',
        ]

        broad_q  = ES_Q('multi_match', query=q, fields=fields, type='best_fields', operator='or')
        phrase_q = ES_Q('multi_match', query=q, fields=fields, type='phrase')

        es = es.query(
            'function_score',
            query=broad_q,
            functions=[{'filter': phrase_q, 'weight': 10}],
            boost_mode='sum',
            score_mode='sum'
        ).sort({'_score': 'desc'}, {'publication_date': 'desc'})

        total = es.count()
        page  = int(self.request.GET.get('page', 1))
        start = (page - 1) * page_size
        end   = start + page_size

        resp = es[start:end].execute()
        ids  = [int(hit.meta.id) for hit in resp]

        episodes = self._with_episode_stats(
            Episode.objects.filter(id__in=ids)
        ).select_related('channel') \
         .prefetch_related('transcripts') \
         .exclude(channel__sanitized_channel_title__isnull=True) \
         .exclude(channel__sanitized_channel_title='')

        id_map    = {e.id: e for e in episodes}
        page_list = [id_map[i] for i in ids if i in id_map]

        paginator = Paginator(range(total), page_size)
        try:
            page_obj = paginator.page(page)
        except EmptyPage:
            page_obj = paginator.page(1)

        return paginator, page_obj, page_list, page_obj.has_other_pages()

    def _compute_suggestions(self, q, limit=10):
        sims = (
            Episode.objects
                   .annotate(sim=TrigramSimilarity('episode_title', q))
                   .filter(sim__gt=0.2)
                   .order_by('-sim')
                   .values_list('episode_title', flat=True)[:limit]
        )
        results = list(sims)
        if len(results) < limit:
            all_titles = list(
                Episode.objects
                       .values_list('episode_title', flat=True)
                       .distinct()[:2000]
            )
            for t in difflib.get_close_matches(q, all_titles, n=limit, cutoff=0.5):
                if t not in results:
                    results.append(t)
                    if len(results) >= limit:
                        break
        return results

    def _compute_channel_suggestions(self, q, limit=10):
        sims = (
            Channel.objects
                   .annotate(sim=TrigramSimilarity('channel_title', q))
                   .filter(sim__gt=0.2)
                   .order_by('-sim')
                   .values_list('channel_title', flat=True)[:limit]
        )
        results = list(sims)
        if len(results) < limit:
            all_titles = list(
                Channel.objects
                       .values_list('channel_title', flat=True)
                       .distinct()[:2000]
            )
            for t in difflib.get_close_matches(q, all_titles, n=limit, cutoff=0.5):
                if t not in results:
                    results.append(t)
                    if len(results) >= limit:
                        break
        return results

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        q           = self.request.GET.get('q','')
        search_type = self.request.GET.get('search_type', 'episodes').strip()

        ctx.update({
            'q':                  q,
            'selected_fields':    self.request.GET.getlist('search_in'),
            'selected_languages': self.request.GET.getlist('search_language'),
            'selected_date':      self.request.GET.get('search_date','anytime'),
        })

        if q:
            if search_type == 'channels':
                ctx['did_you_mean'] = self._compute_channel_suggestions(q)
            else:
                ctx['did_you_mean'] = self._compute_suggestions(q)

        return ctx

    def get_context_object_name(self, object_list):
        if self.request.GET.get('search_type') == 'channels':
            return 'channels'
        return super().get_context_object_name(object_list)

    def render_to_response(self, context, **response_kwargs):
        search_type = self.request.GET.get('search_type','episodes')
        is_ajax    = self.request.GET.get('ajax') == '1'

        if is_ajax:
            tpl = ( search_type=='channels'
                    and 'podcasts/search_results_ch_items.html'
                    or 'podcasts/search_results_items.html' )
            return render(self.request, tpl, context)

        tpl = ( search_type=='channels'
                and 'podcasts/search_results_ch.html'
                or 'podcasts/search_results.html' )
        return render(self.request, tpl, context)

    def get(self, request, *args, **kwargs):
        # record non-AJAX searches
        if request.GET.get('ajax') != '1':
            query  = request.GET.get('q','')
            in_str = ",".join(request.GET.getlist('search_in'))
            date_f = request.GET.get('search_date','anytime')
            ip     = request.META.get('HTTP_X_FORWARDED_FOR',
                                     request.META.get('REMOTE_ADDR'))
            user   = request.user if request.user.is_authenticated else None
            try:
                sq, created = SearchQuery.objects.get_or_create(
                    user=user, query=query,
                    defaults={
                        'search_in': in_str,
                        'search_date': date_f,
                        'ip_address': ip
                    }
                )
                if not created:
                    sq.count += 1
                    sq.last_searched = timezone.now()
                    sq.ip_address    = ip
                    sq.save()
            except:
                pass

        # guard out-of-range AJAX
        paginator, page_obj, _, _ = self.paginate_queryset(
            self.get_queryset(), self.paginate_by
        )
        if request.GET.get('ajax') == '1' and int(request.GET.get('page',1)) > paginator.num_pages:
            return HttpResponse('', status=200)

        return super().get(request, *args, **kwargs)



class FavoritesListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/favorites_list.html'
    context_object_name = 'channels'
    paginate_by = 5  # adjust as needed

    def get_queryset(self):
        # Channels the user follows
        channel_ids = ChannelInteraction.objects.filter(
            user=self.request.user, followed=True
        ).values_list('channel_id', flat=True)

        # Subquery: sum views per channel, isolated from other joins
        visits_sq = (
            ChannelVisit.objects
            .filter(channel=OuterRef('pk'))
            .values('channel')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # NEW: total episodes per channel via subquery
        episodes_sq = (
            Episode.objects
            .filter(channel=OuterRef('pk'))
            .values('channel')
            .annotate(c=Count('*'))
            .values('c')[:1]
        )

        qs = (
            Channel.objects
            .filter(id__in=channel_ids)
            .annotate(
                total_views=Coalesce(Subquery(visits_sq, output_field=IntegerField()), 0),
                episode_count=Coalesce(Subquery(episodes_sq, output_field=IntegerField()), 0),
                favorites_count=Count(
                    'channel_interactions',
                    filter=Q(channel_interactions__followed=True),
                    distinct=True,
                ),
                notifications_count=Count(
                    'channel_interactions',
                    filter=Q(channel_interactions__notifications_enabled=True),
                    distinct=True,
                ),
                avg_rating=Avg('channel_interactions__rating'),
                # count of ratings (number of users who rated), not distinct rating *values*
                rating_count=Count(
                    'channel_interactions__user',
                    filter=Q(channel_interactions__rating__isnull=False),
                    distinct=True,
                ),
            )
            .order_by('channel_title')
        )
        return qs

    def get(self, request, *args, **kwargs):
        self.object_list = self.get_queryset()
        paginator = self.get_paginator(self.object_list, self.paginate_by)
        try:
            page_number = int(request.GET.get('page', 1))
        except ValueError:
            page_number = 1

        if request.GET.get('ajax') and page_number > paginator.num_pages:
            return HttpResponse('')
        return super().get(request, *args, **kwargs)

    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax'):
            return render(self.request, 'podcasts/favorites_list_items.html', context)
        return super().render_to_response(context, **response_kwargs)


class NotificationsListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/notifications_list.html'
    context_object_name = 'episodes'
    paginate_by = 10  # Adjust the number per page as desired

    def get_queryset(self):
        # channels this user enabled notifications for
        channel_ids = ChannelInteraction.objects.filter(
            user=self.request.user,
            notifications_enabled=True
        ).values_list('channel_id', flat=True)

        # subquery: total downloads per episode
        downloads_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        qs = (
            Episode.objects
            .filter(channel__id__in=channel_ids)
            .select_related('channel')
            .order_by('-publication_date')
            .annotate(
                bookmarks_count=Count(
                    'episode_interactions',
                    filter=Q(episode_interactions__bookmarked=True),
                    distinct=True
                ),
                comments_count=Count('comments', distinct=True),
                ep_avg_rating=Avg('episode_interactions__rating'),
                # count users who rated (not distinct rating values)
                ep_rating_count=Count(
                    'episode_interactions__user',
                    filter=Q(episode_interactions__rating__isnull=False),
                    distinct=True
                ),
                total_episode_views=Coalesce(Sum('episodevisit__count'), 0),
                total_downloads=Coalesce(Subquery(downloads_sq, output_field=IntegerField()), Value(0)),
                total_shares=Coalesce(Subquery(shares_sq, output_field=IntegerField()), Value(0)),
            )
        )
        return qs
    
    def get(self, request, *args, **kwargs):
        # Check if an AJAX request asks for a page beyond available pages.
        self.object_list = self.get_queryset()
        paginator = self.get_paginator(self.object_list, self.paginate_by)
        try:
            page_number = int(request.GET.get('page', 1))
        except ValueError:
            page_number = 1

        if request.GET.get('ajax') and page_number > paginator.num_pages:
            # Return an empty response so the infinite scroll JS knows there are no more items.
            return HttpResponse('')
        return super().get(request, *args, **kwargs)

    def render_to_response(self, context, **response_kwargs):
        # If this is an AJAX request, return only the partial (list items) template.
        if self.request.GET.get('ajax'):
            return render(self.request, 'podcasts/notifications_list_items.html', context)
        return super().render_to_response(context, **response_kwargs)


class BookmarksListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/bookmarks_list.html'
    context_object_name = 'episodes'
    paginate_by = 5

    def get_queryset(self):
        # Episodes this user bookmarked
        episode_ids = EpisodeInteraction.objects.filter(
            user=self.request.user,
            bookmarked=True
        ).values_list('episode_id', flat=True)

        # Subquery: total downloads per episode
        downloads_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        qs = (
            Episode.objects
            .filter(id__in=episode_ids)
            .select_related('channel')
            .order_by('-publication_date')
            .annotate(
                bookmarks_count=Count(
                    'episode_interactions',
                    filter=Q(episode_interactions__bookmarked=True),
                    distinct=True
                ),
                # Count comments once per comment
                comments_count=Count('comments', distinct=True),
                ep_avg_rating=Avg('episode_interactions__rating'),
                # Count of users who rated (fixes “distinct rating values” issue)
                ep_rating_count=Count(
                    'episode_interactions__user',
                    filter=Q(episode_interactions__rating__isnull=False),
                    distinct=True
                ),
                total_episode_views=Coalesce(Sum('episodevisit__count'), 0),
                # NEW: total downloads
                total_downloads=Coalesce(Subquery(downloads_sq, output_field=IntegerField()), Value(0)),
                total_shares=Coalesce(Subquery(shares_sq, output_field=IntegerField()), Value(0)),
            )
        )
        return qs

    def get(self, request, *args, **kwargs):
        self.object_list = self.get_queryset()
        paginator = self.get_paginator(self.object_list, self.paginate_by)
        try:
            page_number = int(request.GET.get('page', 1))
        except ValueError:
            page_number = 1
        if self.request.GET.get('ajax') and page_number > paginator.num_pages:
            return HttpResponse('')
        return super().get(request, *args, **kwargs)

    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax'):
            return render(self.request, 'podcasts/bookmarks_list_items.html', context)
        return super().render_to_response(context, **response_kwargs)


class ContributeView(LoginRequiredMixin, TemplateView):
    template_name = 'podcasts/contribute.html'

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        # You can filter here if you only want channels the user created:
        # ctx['channels'] = Channel.objects.filter(created_by=self.request.user)
        ctx['channels'] = Channel.objects.all()
        return ctx
    
@login_required
def support_ticket(request):
    success     = False
    error_code  = None

    # 1) Figure out the user’s cap and current open count
    limit      = getattr(request.user, 'support_ticket_limit', 5)
    open_count = SupportTicket.objects.filter(
        user=request.user,
        status__in=['pending', 'in_progress']
    ).count()

    if request.method == 'POST':
        # 2) Block new tickets if they have no slots
        if open_count >= limit:
            error_code = 'limit'
            form = SupportTicketForm(request.POST, request.FILES)
        else:
            form = SupportTicketForm(request.POST, request.FILES)
            if form.is_valid():
                # Save the ticket
                ticket      = form.save(commit=False)
                ticket.user = request.user
                ticket.save()

                # Save the single attachment if present
                f = form.cleaned_data.get('attachment')
                if f:
                    SupportTicketAttachment.objects.create(
                        ticket=ticket,
                        file=f
                    )

                success = True
                form    = SupportTicketForm()  # reset form for next submission
                # Since we created one, increment open_count so the UI shows updated
                open_count += 1

            else:
                # Inspect only the attachment errors
                errs = form.errors.get('attachment')
                if errs:
                    text     = ' '.join(errs)
                    size_err = '2 MB' in text
                    type_err = any(ext in text.upper() for ext in ('JPG','PNG','GIF'))
                    if size_err and type_err:
                        error_code = 'both'
                    elif size_err:
                        error_code = 'size'
                    else:
                        error_code = 'type'
    else:
        form = SupportTicketForm()

    tickets = request.user.support_tickets.order_by('-submission_date')
    return render(request, 'podcasts/support_ticket.html', {
        'form':       form,
        'tickets':    tickets,
        'success':    success,
        'error_code': error_code,
        'open_count': open_count,
        'limit':      limit,
    })

@login_required
@require_POST
def update_episode_rating(request, episode_id):
    """
    Accept either Episode.pk (base) OR EpisodeTranslations.pk (display).
    Always normalize to the base Episode for storage & aggregation.
    Returns: {"rating": int, "avg_rating": float, "rating_count": int}
    """
    # 1) Parse rating (1..5)
    try:
        # handles FormData or JSON
        rating = request.POST.get('rating') or (request.body and __import__('json').loads(request.body).get('rating'))
        rating = int(rating)
        if rating < 1 or rating > 5:
            raise ValueError
    except Exception:
        return JsonResponse({"error": "Invalid rating"}, status=400)

    # 2) Normalize episode id to the base Episode
    from podcasts.models import Episode, EpisodeTranslations, EpisodeInteraction  # adjust import path
    base = None
    try:
        base = Episode.objects.select_related('channel').get(pk=episode_id)
    except Episode.DoesNotExist:
        try:
            tr = EpisodeTranslations.objects.select_related('episode', 'episode__channel').get(pk=episode_id)
            base = tr.episode
        except EpisodeTranslations.DoesNotExist:
            raise Http404("Episode not found")

    # 3) Upsert user's interaction row
    ei, _ = EpisodeInteraction.objects.get_or_create(user=request.user, episode=base, defaults={"rating": rating})
    if ei.rating != rating:
        ei.rating = rating
        ei.save(update_fields=["rating"])

    # 4) Aggregate fresh numbers
    agg = EpisodeInteraction.objects.filter(episode=base, rating__isnull=False).aggregate(
        avg=Avg('rating'), cnt=Count('rating')
    )
    avg_rating   = float(agg['avg'] or 0.0)  # ensure JSON numbers, not Decimals/None
    rating_count = int(agg['cnt'] or 0)

    # 5) Respond
    return JsonResponse({
        "rating": rating,
        "avg_rating": avg_rating,
        "rating_count": rating_count,
    })

@login_required
@require_POST
def update_channel_rating(request, channel_id):
    ch = get_object_or_404(Channel, pk=channel_id)
    try:
        rating = request.POST.get('rating') or (request.body and json.loads(request.body).get('rating'))
        rating = int(rating)
        if rating < 1 or rating > 5:
            raise ValueError
    except Exception:
        return JsonResponse({"error": "Invalid rating"}, status=400)

    ei, _ = ChannelInteraction.objects.get_or_create(user=request.user, channel=ch, defaults={"rating": rating})
    if ei.rating != rating:
        ei.rating = rating
        ei.save(update_fields=["rating"])

    agg = ChannelInteraction.objects.filter(channel=ch, rating__isnull=False).aggregate(
        avg=Avg('rating'), cnt=Count('rating')
    )
    return JsonResponse({
        "rating": rating,
        "avg_rating": float(agg["avg"] or 0.0),
        "rating_count": int(agg["cnt"] or 0),
    })

@login_required
@require_POST
def channel_rating_summary(request, channel_id):
    ch = get_object_or_404(Channel, pk=channel_id)
    agg = ChannelInteraction.objects.filter(channel=ch, rating__isnull=False).aggregate(
        avg=Avg('rating'), cnt=Count('rating')
    )
    return JsonResponse({
        "avg_rating": float(agg["avg"] or 0.0),
        "rating_count": int(agg["cnt"] or 0),
    })