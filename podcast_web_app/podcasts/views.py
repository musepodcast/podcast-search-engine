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
from allauth.account.models import EmailAddress
from django.conf import settings
import logging, time
from collections import Counter
import re, difflib, unicodedata
import sys
import json
import itertools
import requests
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
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_POST, require_GET, require_http_methods

from podcasts.search.documents import EpisodeDocument, TranscriptDocument, EpisodeTranslationsDocument
from elasticsearch_dsl import Q as ES_Q
from elasticsearch_dsl.connections import connections

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
from django.db.models import (
    Avg, Case, Count, DateTimeField, F, FloatField, IntegerField, OuterRef,
    Q, Subquery, Sum, Value, When
)
from elastic_transport import ConnectionTimeout
from urllib3.exceptions import ReadTimeoutError


_LANG_ALIAS_TO_CODE = {
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

_slug_non_alnum = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

def canon_lang(lang: str) -> str:
    """Collapse locale variants to your folder/index codes."""
    if not lang:
        return "en"
    s = lang.strip().lower().replace("_", "-")
    return _LANG_ALIAS_TO_CODE.get(s, _LANG_ALIAS_TO_CODE.get(s.split("-")[0], s.split("-")[0]))

def slug_norm(s: str) -> str:
    """Diacritic-insensitive, lower, non-alnum→_ normalizer for URL slugs/filenames."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _slug_non_alnum.sub("_", s)
    return re.sub(r"_+", "_", s).strip("_")

def slug_norm(s: str) -> str:
    """Diacritic-insensitive, lower, non-alnum→_ normalizer for episode slugs."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _slug_non_alnum.sub("_", s)
    return re.sub(r"_+", "_", s).strip("_")

_assistant_token_pattern = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_assistant_stopwords = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "there", "they", "this", "to", "was",
    "were", "what", "when", "where", "who", "why", "with", "would", "you", "your",
    "episode", "podcast", "transcript", "about", "tell",
}

def _assistant_tokenize(text: str) -> list[str]:
    return _assistant_token_pattern.findall((text or "").lower())

def _assistant_filtered_tokens(text: str) -> list[str]:
    return [
        tok for tok in _assistant_tokenize(text)
        if len(tok) > 1 and tok not in _assistant_stopwords
    ]

def _assistant_transcript_queryset(base_episode, lang_code: str):
    if lang_code != "en":
        translated = TranscriptTranslations.objects.filter(
            episode=base_episode,
            language__istartswith=lang_code,
        ).order_by("segment_time")
        if translated.exists():
            return translated
    return Transcript.objects.filter(episode=base_episode).order_by("segment_time")

def _assistant_segment_time(seg) -> str:
    return (getattr(seg, "segment_time", "") or "").strip()

def _assistant_segment_text(seg) -> str:
    return (getattr(seg, "segment_text", "") or "").strip()

def _assistant_block_time(block_segments) -> str:
    if not block_segments:
        return ""

    start_raw = _assistant_segment_time(block_segments[0])
    end_raw = _assistant_segment_time(block_segments[-1])

    start = start_raw.split(" - ", 1)[0].strip() if start_raw else ""
    end = end_raw.rsplit(" - ", 1)[-1].strip() if end_raw else ""

    if start and end and start != end:
        return f"{start} - {end}"
    return start or end or start_raw or end_raw

def _assistant_format_context(block_segments) -> str:
    lines = []
    for seg in block_segments:
        seg_time = _assistant_segment_time(seg)
        seg_text = _assistant_segment_text(seg)
        speaker = (getattr(seg, "speaker", "") or "").strip()
        if not seg_text:
            continue
        if speaker:
            lines.append(f"[{seg_time}] {speaker}: {seg_text}")
        else:
            lines.append(f"[{seg_time}] {seg_text}")
    return "\n".join(lines)

def _assistant_score_text(question_tokens: list[str], text: str) -> float:
    block_tokens = _assistant_filtered_tokens(text)
    if not block_tokens:
        return 0.0

    question_unique = list(dict.fromkeys(question_tokens))
    block_unique = list(dict.fromkeys(block_tokens))
    block_set = set(block_unique)

    exact_overlap = sum(1 for tok in question_unique if tok in block_set)

    fuzzy_overlap = 0.0
    for tok in question_unique:
        if tok in block_set or len(tok) < 4:
            continue
        if difflib.get_close_matches(tok, block_unique, n=1, cutoff=0.82):
            fuzzy_overlap += 0.8

    coverage = exact_overlap / max(1, len(question_unique))
    question_phrase = " ".join(question_unique)
    block_phrase = " ".join(block_unique[:160])
    phrase_ratio = difflib.SequenceMatcher(None, question_phrase, block_phrase).ratio()

    return (exact_overlap * 4.0) + (fuzzy_overlap * 2.0) + (coverage * 3.0) + phrase_ratio

def _assistant_pick_segments(question: str, segments, limit: int):
    if not segments:
        return []

    question_tokens = _assistant_filtered_tokens(question)
    if not question_tokens:
        question_tokens = _assistant_tokenize(question)

    radius = 1
    ranked_ranges = []

    for idx in range(len(segments)):
        start = max(0, idx - radius)
        end = min(len(segments), idx + radius + 1)
        neighborhood = segments[start:end]
        neighborhood_text = " ".join(_assistant_segment_text(seg) for seg in neighborhood)
        score = _assistant_score_text(question_tokens, neighborhood_text)
        ranked_ranges.append({
            "start": start,
            "end": end - 1,
            "score": score,
        })

    ranked_ranges.sort(key=lambda item: (item["score"], -item["start"]), reverse=True)

    selected_ranges = []
    for item in ranked_ranges:
        if item["score"] <= 0 and selected_ranges:
            break

        overlaps_existing = any(
            not (item["end"] < current["start"] - 1 or item["start"] > current["end"] + 1)
            for current in selected_ranges
        )
        if overlaps_existing:
            continue

        selected_ranges.append(item)
        if len(selected_ranges) >= limit:
            break

    if not selected_ranges:
        fallback_end = min(len(segments), max(1, limit))
        selected_ranges = [{"start": 0, "end": fallback_end - 1, "score": 0.0}]

    selected_ranges.sort(key=lambda item: item["start"])

    merged_ranges = []
    for item in selected_ranges:
        if not merged_ranges or item["start"] > merged_ranges[-1]["end"] + 1:
            merged_ranges.append(item.copy())
            continue
        merged_ranges[-1]["end"] = max(merged_ranges[-1]["end"], item["end"])
        merged_ranges[-1]["score"] = max(merged_ranges[-1]["score"], item["score"])

    blocks = []
    for item in merged_ranges:
        block_segments = segments[item["start"]:item["end"] + 1]
        context = _assistant_format_context(block_segments)
        if not context:
            continue
        blocks.append({
            "time": _assistant_block_time(block_segments),
            "speaker": "",
            "text": " ".join(_assistant_segment_text(seg) for seg in block_segments if _assistant_segment_text(seg)),
            "context": context,
            "score": item["score"],
        })

    return blocks

@require_POST
def episode_assistant_chat(request, episode_id):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    question = (payload.get("message") or "").strip()
    if not question:
        return JsonResponse({"error": "Message is required."}, status=400)

    if len(question) > 500:
        return JsonResponse({"error": "Message is too long."}, status=400)

    base_episode = get_object_or_404(Episode.objects.select_related("channel"), pk=episode_id)
    lang_code = canon_lang(get_selected_language(request) or "en")
    segments = list(_assistant_transcript_queryset(base_episode, lang_code))

    if not segments:
        return JsonResponse({"error": "No transcript available for this episode."}, status=404)

    selected_segments = _assistant_pick_segments(
        question=question,
        segments=segments,
        limit=max(1, settings.EPISODE_ASSISTANT_MAX_SEGMENTS),
    )

    if not selected_segments:
        return JsonResponse({"error": "No transcript available for this episode."}, status=404)

    transcript_context = "\n\n".join(
        f"Context block [{item['time']}]\n{item['context']}"
        for item in selected_segments
    )

    system_prompt = (
        "You are an assistant for a podcast episode page. "
        "Answer only from the provided transcript context. "
        "If the answer is not supported by the transcript context, say you cannot find it in this episode transcript. "
        "Use the names and facts exactly as they appear in the transcript context. "
        "Keep the answer concise and mention timestamps when useful."
    )

    user_prompt = (
        f"Episode title: {base_episode.episode_title}\n"
        f"Channel: {base_episode.channel.channel_title}\n\n"
        f"Transcript context:\n{transcript_context}\n\n"
        f"Question: {question}"
    )

    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": 0.2,
                },
            },
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("error") or "Local AI model is unavailable right now."
            except ValueError:
                error_message = "Local AI model is unavailable right now."
            return JsonResponse({"error": error_message}, status=503)
        data = response.json()
    except (requests.RequestException, ValueError):
        return JsonResponse(
            {"error": "Local AI model is unavailable right now."},
            status=503,
        )

    answer = (data.get("message") or {}).get("content", "").strip()
    if not answer:
        answer = "I could not generate an answer from this episode transcript."

    return JsonResponse({
        "answer": answer,
        "sources": [
            {"time": item["time"], "speaker": item["speaker"]}
            for item in selected_segments[:3]
        ],
    })

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
    raw = request.GET.get('lang', getattr(request, 'LANGUAGE_CODE', 'en'))
    return canon_lang(raw)   # <- returns 'en', 'pt', 'es', 'cn', 'tw', etc.



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
        
        lang = get_selected_language(self.request)
        if lang != 'en':
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
    TITLE_PHRASE_BONUS = 100  # exact phrase in title

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
        lang = get_selected_language(self.request)

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

        tokens  = self._tokenize(q)
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
                # Load all titles for this channel in one go (per-channel is usually small)
                title_rows = base.episodes.values('id', 'episode_title')
            else:
                title_rows = EpisodeTranslations.objects.filter(
                    episode__channel=base, language=lang, translated=True
                ).values('episode_id', 'episode_title')

            title_overlap_map = {}  # episode_id -> number of tokens that appear in the title
            phrase_title_ids  = set()

            for row in title_rows:
                eid   = row.get('id') or row.get('episode_id')
                ttext = (row['episode_title'] or '').lower()

                # exact phrase match (contiguous substring of full query)
                if q and q.lower() in ttext:
                    phrase_title_ids.add(eid)

                # count how many tokens appear anywhere in the title
                overlap = sum(1 for tok in tokens if tok in ttext)
                if overlap > 0:
                    title_overlap_map[eid] = overlap


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

            candidate_ids = (
                {eid for eid, _ in es_hits}
                | {eid for eid, _ in epi_hits}
                | set(phrase_title_ids)
                | set(title_overlap_map.keys())
            )
            if not candidate_ids:
                paginator = Paginator(range(0), page_size)
                try:
                    page_obj = paginator.page(page_num)
                except EmptyPage:
                    page_obj = paginator.page(1)
                ctx['episodes'] = []
                return ctx


            # (3) NEW RANKING: count occurrences per episode, then 100 + 10*k
            occ_map = self._occurrence_counts(candidate_ids, tokens)  # {eid: count}
            score_map = {}
            for eid in candidate_ids:
                occ = int(occ_map.get(eid, 0))

                # transcript contribution
                score = self.PER_OCC_BONUS * occ  # 10 * occurrences

                # title contributions
                overlap = title_overlap_map.get(eid, 0)
                score += overlap * self.TITLE_BONUS  # e.g. 3 matching words => +120

                if eid in phrase_title_ids:
                    score += self.TITLE_PHRASE_BONUS  # strong boost if full query phrase appears

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
        lang    = get_selected_language(self.request)

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
        lang_raw  = get_selected_language(self.request) or 'en'
        lang_code = lang_raw.lower().split('-', 1)[0]

        if lang_code != 'en':
            tr = EpisodeTranslations.objects.select_related('episode', 'episode__channel').filter(
                episode=base,
                language__istartswith=lang_code,
                translated=True
            ).first()
            if tr:
                return tr

        return base



    def get_queryset(self):
        lang_raw  = (get_selected_language(self.request) or 'en')
        lang_code = lang_raw.lower().split('-', 1)[0]

        if lang_code == 'en':
            return Episode.objects.select_related('channel')
        return EpisodeTranslations.objects.filter(language__istartswith=lang_code, translated=True)

    def get_context_data(self, **kwargs):
        ctx  = super().get_context_data(**kwargs)
        disp = self.display_episode
        base = self.base_episode

        lang_raw  = (get_selected_language(self.request) or 'en')
        lang_code = lang_raw.lower().split('-', 1)[0]


        # 1) TRANSCRIPTS
        if isinstance(disp, EpisodeTranslations):
            tr_qs = TranscriptTranslations.objects.filter(
                episode=base,
                language__istartswith=lang_code,
            ).order_by("segment_time")

            segments = tr_qs if tr_qs.exists() else Transcript.objects.filter(
                episode=base
            ).order_by("segment_time")
        else:
            segments = Transcript.objects.filter(episode=base).order_by("segment_time")


        # 2) CHAPTERS
        if isinstance(disp, EpisodeTranslations):
            ch_qs = ChapterTranslations.objects.filter(
                episode=base,
                language__istartswith=lang_code,
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
        # per-language downloads
        try:
            canon_lang = _canon_lang(lang_raw)   # or lang_code
        except NameError:
            canon_lang = lang_code

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
        ctx['selected_language'] = lang_code
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

    ALLOWED_SORTS = {'trending', 'recent', 'views', 'bookmarks', 'comments', 'stars', 'downloaded', 'shared', 'title'}
    ALLOWED_DIRS = {'asc', 'desc'}

    def _parse_sort(self):
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
        dt_24h  = now - timedelta(hours=24)
        dt_7d   = now - timedelta(days=7)
        dt_30d  = now - timedelta(days=30)
        dt_365d = now - timedelta(days=365)

        old_pub_default = timezone.make_aware(datetime(1970, 1, 1))

        # ============================================================
        # Views subqueries (Episode base)
        # ============================================================
        views_total_sq = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_24h_sq = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('pk'), last_visited__gte=dt_24h)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_7d_sq = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('pk'), last_visited__gte=dt_7d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_30d_sq = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('pk'), last_visited__gte=dt_30d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_365d_sq = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('pk'), last_visited__gte=dt_365d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # ============================================================
        # Views subqueries (Translations: OuterRef('episode'))
        # ============================================================
        views_total_sq_tr = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('episode'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_24h_sq_tr = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('episode'), last_visited__gte=dt_24h)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_7d_sq_tr = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('episode'), last_visited__gte=dt_7d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_30d_sq_tr = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('episode'), last_visited__gte=dt_30d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        views_365d_sq_tr = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('episode'), last_visited__gte=dt_365d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # ============================================================
        # Downloads/Shares totals (base + translations)
        # ============================================================
        downloads_total_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_total_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        downloads_total_sq_tr = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('episode'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_total_sq_tr = (
            EpisodeShare.objects
            .filter(episode=OuterRef('episode'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # ============================================================
        # Downloads windows via subquery (avoid JOIN multiplication)
        # ============================================================
        downloads_7d_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'), last_downloaded__gte=dt_7d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        downloads_30d_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'), last_downloaded__gte=dt_30d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        downloads_365d_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'), last_downloaded__gte=dt_365d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        downloads_7d_sq_tr = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('episode'), last_downloaded__gte=dt_7d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        downloads_30d_sq_tr = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('episode'), last_downloaded__gte=dt_30d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        downloads_365d_sq_tr = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('episode'), last_downloaded__gte=dt_365d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # ============================================================
        # Shares windows via subquery (avoid JOIN multiplication)
        # ============================================================
        shares_7d_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'), last_shared__gte=dt_7d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_30d_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'), last_shared__gte=dt_30d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_365d_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'), last_shared__gte=dt_365d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        shares_7d_sq_tr = (
            EpisodeShare.objects
            .filter(episode=OuterRef('episode'), last_shared__gte=dt_7d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_30d_sq_tr = (
            EpisodeShare.objects
            .filter(episode=OuterRef('episode'), last_shared__gte=dt_30d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
        shares_365d_sq_tr = (
            EpisodeShare.objects
            .filter(episode=OuterRef('episode'), last_shared__gte=dt_365d)
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # ============================================================
        # EN branch (Episode)
        # ============================================================
        if lang in ('en', 'en-us'):
            qs = (
                Episode.objects
                .select_related('channel')
                .prefetch_related('transcripts', 'chapters')
                .annotate(
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

                    total_episode_views=Coalesce(Subquery(views_total_sq, output_field=IntegerField()), Value(0)),
                    total_downloads=Coalesce(Subquery(downloads_total_sq, output_field=IntegerField()), Value(0)),
                    total_shares=Coalesce(Subquery(shares_total_sq, output_field=IntegerField()), Value(0)),

                    # views windows
                    views_24h=Coalesce(Subquery(views_24h_sq, output_field=IntegerField()), Value(0)),
                    views_7d=Coalesce(Subquery(views_7d_sq, output_field=IntegerField()), Value(0)),
                    views_30d=Coalesce(Subquery(views_30d_sq, output_field=IntegerField()), Value(0)),
                    views_365d=Coalesce(Subquery(views_365d_sq, output_field=IntegerField()), Value(0)),

                    # downloads windows (subquery = correct even with other joins)
                    downloads_7d=Coalesce(Subquery(downloads_7d_sq, output_field=IntegerField()), Value(0)),
                    downloads_30d=Coalesce(Subquery(downloads_30d_sq, output_field=IntegerField()), Value(0)),
                    downloads_365d=Coalesce(Subquery(downloads_365d_sq, output_field=IntegerField()), Value(0)),

                    # shares windows
                    shares_7d=Coalesce(Subquery(shares_7d_sq, output_field=IntegerField()), Value(0)),
                    shares_30d=Coalesce(Subquery(shares_30d_sq, output_field=IntegerField()), Value(0)),
                    shares_365d=Coalesce(Subquery(shares_365d_sq, output_field=IntegerField()), Value(0)),

                    # comments windows (Count distinct is safe)
                    comments_7d=Coalesce(
                        Count('comments', filter=Q(comments__created_at__gte=dt_7d), distinct=True),
                        0
                    ),
                    comments_30d=Coalesce(
                        Count('comments', filter=Q(comments__created_at__gte=dt_30d), distinct=True),
                        0
                    ),
                    comments_365d=Coalesce(
                        Count('comments', filter=Q(comments__created_at__gte=dt_365d), distinct=True),
                        0
                    ),
                )
                .annotate(
                    # Windowed weighted scores
                    trending_score_7d=(
                        F('views_7d') * 1.0 +
                        F('downloads_7d') * 3.0 +
                        F('shares_7d') * 4.0 +
                        F('comments_7d') * 2.0 +
                        F('ep_rating_count') * 1.0
                    ),
                    trending_score_30d=(
                        F('views_30d') * 1.0 +
                        F('downloads_30d') * 3.0 +
                        F('shares_30d') * 4.0 +
                        F('comments_30d') * 2.0 +
                        F('ep_rating_count') * 1.0
                    ),
                    trending_score_365d=(
                        F('views_365d') * 1.0 +
                        F('downloads_365d') * 3.0 +
                        F('shares_365d') * 4.0 +
                        F('comments_365d') * 2.0 +
                        F('ep_rating_count') * 1.0
                    ),
                    trending_score_all=(
                        F('total_episode_views') * 1.0 +
                        F('total_downloads') * 3.0 +
                        F('total_shares') * 4.0 +
                        F('comments_count') * 2.0 +
                        F('ep_rating_count') * 1.0
                    ),
                )
                .annotate(
                    pub_date=F('publication_date'),
                    trend_bucket=Case(
                        When(publication_date__isnull=True, then=Value(4)),   # unknown last
                        When(publication_date__gte=dt_7d, then=Value(0)),      # <= 7d
                        When(publication_date__gte=dt_30d, then=Value(1)),     # 8-30d
                        When(publication_date__gte=dt_365d, then=Value(2)),    # 31-365d
                        default=Value(3),                                      # >365d
                        output_field=IntegerField(),
                    ),
                )
                .annotate(
                    # choose correct score by bucket
                    trending_score=Case(
                        When(trend_bucket=0, then=F('trending_score_7d')),
                        When(trend_bucket=1, then=F('trending_score_30d')),
                        When(trend_bucket=2, then=F('trending_score_365d')),
                        default=F('trending_score_all'),
                        output_field=FloatField(),
                    )
                )
                .annotate(
                    views_24h_sort=Case(
                        When(trend_bucket__lt=3, then=F('views_24h')),
                        default=Value(0),
                        output_field=IntegerField(),
                    ),
                    pub_sort=Case(
                        When(trend_bucket__lt=3, then=F('pub_date')),
                        default=Value(old_pub_default),  # bucket 3 ignores recency
                        output_field=DateTimeField(),
                    ),
                    old_views_sort=Case(
                        When(trend_bucket__gte=3, then=F('total_episode_views')),
                        default=Value(0),
                        output_field=IntegerField(),
                    ),
                )
            )

            if sort == 'trending':
                # bucket order fixed; direction affects ranking inside buckets
                if direction == 'desc':
                    ordering = [
                        'trend_bucket',
                        F('trending_score').desc(nulls_last=True),
                        F('views_24h_sort').desc(nulls_last=True),
                        F('pub_sort').desc(nulls_last=True),
                        F('old_views_sort').desc(nulls_last=True),
                        'episode_title',
                    ]
                else:
                    ordering = [
                        'trend_bucket',
                        F('trending_score').asc(nulls_last=True),
                        F('views_24h_sort').asc(nulls_last=True),
                        F('pub_sort').desc(nulls_last=True),   # keep newest first when tied
                        F('old_views_sort').asc(nulls_last=True),
                        'episode_title',
                    ]

            elif sort == 'recent':
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
                ordering = [f'{dir_prefix}episode_title']

            else:
                ordering = [F('publication_date').desc(nulls_last=True), 'episode_title']

            return qs.order_by(*ordering)

        # ============================================================
        # Non-English branch (EpisodeTranslations)
        # ============================================================
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

                # Use translation date if present, else episode date
                pub_date=Coalesce(F('publication_date'), F('episode__publication_date')),

                ep_avg_rating=Avg('episode__episode_interactions__rating'),
                ep_rating_count=Count(
                    'episode__episode_interactions__user',
                    filter=Q(episode__episode_interactions__rating__isnull=False),
                    distinct=True
                ),

                total_downloads=Coalesce(Subquery(downloads_total_sq_tr, output_field=IntegerField()), Value(0)),
                total_shares=Coalesce(Subquery(shares_total_sq_tr, output_field=IntegerField()), Value(0)),
                total_episode_views=Coalesce(Subquery(views_total_sq_tr, output_field=IntegerField()), Value(0)),

                # views windows
                views_24h=Coalesce(Subquery(views_24h_sq_tr, output_field=IntegerField()), Value(0)),
                views_7d=Coalesce(Subquery(views_7d_sq_tr, output_field=IntegerField()), Value(0)),
                views_30d=Coalesce(Subquery(views_30d_sq_tr, output_field=IntegerField()), Value(0)),
                views_365d=Coalesce(Subquery(views_365d_sq_tr, output_field=IntegerField()), Value(0)),

                # downloads windows
                downloads_7d=Coalesce(Subquery(downloads_7d_sq_tr, output_field=IntegerField()), Value(0)),
                downloads_30d=Coalesce(Subquery(downloads_30d_sq_tr, output_field=IntegerField()), Value(0)),
                downloads_365d=Coalesce(Subquery(downloads_365d_sq_tr, output_field=IntegerField()), Value(0)),

                # shares windows
                shares_7d=Coalesce(Subquery(shares_7d_sq_tr, output_field=IntegerField()), Value(0)),
                shares_30d=Coalesce(Subquery(shares_30d_sq_tr, output_field=IntegerField()), Value(0)),
                shares_365d=Coalesce(Subquery(shares_365d_sq_tr, output_field=IntegerField()), Value(0)),

                # comments windows (Count distinct is safe)
                comments_7d=Coalesce(
                    Count('episode__comments', filter=Q(episode__comments__created_at__gte=dt_7d), distinct=True),
                    0
                ),
                comments_30d=Coalesce(
                    Count('episode__comments', filter=Q(episode__comments__created_at__gte=dt_30d), distinct=True),
                    0
                ),
                comments_365d=Coalesce(
                    Count('episode__comments', filter=Q(episode__comments__created_at__gte=dt_365d), distinct=True),
                    0
                ),
            )
            .annotate(
                trending_score_7d=(
                    F('views_7d') * 1.0 +
                    F('downloads_7d') * 3.0 +
                    F('shares_7d') * 4.0 +
                    F('comments_7d') * 2.0 +
                    F('ep_rating_count') * 1.0
                ),
                trending_score_30d=(
                    F('views_30d') * 1.0 +
                    F('downloads_30d') * 3.0 +
                    F('shares_30d') * 4.0 +
                    F('comments_30d') * 2.0 +
                    F('ep_rating_count') * 1.0
                ),
                trending_score_365d=(
                    F('views_365d') * 1.0 +
                    F('downloads_365d') * 3.0 +
                    F('shares_365d') * 4.0 +
                    F('comments_365d') * 2.0 +
                    F('ep_rating_count') * 1.0
                ),
                trending_score_all=(
                    F('total_episode_views') * 1.0 +
                    F('total_downloads') * 3.0 +
                    F('total_shares') * 4.0 +
                    F('comments_count') * 2.0 +
                    F('ep_rating_count') * 1.0
                ),
            )
            .annotate(
                trend_bucket=Case(
                    When(pub_date__isnull=True, then=Value(4)),
                    When(pub_date__gte=dt_7d, then=Value(0)),
                    When(pub_date__gte=dt_30d, then=Value(1)),
                    When(pub_date__gte=dt_365d, then=Value(2)),
                    default=Value(3),
                    output_field=IntegerField(),
                ),
            )
            .annotate(
                trending_score=Case(
                    When(trend_bucket=0, then=F('trending_score_7d')),
                    When(trend_bucket=1, then=F('trending_score_30d')),
                    When(trend_bucket=2, then=F('trending_score_365d')),
                    default=F('trending_score_all'),
                    output_field=FloatField(),
                ),
                views_24h_sort=Case(
                    When(trend_bucket__lt=3, then=F('views_24h')),
                    default=Value(0),
                    output_field=IntegerField(),
                ),
                pub_sort=Case(
                    When(trend_bucket__lt=3, then=F('pub_date')),
                    default=Value(old_pub_default),
                    output_field=DateTimeField(),
                ),
                old_views_sort=Case(
                    When(trend_bucket__gte=3, then=F('total_episode_views')),
                    default=Value(0),
                    output_field=IntegerField(),
                ),
            )
        )

        if sort == 'trending':
            if direction == 'desc':
                ordering = [
                    'trend_bucket',
                    F('trending_score').desc(nulls_last=True),
                    F('views_24h_sort').desc(nulls_last=True),
                    F('pub_sort').desc(nulls_last=True),
                    F('old_views_sort').desc(nulls_last=True),
                    'episode_title',
                ]
            else:
                ordering = [
                    'trend_bucket',
                    F('trending_score').asc(nulls_last=True),
                    F('views_24h_sort').asc(nulls_last=True),
                    F('pub_sort').desc(nulls_last=True),
                    F('old_views_sort').asc(nulls_last=True),
                    'episode_title',
                ]

        elif sort == 'recent':
            # IMPORTANT: use pub_date (translation date fallback to episode date)
            ordering = [
                F('pub_date').desc(nulls_last=True) if direction == 'desc'
                else F('pub_date').asc(nulls_last=True),
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
            ordering = [F('pub_date').desc(nulls_last=True), 'episode_title']

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
        # per-episode downloads
        downloads_sq = (
            EpisodeDownload.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # per-episode shares
        shares_sq = (
            EpisodeShare.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )

        # ✅ per-episode views (fixes inflation)
        views_sq = (
            EpisodeVisit.objects
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

                # ✅ NEW (matches EpisodeDetailView semantics)
                total_episode_views=Coalesce(
                    Subquery(views_sq, output_field=IntegerField()),
                    Value(0)
                ),

                total_downloads=Coalesce(
                    Subquery(downloads_sq, output_field=IntegerField()),
                    Value(0)
                ),
                total_shares=Coalesce(
                    Subquery(shares_sq, output_field=IntegerField()),
                    Value(0)
                ),
            )
        )
    def _decorate_items_for_episode_links(self, items, selected_langs):
        """
        Adds:
        - item.base_slug_ch  (channel slug for URL)
        - item.base_slug_ep  (episode base slug for URL)
        - item.link_lang     (lang query param that matches the row being displayed)
        """
        from .models import EpisodeTranslations

        def _norm(x):
            return (x or "").strip().lower().replace("_", "-")

        def _is_en(x):
            x = _norm(x)
            return x == "en" or x.startswith("en-") or x in ("eng", "english")

        langs = [_norm(l) for l in (selected_langs or []) if (l or "").strip()]
        non_en = [l for l in langs if l and not _is_en(l)]
        multi_mode = (len(langs) > 1 and len(non_en) >= 1)

        single_lang = non_en[0] if non_en else "en"

        for obj in items:
            # Translation row (multi-language expand mode)
            if isinstance(obj, EpisodeTranslations) or (hasattr(obj, "episode") and getattr(obj, "episode", None) is not None):
                base_ep = obj.episode
                obj.base_slug_ch = base_ep.channel.sanitized_channel_title
                obj.base_slug_ep = base_ep.sanitized_episode_title
                obj.link_lang = (getattr(obj, "language", None) or single_lang or "en")

                # Optional: makes templates consistent
                obj.channel = base_ep.channel
                obj.sanitized_episode_title = base_ep.sanitized_episode_title

            else:
                # Base Episode row
                obj.base_slug_ch = obj.channel.sanitized_channel_title
                obj.base_slug_ep = obj.sanitized_episode_title

                # Multi-mode shows EN base rows; single-lang overlay should open in that lang
                obj.link_lang = ("en" if multi_mode else single_lang)



    def _expand_multi_language_items(self, ep_qs, selected_langs):
        """
        Turn an Episode queryset into a list of items that can include:
        - EpisodeTranslations rows for each selected non-English language
        - Episode rows if English is selected

        This is what you need if you want to *see multiple languages at once*,
        not just choose one display overlay.
        """
        EN_ALIASES = {"en", "eng", "english"}

        langs = [l.strip().lower() for l in (selected_langs or []) if (l or "").strip()]
        wants_en = any(l in EN_ALIASES for l in langs)
        non_en = [l for l in langs if l not in EN_ALIASES]

        # Evaluate episodes once (these already have your stats annotations)
        base_eps = list(ep_qs)
        if not base_eps:
            return []

        # If user only wants English, we're done
        if not non_en:
            return base_eps if wants_en else base_eps  # (base_eps should already be English filtered upstream)

        ep_by_id = {e.id: e for e in base_eps}
        ep_ids = list(ep_by_id.keys())

        # Pull translations for ALL selected non-English languages
        q_lang = Q()
        for lang in non_en:
            q_lang |= Q(language=lang) | Q(language__istartswith=f"{lang}-")

        t_qs = (
            EpisodeTranslations.objects
            .filter(episode_id__in=ep_ids)
            .filter(q_lang)
            .select_related("episode", "episode__channel")
        )


        et_fields = {f.name for f in EpisodeTranslations._meta.get_fields()}
        if "translated" in et_fields:
            t_qs = t_qs.filter(translated=True)

        trans_items = list(t_qs)

        # Copy the per-episode stats from the base Episode objects onto translation objects
        stats_fields = (
            "bookmarks_count",
            "comments_count",
            "ep_avg_rating",
            "ep_rating_count",
            "total_episode_views",
            "total_downloads",
            "total_shares",
        )
        for t in trans_items:
            base = ep_by_id.get(t.episode_id)
            if base:
                for f in stats_fields:
                    setattr(t, f, getattr(base, f, 0))

        # Build a combined list:
        # - translations for each selected lang
        # - plus English base episodes if English selected
        combined = []
        combined.extend(trans_items)
        if wants_en:
            combined.extend(base_eps)

        # Sort: newest episode first; within same episode, respect selected language order; English last
        # Sort: newest episode first; within same episode, respect selected language order; English last
        lang_rank = {lang: i for i, lang in enumerate(non_en)}

        def _pub_dt(obj):
            if hasattr(obj, "episode") and getattr(obj, "episode", None) is not None:
                return obj.episode.publication_date
            return obj.publication_date

        def _master_id(obj):
            return obj.episode_id if hasattr(obj, "episode_id") else obj.id

        def _lang_order(obj):
            # translations first, in non_en order; base episode after
            if hasattr(obj, "language") and (obj.language or "").lower() in lang_rank:
                return lang_rank[(obj.language or "").lower()]
            return 9999  # base episode goes last

        combined.sort(
            key=lambda o: (
                _pub_dt(o) or datetime.min.replace(tzinfo=timezone.get_current_timezone()),
                _master_id(o),
                -_lang_order(o),   # we’ll reverse overall, so invert to keep translations first
            ),
            reverse=True
        )

        return combined

    def _apply_episode_display_language(self, qs, selected_langs):
        """
        Annotate Episodes with display_* fields, picking the "best" translation
        when user selects one OR multiple languages.

        Priority:
        - languages in the order user selected them (non-English only)
        - fallback to base Episode fields if no translation exists
        """
        EN_ALIASES = {"en", "eng", "english"}

        langs = [l.strip().lower() for l in (selected_langs or []) if (l or "").strip()]
        non_en = [l for l in langs if l not in EN_ALIASES]

        # If user didn't pick any non-English, do nothing
        if not non_en:
            return qs

        et_fields = {f.name for f in EpisodeTranslations._meta.get_fields()}

        # translation base queryset: translation row for this episode, limited to selected langs
        et_qs = EpisodeTranslations.objects.filter(
            episode_id=OuterRef("pk"),
            language__in=non_en,
        )

        if "translated" in et_fields:
            et_qs = et_qs.filter(translated=True)

        # We want deterministic choice when multiple langs:
        # pick translation row with language priority by selected order.
        # Use a CASE expression for ordering.


        whens = [When(language=lang, then=idx) for idx, lang in enumerate(non_en)]
        et_qs = et_qs.annotate(_prio=Case(*whens, default=9999, output_field=IntegerField())).order_by("_prio")

        # choose which translation fields exist
        title_field = "episode_title" if "episode_title" in et_fields else None

        desc_field = None
        for cand in ("description", "episode_description"):
            if cand in et_fields:
                desc_field = cand
                break

        slug_field = "sanitized_episode_title" if "sanitized_episode_title" in et_fields else None
        image_field = "image_url" if "image_url" in et_fields else None

        annotations = {}

        annotations["display_episode_title"] = (
            Coalesce(Subquery(et_qs.values(title_field)[:1]), F("episode_title"))
            if title_field else F("episode_title")
        )

        annotations["display_description"] = (
            Coalesce(Subquery(et_qs.values(desc_field)[:1]), F("description"))
            if desc_field else F("description")
        )

        annotations["display_sanitized_episode_title"] = (
            Coalesce(Subquery(et_qs.values(slug_field)[:1]), F("sanitized_episode_title"))
            if slug_field else F("sanitized_episode_title")
        )

        annotations["display_image_url"] = (
            Coalesce(Subquery(et_qs.values(image_field)[:1]), F("image_url"))
            if image_field else F("image_url")
        )

        return qs.annotate(**annotations)

    def _set_total_display_from_base_qs(self, base_qs, selected_langs):
        """
        Computes how many *display rows* exist when multi-language expand is enabled.
        Stores it on self for get_context_data().
        """
        def _norm_lang(x):
            return (x or "").strip().lower().replace("_", "-")

        def _is_english_code(x):
            x = _norm_lang(x)
            return x == "en" or x.startswith("en-") or x in ("eng", "english")

        langs = [_norm_lang(l) for l in (selected_langs or []) if (l or "").strip()]
        wants_en = any(_is_english_code(l) for l in langs)
        non_en = [l for l in langs if l and not _is_english_code(l)]

        base_total = base_qs.count()

        # default: if you aren't in multi-language mode, total == base episodes
        self._total_display_items = base_total

        # only adjust when user selected multiple languages AND at least one non-English
        if not (len(langs) > 1 and len(non_en) >= 1):
            return

        # count translation rows for these episodes and langs
        q_lang = Q()
        for lang in non_en:
            q_lang |= Q(language=lang) | Q(language__istartswith=f"{lang}-")

        t_qs = EpisodeTranslations.objects.filter(episode__in=base_qs).filter(q_lang)

        et_fields = {f.name for f in EpisodeTranslations._meta.get_fields()}
        if "translated" in et_fields:
            t_qs = t_qs.filter(translated=True)

        trans_total = t_qs.count()

        # display rows = translations + (optional) base episodes if English selected
        self._total_display_items = (base_total if wants_en else 0) + trans_total


    def paginate_queryset(self, qs, page_size):
        """
        Returns: (paginator, page_obj, object_list, has_other_pages)
        where object_list is a list of Episodes (or Channels when search_type=channels).

        Key behavior:
        - Transcript-only searches use ES collapse + cardinality agg (no >10k window).
        - Transcript + anything else uses OR/UNION behavior (so transcript+description >= transcript-only).
        - channel_title + description are DB-only fields (because ES mapping often won't include them reliably).
        - Other episode fields (episode_title, translations.*) use EpisodeDocument ES.
        """

        # ---------------- basics ----------------
        from .models import Episode  # keep local import pattern

        q = (self.request.GET.get('q', '') or '').strip()
        search_type = (self.request.GET.get('search_type', 'episodes') or 'episodes').strip()

        # Language filter (episodes + channels)
        selected_langs = self.request.GET.getlist("search_language")  # e.g. ["en"], ["pt"], ["en","pt"]
        selected_langs = [l.strip().lower() for l in selected_langs if (l or "").strip()]
        if not selected_langs:
            selected_langs = ["en"]  # default

        EN_ALIASES = {"en", "eng", "english"}
        def _lang_base(x):
            # "pt-br" -> "pt"
            return _norm_lang(x).split("-", 1)[0]

        def _single_non_english_only():
            wants_en = _lang_wants_english(selected_langs)
            non_en = _lang_non_english(selected_langs)
            if (not wants_en) and len(non_en) == 1:
                return _norm_lang(non_en[0])
            return None


        def _needs_multi_language_expand(selected_langs):
            EN_ALIASES = {"en", "eng", "english"}
            langs = [l.strip().lower() for l in (selected_langs or []) if (l or "").strip()]
            non_en = [l for l in langs if l not in EN_ALIASES]
            # expand only when user selected 2+ languages OR multiple non-English
            return len(langs) > 1 and (len(non_en) >= 1)


        def _norm_lang(x):
            return (x or "").strip().lower().replace("_", "-")

        def _is_english_code(x):
            x = _norm_lang(x)
            return x == "en" or x.startswith("en-") or x in ("eng", "english")

        def _lang_wants_english(langs):
            return any(_is_english_code(l) for l in (langs or []))

        def _lang_non_english(langs):
            # keep original codes (pt, pt-br, es, etc) but normalized
            out = []
            for l in (langs or []):
                nl = _norm_lang(l)
                if nl and not _is_english_code(nl):
                    out.append(nl)
            return out

        #HERE
        def _episode_allowed_ids_by_language():
            wants_en = _lang_wants_english(selected_langs)
            non_en = _lang_non_english(selected_langs)

            ids = set()

            if wants_en:
                ids |= set(
                    Episode.objects.filter(
                        Q(language__istartswith="en") | Q(language__isnull=True) | Q(language__exact="")
                    ).values_list("id", flat=True)
                )

            if non_en:
                q_lang = Q()
                for lang in non_en:
                    # match exact (pt) and regional variants (pt-br)
                    q_lang |= Q(language=lang) | Q(language__istartswith=f"{lang}-")

                ids |= set(
                    EpisodeTranslations.objects
                    .filter(q_lang)
                    .values_list("episode_id", flat=True)
                )

            return ids


        def _paginate_ids_fast(base_qs, order_by=('-publication_date', '-id')):
            """
            base_qs MUST be un-annotated (no _with_episode_stats).
            Returns: (paginator, page_obj, page_ids)
            """
            # total count (cheap when base_qs is simple)
            total = base_qs.count()

            # build paginator without triggering count on an annotated qs
            paginator = Paginator(range(total), page_size)
            page_obj = paginator.get_page(page)

            # get IDs for this page (cheap)
            qs_ids = (
                base_qs.order_by(*order_by)
                .values_list('id', flat=True)[start:end]
            )
            page_ids = list(qs_ids)

            return paginator, page_obj, page_ids
        #HERE
        def _fetch_page_items_with_stats(page_ids):
            """
            Fetch only the page's Episodes, apply heavy stats, then:
            - multi-language mode: expand into multiple rows (translations + optional EN)
            - single non-English only: return EpisodeTranslations rows (so templates show translated fields)
            - otherwise: return Episodes with display_* overlay (your existing behavior)
            """
            if not page_ids:
                return []

            order = Case(
                *[When(id=pk, then=pos) for pos, pk in enumerate(page_ids)],
                output_field=IntegerField(),
            )

            ep_qs = self._with_episode_stats(
                Episode.objects.filter(id__in=page_ids)
            ).select_related("channel") \
            .exclude(channel__sanitized_channel_title__isnull=True) \
            .exclude(channel__sanitized_channel_title='') \
            .order_by(order)

            # Only prefetch transcripts when transcript searching (base episodes)
            needs_transcripts = bool(selected & transcript_selectors)
            if needs_transcripts:
                ep_qs = ep_qs.prefetch_related("transcripts")

            # MULTI-LANGUAGE MODE (unchanged)
            if _needs_multi_language_expand(selected_langs):
                items = self._expand_multi_language_items(ep_qs, selected_langs)

                # Prefetch translation transcripts if transcript search
                if needs_transcripts:
                    trans_ids = [t.id for t in items if hasattr(t, "episode") and hasattr(t, "language")]
                    if trans_ids:
                        t_qs = (
                            EpisodeTranslations.objects
                            .filter(id__in=trans_ids)
                            .select_related("episode", "episode__channel")
                            .prefetch_related("transcriptstranslations")  # adjust if your related_name differs
                        )
                        t_map = {t.id: t for t in t_qs}
                        items = [t_map.get(x.id, x) if hasattr(x, "episode") and hasattr(x, "language") else x for x in items]

                return items

            # ✅ SINGLE NON-ENGLISH ONLY MODE (NEW)
            single_lang = _single_non_english_only()
            if single_lang:
                base_code = _lang_base(single_lang)  # pt-br -> pt

                # Build priority: exact match first, then base-code fallback
                prio_whens = [
                    When(language=single_lang, then=Value(0)),
                    When(language=base_code, then=Value(1)),
                ]

                t_qs = (
                    EpisodeTranslations.objects
                    .filter(episode_id__in=page_ids)
                    .filter(Q(language=single_lang) | Q(language__istartswith=f"{base_code}-") | Q(language=base_code))
                    .select_related("episode", "episode__channel")
                )

                et_fields = {f.name for f in EpisodeTranslations._meta.get_fields()}
                if "translated" in et_fields:
                    t_qs = t_qs.filter(translated=True)

                # If transcript searching, pull translated segments too
                if needs_transcripts:
                    t_qs = t_qs.prefetch_related("transcriptstranslations")  # adjust if needed

                t_qs = t_qs.annotate(
                    _prio=Case(*prio_whens, default=Value(9999), output_field=IntegerField())
                ).order_by("episode_id", "_prio", "id")

                # Take the best translation per episode_id
                tr_map = {}
                for tr in t_qs:
                    if tr.episode_id not in tr_map:
                        tr_map[tr.episode_id] = tr

                # Copy stats fields from base episode rows onto translations
                base_by_id = {e.id: e for e in ep_qs}
                stats_fields = (
                    "bookmarks_count",
                    "comments_count",
                    "ep_avg_rating",
                    "ep_rating_count",
                    "total_episode_views",
                    "total_downloads",
                    "total_shares",
                )

                items = []
                for eid in page_ids:
                    tr = tr_map.get(eid)
                    if tr:
                        base = base_by_id.get(eid)
                        if base:
                            for f in stats_fields:
                                setattr(tr, f, getattr(base, f, 0))

                        # Template compatibility (same trick you used elsewhere)
                        tr.channel = tr.episode.channel
                        tr.sanitized_episode_title = tr.episode.sanitized_episode_title

                        items.append(tr)
                    else:
                        # Fallback to base episode if somehow missing translation
                        items.append(base_by_id.get(eid))

                return [x for x in items if x is not None]

            # SINGLE-LANGUAGE (English or English+something but not expand): keep your overlay behavior
            ep_qs = self._apply_episode_display_language(ep_qs, selected_langs)
            return list(ep_qs)


        def _channel_allowed_ids_by_language():
            """
            Returns a set of Channel IDs allowed by selected_langs.

            English:
            - include all Channels
            Non-English:
            - include channels that have a translation row in ChannelTranslations for that language
                (maps back to master channel id)
            """
            wants_en = _lang_wants_english(selected_langs)
            non_en = _lang_non_english(selected_langs)

            ids = set()

            if wants_en:
                ids |= set(Channel.objects.values_list("id", flat=True))

            if non_en:
                qs = ChannelTranslations.objects.filter(language__in=non_en)
                # If your translation table has a boolean:
                if "translated" in {f.name for f in ChannelTranslations._meta.get_fields()}:
                    qs = qs.filter(translated=True)

                # Figure out how to map translation rows -> master Channel id
                field_names = {f.name for f in ChannelTranslations._meta.get_fields()}

                if "channel" in field_names:
                    # FK named `channel`
                    ids |= set(qs.values_list("channel_id", flat=True))
                elif "channel_id" in field_names:
                    # plain integer field actually named channel_id
                    ids |= set(qs.values_list("channel_id", flat=True))
                elif "master_channel" in field_names:
                    ids |= set(qs.values_list("master_channel_id", flat=True))
                elif "channel_ref" in field_names:
                    ids |= set(qs.values_list("channel_ref_id", flat=True))
                else:
                    # last resort: raise a helpful error with the actual fields
                    raise RuntimeError(
                        f"ChannelTranslations has no recognizable link field. Fields: {sorted(field_names)}"
                    )

            return ids


        # Compute once (cheap and keeps logic consistent everywhere)

        allowed_episode_ids = _episode_allowed_ids_by_language()



        # 1) No query → normal ListView pagination
        if not q:
            return super().paginate_queryset(qs, page_size)

        # ---------------- CHANNELS branch ----------------
        if search_type == 'channels':
            allowed_channel_ids = _channel_allowed_ids_by_language()
            wants = self.request.GET.getlist('search_in')
            filters = []
            if 'channel_title' in wants:
                filters.append(Q(channel_title__icontains=q))
            if 'channel_author' in wants:
                filters.append(Q(channel_author__icontains=q))
            if 'channel_summary' in wants:
                filters.append(Q(channel_summary__icontains=q))

            chans = Channel.objects.all().filter(id__in=allowed_channel_ids)
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
                .only(
                    'id', 'channel_title', 'channel_author', 'channel_summary',
                    'channel_image_url', 'sanitized_channel_title'
                )
                .order_by('channel_title', 'id')
            )

            paginator = Paginator(chans, page_size)
            page_num = int(self.request.GET.get('page', 1))
            page_obj = paginator.get_page(page_num)
            return paginator, page_obj, list(page_obj.object_list), page_obj.has_other_pages()

        # ---------------- EPISODES branch ----------------

        # Date window
        date_filter = self.request.GET.get('search_date', 'anytime')
        window = None
        if date_filter != 'anytime':
            days = int(date_filter)
            window = (timezone.timedelta(hours=24) if days == 24 else timezone.timedelta(days=days))
        
        # Build eligible episode ids for date filtering in TranscriptDocument (ES can't join)
        eligible_episode_ids = None

        base_eligible_qs = (
            Episode.objects
            .filter(id__in=allowed_episode_ids)
            .exclude(channel__sanitized_channel_title__isnull=True)
            .exclude(channel__sanitized_channel_title='')
        )

        if window:
            base_eligible_qs = base_eligible_qs.filter(publication_date__gte=timezone.now() - window)

        # Always set it when transcript searching (keeps counts consistent)
        eligible_episode_ids = list(base_eligible_qs.values_list("id", flat=True))


        # Selected UI fields
        selected = set(self.request.GET.getlist('search_in'))

        transcript_selectors = {self.SEGMENT_FIELD, self.SEGMENT_ALIAS_FIELD, self.TRANSCRIPTS_FIELD}
        transcript_only_sets = [{self.SEGMENT_FIELD}, {self.SEGMENT_ALIAS_FIELD}, {self.TRANSCRIPTS_FIELD}]

        # Fields that are DB-only because your ES mapping may not index them reliably
        DB_ONLY = {'channel_title', 'description', 'episode_description'}

        # Helper: stable dedupe
        def _dedupe_preserve_order(items):
            seen = set()
            out = []
            for x in items:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        page = int(self.request.GET.get('page', 1))
        start = (page - 1) * page_size
        end = start + page_size

        # -------- Transcript ES helper (SAFE: no >10k window) --------
        def _transcript_page_episode_ids_with_total(start_, size_):
            wants_en = _lang_wants_english(selected_langs)
            non_en = _lang_non_english(selected_langs)

            def _base_search(doc_cls):
                broad_q = ES_Q("match", segment_text={"query": q, "operator": "or"})
                phrase_q = ES_Q("match_phrase", segment_text={"query": q})

                s = (
                    doc_cls.search()
                    .query(
                        "function_score",
                        query=broad_q,
                        functions=[{"filter": phrase_q, "weight": 10}],
                        boost_mode="sum",
                        score_mode="sum",
                    )
                    .params(collapse={"field": "episode_id", "inner_hits": {"name": "top_segment", "size": 1}})
                    .sort({"_score": "desc"})
                )

                # Date window (episode_id terms filter)
                if eligible_episode_ids is not None:
                    if not eligible_episode_ids:
                        return None
                    s = s.filter("terms", episode_id=eligible_episode_ids)

                return s

            def _cardinality_total(s):
                # count unique episode_id without being limited by size/from
                s0 = s.extra(size=0)
                s0.aggs.bucket("uniq_eps", "cardinality", field="episode_id")
                resp = s0.execute()
                try:
                    return int(resp.aggregations.uniq_eps.value or 0)
                except Exception:
                    return 0

            def _page_ids_scores(s, from_, size_):
                resp = s.extra(from_=from_, size=size_).execute()
                hits = resp.hits.hits
                return [(int(h["_source"]["episode_id"]), float(h["_score"])) for h in hits]

            # ------------------------------------------------------------
            # 1) FAST + EXACT for the common cases (single “source”)
            # ------------------------------------------------------------

            # English only
            if wants_en and not non_en:
                s = _base_search(TranscriptDocument)
                if s is None:
                    return 0, []
                total_unique = _cardinality_total(s)
                page_ids_scores = _page_ids_scores(s, start_, size_)
                return total_unique, page_ids_scores

            # Single non-English language only
            if (not wants_en) and len(non_en) == 1:
                lang = non_en[0]
                s = _base_search(TranscriptTranslationsDocument)
                if s is None:
                    return 0, []
                s = s.filter("term", language=lang)
                total_unique = _cardinality_total(s)
                page_ids_scores = _page_ids_scores(s, start_, size_)
                return total_unique, page_ids_scores

            # ------------------------------------------------------------
            # 2) Multi-language union (your current “merge by best score”)
            #    We must fetch enough hits to fill the requested page.
            # ------------------------------------------------------------

            need = start_ + size_
            fetch_n = max(500, need)  # <<< key change: don’t start at 100

            collected = []

            if wants_en:
                s_en = _base_search(TranscriptDocument)
                if s_en is not None:
                    collected += _page_ids_scores(s_en, 0, fetch_n)

            if non_en:
                s_tr = _base_search(TranscriptTranslationsDocument)
                if s_tr is not None:
                    s_tr = s_tr.filter("terms", language=non_en)
                    collected += _page_ids_scores(s_tr, 0, fetch_n)

            # Merge + dedupe by episode_id keeping best score
            best = {}
            for eid, sc in collected:
                if eid not in best or sc > best[eid]:
                    best[eid] = sc

            sorted_items = sorted(best.items(), key=lambda x: x[1], reverse=True)
            page_items = sorted_items[start_:start_ + size_]

            # NOTE: this is an approximation for multi-language union unless you do the heavier exact-union count
            total_unique = len(best)

            return total_unique, [(eid, score) for eid, score in page_items]




        #HERE
        def _transcript_ids_for_union_fast(cap):
            """
            Cheap way to get episode_ids for UNION:
            - no inner_hits
            - only fetch episode_id
            - collapse by episode_id
            - cap results to avoid timeouts
            """
            wants_en = _lang_wants_english(selected_langs)
            non_en   = _lang_non_english(selected_langs)

            def _run(doc_cls, lang=None):
                s = (
                    doc_cls.search()
                    .query("match", segment_text={"query": q, "operator": "or"})
                    .params(collapse={"field": "episode_id"})
                    .source(["episode_id"])
                    .sort({"_score": "desc"})
                    .extra(size=cap)
                    .params(request_timeout=30)
                )

                if eligible_episode_ids is not None:
                    if not eligible_episode_ids:
                        return set()
                    s = s.filter("terms", episode_id=eligible_episode_ids)

                if lang is not None:
                    s = s.filter("term", language=lang)

                resp = s.execute()
                return {int(h.episode_id) for h in resp}

            out = set()
            if wants_en:
                out |= _run(TranscriptDocument)

            if non_en:
                # important: translations doc supports terms
                s = (
                    TranscriptTranslationsDocument.search()
                    .query("match", segment_text={"query": q, "operator": "or"})
                    .filter("terms", language=non_en)
                    .params(collapse={"field": "episode_id"})
                    .source(["episode_id"])
                    .sort({"_score": "desc"})
                    .extra(size=cap)
                    .params(request_timeout=30)
                )
                if eligible_episode_ids is not None:
                    if not eligible_episode_ids:
                        return out
                    s = s.filter("terms", episode_id=eligible_episode_ids)

                resp = s.execute()
                out |= {int(h.episode_id) for h in resp}

            return out



        # -------- DB helper (channel_title/description OR) --------
        def _db_ids_for_db_only_fields(sel_set):
            """
            Returns a set of Episode IDs that match any selected DB-only fields (OR).
            """
            db_filters = Q()
            if 'channel_title' in sel_set:
                # master channel title match (English/original)
                db_filters |= Q(channel__channel_title__icontains=q)

                # translated channel title match (Portuguese/etc) via slug mapping
                non_en = _lang_non_english(selected_langs)
                if non_en:
                    ct_fields = {f.name for f in ChannelTranslations._meta.get_fields()}
                    ct_qs = ChannelTranslations.objects.filter(
                        language__in=non_en,
                        channel_title__icontains=q,
                    )
                    if "translated" in ct_fields:
                        ct_qs = ct_qs.filter(translated=True)

                    translated_slugs = ct_qs.values_list("sanitized_channel_title", flat=True)

                    # map translated channel slugs -> master channels -> episodes
                    db_filters |= Q(channel__sanitized_channel_title__in=translated_slugs)


            if 'description' in sel_set or 'episode_description' in sel_set:
                db_filters |= Q(description__icontains=q)

            if not db_filters:
                return set()

            db_qs = Episode.objects.filter(db_filters)

            # ✅ apply language filter
            db_qs = db_qs.filter(id__in=allowed_episode_ids)

            if window:
                db_qs = db_qs.filter(publication_date__gte=timezone.now() - window)

            return set(
                db_qs.exclude(channel__sanitized_channel_title__isnull=True)
                    .exclude(channel__sanitized_channel_title='')
                    .values_list('id', flat=True)
            )


        # -------- EpisodeDocument ES helper (titles/translations/etc) --------
        def _episode_es_ids_for_selected_fields(sel_set):
            """
            Returns a set of master Episode IDs that match selected ES fields, respecting language.
            - English => search EpisodeDocument (masters)
            - Non-English => search EpisodeTranslationsDocument and map to master episode_id
            - English + Non-English => union
            """
            ES_FIELD_MAP_MASTER = {
                "episode_title": ["episode_title"],
            }
            ES_FIELD_MAP_TRANSL = {
                "episode_title": ["episode_title"],
            }

            fields_master = []
            fields_transl = []
            for key in sel_set:
                fields_master += ES_FIELD_MAP_MASTER.get(key, [])
                fields_transl += ES_FIELD_MAP_TRANSL.get(key, [])

            fields_master = _dedupe_preserve_order(fields_master)
            fields_transl = _dedupe_preserve_order(fields_transl)

            wants_en = _lang_wants_english(selected_langs)
            non_en = _lang_non_english(selected_langs)

            out_ids = set()

            # ---- Masters (English/untranslated) ----
            if wants_en and fields_master:
                es = EpisodeDocument.search()
                if window:
                    es = es.filter("range", publication_date={"gte": timezone.now() - window})

                broad_q = ES_Q("multi_match", query=q, fields=fields_master, type="best_fields", operator="or")
                phrase_q = ES_Q("multi_match", query=q, fields=fields_master, type="phrase")

                es = es.query(
                    "function_score",
                    query=broad_q,
                    functions=[{"filter": phrase_q, "weight": 10}],
                    boost_mode="sum",
                    score_mode="sum",
                ).extra(size=10000)

                resp = es.execute()
                out_ids |= {int(hit.meta.id) for hit in resp}

            # ---- Translations (Portuguese, etc) ----
            if non_en and fields_transl:
                tes = EpisodeTranslationsDocument.search()
                tes = tes.filter("terms", language=non_en)

                if window:
                    tes = tes.filter("range", publication_date={"gte": timezone.now() - window})

                broad_q = ES_Q("multi_match", query=q, fields=fields_transl, type="best_fields", operator="or")
                phrase_q = ES_Q("multi_match", query=q, fields=fields_transl, type="phrase")

                tes = tes.query(
                    "function_score",
                    query=broad_q,
                    functions=[{"filter": phrase_q, "weight": 10}],
                    boost_mode="sum",
                    score_mode="sum",
                ).extra(size=10000)

                tresp = tes.execute()
                # EpisodeTranslationsDocument has episode_id field that is the master episode id
                out_ids |= {int(hit.episode_id) for hit in tresp}

            return out_ids & allowed_episode_ids



        # ============================================================
        # 3) Transcript-only searches
        # ============================================================
        if selected in transcript_only_sets:
            total_unique, page_ids_scores = _transcript_page_episode_ids_with_total(start, page_size)
            page_ids = [eid for eid, _ in page_ids_scores]

            episode_qs = self._with_episode_stats(
                Episode.objects.filter(id__in=page_ids).filter(id__in=allowed_episode_ids)
            ).select_related('channel') \
            .prefetch_related('transcripts') \
            .exclude(channel__sanitized_channel_title__isnull=True) \
            .exclude(channel__sanitized_channel_title='')

            id_map = {e.id: e for e in episode_qs}
            page_list = [id_map[eid] for eid in page_ids if eid in id_map]

            # ✅ If single foreign language, show EpisodeTranslations rows (not base Episode rows)
            single_lang = _single_non_english_only()
            if single_lang:
                base_code = _lang_base(single_lang)

                t_qs = (
                    EpisodeTranslations.objects
                    .filter(episode_id__in=page_ids)
                    .filter(Q(language=single_lang) | Q(language__istartswith=f"{base_code}-") | Q(language=base_code))
                    .select_related("episode", "episode__channel")
                    .prefetch_related("transcriptstranslations")  # adjust if needed
                )
                et_fields = {f.name for f in EpisodeTranslations._meta.get_fields()}
                if "translated" in et_fields:
                    t_qs = t_qs.filter(translated=True)

                # Choose best translation per episode (exact lang first)
                t_qs = t_qs.annotate(
                    _prio=Case(
                        When(language=single_lang, then=Value(0)),
                        When(language=base_code, then=Value(1)),
                        default=Value(9999),
                        output_field=IntegerField(),
                    )
                ).order_by("episode_id", "_prio", "id")

                tr_map = {}
                for tr in t_qs:
                    if tr.episode_id not in tr_map:
                        tr_map[tr.episode_id] = tr

                stats_fields = (
                    "bookmarks_count",
                    "comments_count",
                    "ep_avg_rating",
                    "ep_rating_count",
                    "total_episode_views",
                    "total_downloads",
                    "total_shares",
                )

                page_list_tr = []
                for eid in page_ids:
                    tr = tr_map.get(eid)
                    base = id_map.get(eid)
                    if tr:
                        if base:
                            for f in stats_fields:
                                setattr(tr, f, getattr(base, f, 0))
                        tr.channel = tr.episode.channel
                        tr.sanitized_episode_title = tr.episode.sanitized_episode_title
                        page_list_tr.append(tr)
                    elif base:
                        page_list_tr.append(base)

                page_list = page_list_tr


            paginator = Paginator(range(total_unique), page_size)
            try:
                page_obj = paginator.page(page)
            except EmptyPage:
                page_obj = paginator.page(1)
            self._decorate_items_for_episode_links(page_list, selected_langs)
            return paginator, page_obj, page_list, page_obj.has_other_pages()

        # ============================================================
        # 4) Transcript + anything else (OR/UNION behavior)
        # Ensures transcript+description >= transcript-only.
        # ============================================================
        if (selected & transcript_selectors) and (selected - transcript_selectors):
            db_ids = _db_ids_for_db_only_fields(selected & DB_ONLY)
            selected_es = (selected - transcript_selectors - DB_ONLY)
            es_ids = _episode_es_ids_for_selected_fields(selected_es)

            TRANSCRIPT_UNION_CAP = 10000  # start smaller; tune later (500-1500)
            try:
                transcript_ids = _transcript_ids_for_union_fast(TRANSCRIPT_UNION_CAP)
            except (ConnectionTimeout, ReadTimeoutError, Exception):
                transcript_ids = set()   # degrade gracefully instead of 500 error

            all_ids = list((db_ids | es_ids | transcript_ids) & allowed_episode_ids)

            base_qs = (
                Episode.objects.filter(id__in=all_ids)
                .exclude(channel__sanitized_channel_title__isnull=True)
                .exclude(channel__sanitized_channel_title='')
                .order_by("-publication_date", "-id")
            )
            if window:
                base_qs = base_qs.filter(publication_date__gte=timezone.now() - window)

            # ✅ store expanded total for the UI (see helper below)
            self._set_total_display_from_base_qs(base_qs, selected_langs)

            paginator, page_obj, page_ids = _paginate_ids_fast(base_qs, order_by=("-publication_date", "-id"))
            page_list = _fetch_page_items_with_stats(page_ids)
            self._decorate_items_for_episode_links(page_list, selected_langs)
            return paginator, page_obj, page_list, page_obj.has_other_pages()



        # ============================================================
        # 5) DB-only searches (channel_title/description only)
        # ============================================================
        if selected and selected.issubset(DB_ONLY):
            ep_ids = _db_ids_for_db_only_fields(selected)
            ep_ids = set(ep_ids) & allowed_episode_ids

            all_ids = list(ep_ids)  # ✅ define all_ids here

            base_qs = (
                Episode.objects.filter(id__in=all_ids)
                .exclude(channel__sanitized_channel_title__isnull=True)
                .exclude(channel__sanitized_channel_title='')
            )

            if window:
                base_qs = base_qs.filter(publication_date__gte=timezone.now() - window)

            self._set_total_display_from_base_qs(base_qs, selected_langs)  # ✅ ADD

            paginator, page_obj, page_ids = _paginate_ids_fast(base_qs)
            page_list = _fetch_page_items_with_stats(page_ids)
            self._decorate_items_for_episode_links(page_list, selected_langs)
            return paginator, page_obj, page_list, page_obj.has_other_pages()





        # ============================================================
        # 6) DB-only + other ES fields (UNION OR behavior)
        # e.g. description + episode_title, channel_title + episode_title
        # ============================================================
        if selected & DB_ONLY:
            db_ids = _db_ids_for_db_only_fields(selected & DB_ONLY)
            es_ids = _episode_es_ids_for_selected_fields(selected - DB_ONLY)

            all_ids = list(set(db_ids | es_ids) & allowed_episode_ids)

            base_qs = Episode.objects.filter(id__in=all_ids)
            base_qs = base_qs.exclude(channel__sanitized_channel_title__isnull=True).exclude(channel__sanitized_channel_title='')

            if window:
                base_qs = base_qs.filter(publication_date__gte=timezone.now() - window)

            self._set_total_display_from_base_qs(base_qs, selected_langs)  # ✅ ADD

            paginator, page_obj, page_ids = _paginate_ids_fast(base_qs)
            page_list = _fetch_page_items_with_stats(page_ids)
            self._decorate_items_for_episode_links(page_list, selected_langs)
            return paginator, page_obj, page_list, page_obj.has_other_pages()


        # ============================================================
        # 7) ES-only searches (episode_title/translations/etc)
        # ============================================================
        # ES-only searches
        es_ids = _episode_es_ids_for_selected_fields(selected)
        es_ids = set(es_ids) & allowed_episode_ids
        all_ids = list(es_ids)

        if not all_ids:
            es_ids = _episode_es_ids_for_selected_fields({'episode_title'})
            es_ids = set(es_ids) & allowed_episode_ids
            all_ids = list(es_ids)

        base_qs = Episode.objects.filter(id__in=all_ids) \
            .exclude(channel__sanitized_channel_title__isnull=True) \
            .exclude(channel__sanitized_channel_title='')

        if window:
            base_qs = base_qs.filter(publication_date__gte=timezone.now() - window)

        self._set_total_display_from_base_qs(base_qs, selected_langs)  # ✅ ADD

        paginator, page_obj, page_ids = _paginate_ids_fast(base_qs)
        page_list = _fetch_page_items_with_stats(page_ids)
        self._decorate_items_for_episode_links(page_list, selected_langs)
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

        # ✅ EPISODES counts (non-channels searches)
        if search_type != 'channels':
            paginator = ctx.get('paginator')
            page_obj  = ctx.get('page_obj')

            if hasattr(self, "_total_display_items"):
                ctx["total_episodes"] = self._total_display_items
            elif paginator is not None:
                ctx["total_episodes"] = paginator.count
            else:
                ctx["total_episodes"] = len(ctx.get("episodes", []))


            if page_obj is not None:
                try:
                    ctx['displayed_episodes'] = page_obj.end_index()
                except Exception:
                    ctx['displayed_episodes'] = len(ctx.get('episodes', []))
            else:
                ctx['displayed_episodes'] = len(ctx.get('episodes', []))

        # ✅ CHANNELS counts
        if search_type == 'channels':
            paginator = ctx.get('paginator')
            page_obj  = ctx.get('page_obj')

            # total matches for the query
            if paginator is not None:
                ctx['total_channels'] = paginator.count
            else:
                # fallback if paginator missing for any reason
                chans = ctx.get('channels', [])
                ctx['total_channels'] = len(chans)

            # how many are currently displayed (page 1 shows 10, after scroll more, JS updates it)
            if page_obj is not None:
                try:
                    ctx['displayed_channels'] = page_obj.end_index()
                except Exception:
                    ctx['displayed_channels'] = len(ctx.get('channels', []))
            else:
                ctx['displayed_channels'] = len(ctx.get('channels', []))

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
    
    #HERE
    def get(self, request, *args, **kwargs):
        # record non-AJAX searches (keep your existing block unchanged)
        if request.GET.get('ajax') != '1':
            query  = request.GET.get('q','')
            in_str = ",".join(request.GET.getlist('search_in'))
            date_f = request.GET.get('search_date','anytime')
            ip     = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR'))
            user   = request.user if request.user.is_authenticated else None
            try:
                sq, created = SearchQuery.objects.get_or_create(
                    user=user, query=query,
                    defaults={'search_in': in_str, 'search_date': date_f, 'ip_address': ip}
                )
                if not created:
                    sq.count += 1
                    sq.last_searched = timezone.now()
                    sq.ip_address    = ip
                    sq.save()
            except:
                pass

        # AJAX: handle fully here (NO double-run, NO super().get())
        if request.GET.get('ajax') == '1':
            paginator, page_obj, object_list, _ = self.paginate_queryset(
                self.get_queryset(), self.paginate_by
            )
            if int(request.GET.get('page', 1)) > paginator.num_pages:
                return HttpResponse('', status=200)

            context = self.get_context_data(object_list=object_list)
            return self.render_to_response(context)

        # Non-AJAX: normal flow (paginate_queryset runs once)
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

        # ✅ per-episode views (subquery, avoids fan-out inflation)
        views_sq = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
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
                # ✅ Fixed total episode views:
                total_episode_views=Coalesce(
                    Subquery(views_sq, output_field=IntegerField()),
                    Value(0)
                ),
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

        # ✅ per-episode views (subquery, avoids fan-out inflation)
        views_sq = (
            EpisodeVisit.objects
            .filter(episode=OuterRef('pk'))
            .values('episode')
            .annotate(total=Sum('count'))
            .values('total')[:1]
        )
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
                # ✅ Fixed total episode views:
                total_episode_views=Coalesce(
                    Subquery(views_sq, output_field=IntegerField()),
                    Value(0)
                ),
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

def is_admin_user(user):
    return user.is_authenticated and (user.is_staff or user.is_superuser)

@login_required
@user_passes_test(is_admin_user)
def ticket_notifications(request):
    tickets = SupportTicket.objects.filter(
        status__in=["pending", "in_progress"]
    ).order_by("-submission_date")

    return render(request, "podcasts/ticket_notifications.html", {
        "tickets": tickets,
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
