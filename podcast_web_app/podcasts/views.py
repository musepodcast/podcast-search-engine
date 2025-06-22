# podcast_web_app/podcasts/views.py

from django.views.generic import ListView, DetailView, TemplateView
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q, Prefetch, Avg, Count, Sum, F, OuterRef, Subquery, Value, IntegerField, FloatField
from django.db.models.functions import Coalesce
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.timesince import timesince
import logging
from collections import Counter
import re, difflib
import sys
import json
import itertools
from django.http import Http404, HttpResponse, HttpResponseForbidden, JsonResponse
from .models import (
    Channel, ChannelTranslations, ChannelVisit,
    Episode, EpisodeTranslations, EpisodeVisit,
    Transcript, TranscriptTranslations,
    Chapter, ChapterTranslations,
    SearchQuery, ChannelInteraction, EpisodeInteraction,
    Comment, CommentReaction, Reply

)
from .filters import EpisodeFilter

from .forms import CustomAuthenticationForm

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
from django.views.decorators.http import require_POST, require_GET
from podcasts.search.documents import EpisodeDocument, TranscriptDocument
from elasticsearch_dsl import Q as DSLQ
from django.contrib.postgres.search import TrigramSimilarity


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
    success_url = reverse_lazy('podcasts:login')  # Redirect after successful signup.
    template_name = 'podcasts/signup.html'

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


class ChannelListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/channel_list.html'
    context_object_name = 'channels'
    paginate_by = 10      
    
    def get_queryset(self):
        # Annotate the Channel table with all of your counts.
        return (
            Channel.objects
                   .annotate(
                       favorites_count=Count(
                           'channel_interactions',
                           filter=Q(channel_interactions__followed=True),
                           distinct=True
                       ),
                       notifications_count=Count(
                           'channel_interactions',
                           filter=Q(channel_interactions__notifications_enabled=True),
                           distinct=True
                       ),
                       avg_rating=Avg('channel_interactions__rating'),
                       rating_count=Count('channel_interactions__rating'),
                       total_views=Coalesce(Sum('channelvisit__count'), 0),
                   )
                   .order_by('-total_views', 'channel_title')
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        lang = get_selected_language(self.request)

        if lang not in ('en', 'en-us'):
            # Grab all the translated rows in one query
            translations = ChannelTranslations.objects.filter(
                language=lang,
                translated=True,
                sanitized_channel_title__in=[
                    ch.sanitized_channel_title for ch in context['channels']
                ]
            )
            trans_map = {
                tr.sanitized_channel_title: tr
                for tr in translations
            }

            # Overwrite only the fields you translate
            for ch in context['channels']:
                tr = trans_map.get(ch.sanitized_channel_title)
                if not tr:
                    continue
                ch.channel_title   = tr.channel_title
                ch.channel_summary = tr.channel_summary
                # if you also translate author:
                # ch.channel_author  = tr.channel_author

        context['selected_language'] = lang
        return context

    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax') == '1':
            return render(self.request,
                          'podcasts/channel_list_items.html',
                          context)
        return super().render_to_response(context, **response_kwargs)

class ChannelDetailView(LoginRequiredMixin, DetailView):
    login_url           = reverse_lazy('podcasts:home')
    template_name       = 'podcasts/channel_detail.html'
    context_object_name = 'channel'

    def get_object(self):
        slug = self.kwargs['sanitized_channel_title']
        lang = get_selected_language(self.request)
        if lang in ('en', 'en-us'):
            return get_object_or_404(Channel, sanitized_channel_title=slug)
        return get_object_or_404(
            ChannelTranslations,
            sanitized_channel_title=slug,
            language=lang,
            translated=True
        )

    def dispatch(self, request, *args, **kwargs):
        disp = self.get_object()
        if isinstance(disp, Channel):
            base = disp
        else:
            # find the real Channel and overlay its title/summary
            base = get_object_or_404(
                Channel, sanitized_channel_title=disp.sanitized_channel_title
            )
            base.channel_title   = disp.channel_title
            base.channel_summary = disp.channel_summary
            base.channel_author  = getattr(disp, 'channel_author', base.channel_author)

        self.base_channel    = base
        self.display_channel = disp

        # record a view
        if request.user.is_authenticated and request.GET.get('ajax') != '1':
            ip = request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0] or request.META.get('REMOTE_ADDR')
            visit, _ = ChannelVisit.objects.get_or_create(user=request.user, channel=base)
            visit.count          += 1
            visit.last_visited    = timezone.now()
            visit.last_ip_address = ip
            visit.save()

        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        ctx  = super().get_context_data(**kwargs)
        base = self.base_channel
        lang = get_selected_language(self.request)

        # expose the translated-or-not display object
        ctx['channel'] = self.display_channel if not isinstance(self.display_channel, Channel) else base

        # toggles & your five aggregates
        interaction, _ = ChannelInteraction.objects.get_or_create(
            user=self.request.user, channel=base
        )
        ctx.update({
            'has_followed_channel':  interaction.followed,
            'receive_notifications': interaction.notifications_enabled,
            'channel_rating':        interaction.rating,
            'star_range':            range(1, 6),
            'favorites_count':     ChannelInteraction.objects.filter(
                                     channel=base, followed=True
                                   ).count(),
            'notifications_count': ChannelInteraction.objects.filter(
                                     channel=base, notifications_enabled=True
                                   ).count(),
        })
        rating_stats = ChannelInteraction.objects.filter(
            channel=base, rating__isnull=False
        ).aggregate(
            avg_rating=Avg('rating'),
            rating_count=Count('rating')
        )
        ctx['avg_rating']   = rating_stats['avg_rating']   or 0.0
        ctx['rating_count'] = rating_stats['rating_count'] or 0
        ctx['total_views']  = ChannelVisit.objects.filter(
                                channel=base
                             ).aggregate(total=Sum('count'))['total'] or 0

        # ---- now annotate episodes exactly like ChannelListView did for channels ----

        # build our per-episode subqueries
        lookup = OuterRef('episode') if lang not in ('en','en-us') else OuterRef('pk')
        bookmarks_sq = EpisodeInteraction.objects.filter(
                          episode=lookup, bookmarked=True
                      ).order_by().values('episode') \
                       .annotate(c=Count('*')).values('c')
        avg_rating_sq = EpisodeInteraction.objects.filter(
                          episode=lookup
                       ).order_by().values('episode') \
                       .annotate(a=Avg('rating')).values('a')
        rating_count_sq = EpisodeInteraction.objects.filter(
                             episode=lookup, rating__isnull=False
                         ).order_by().values('episode') \
                          .annotate(c=Count('rating')).values('c')
        views_sq = EpisodeVisit.objects.filter(
                       episode=lookup
                   ).order_by().values('episode') \
                    .annotate(s=Sum('count')).values('s')

        page_num = int(self.request.GET.get('page', 1))

        if lang in ('en', 'en-us'):
            eps_qs = base.episodes.all().order_by('-publication_date').annotate(
                bookmarks_count     = Coalesce(Subquery(bookmarks_sq,   output_field=IntegerField()), Value(0)),
                ep_avg_rating       = Coalesce(Subquery(avg_rating_sq, output_field=FloatField()),   Value(0.0)),
                ep_rating_count     = Coalesce(Subquery(rating_count_sq, output_field=IntegerField()), Value(0)),
                total_episode_views = Coalesce(Subquery(views_sq,         output_field=IntegerField()), Value(0)),
            )
            paginator = Paginator(eps_qs, 10)
            page_num = int(self.request.GET.get('page', 1))
            try:
                page_obj = paginator.page(page_num)
            except EmptyPage:
                # out-of-range → no episodes
                page_obj = []
            ctx['episodes'] = page_obj
        else:
            tr_qs = EpisodeTranslations.objects.filter(
                episode__channel=base,
                language=lang,
                translated=True
            ).order_by('-publication_date').annotate(
                bookmarks_count     = Coalesce(Subquery(bookmarks_sq,   output_field=IntegerField()), Value(0)),
                ep_avg_rating       = Coalesce(Subquery(avg_rating_sq, output_field=FloatField()),   Value(0.0)),
                ep_rating_count     = Coalesce(Subquery(rating_count_sq, output_field=IntegerField()), Value(0)),
                total_episode_views = Coalesce(Subquery(views_sq,         output_field=IntegerField()), Value(0)),
            )
            # patch the .channel onto each translation
            for tr in tr_qs:
                tr.channel = base
            paginator = Paginator(tr_qs, 10)
            page_num = int(self.request.GET.get('page', 1))
            try:
                page_obj = paginator.page(page_num)
            except EmptyPage:
                # out-of-range → no episodes
                page_obj = []
            ctx['episodes'] = page_obj    
        return ctx

    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax') == '1':
            # if no episodes (empty list), just return empty response
            if not context['episodes']:
                return HttpResponse('')
            return render(
                self.request,
                'podcasts/channel_detail_item.html',
                context
            )
        return super().render_to_response(context, **response_kwargs)


class EpisodeDetailView(LoginRequiredMixin, DetailView):
    login_url           = reverse_lazy('podcasts:home')
    template_name       = 'podcasts/episode_detail.html'
    context_object_name = 'episode'

    def dispatch(self, request, *args, **kwargs):
        disp = self.get_object()
        if isinstance(disp, Episode):
            base = disp
        else:
            base = disp.episode
            # so your template’s `episode.channel` still works
            disp.channel = base.channel

        self.base_episode    = base
        self.display_episode = disp

        # record a view
        if request.user.is_authenticated and request.GET.get('ajax') != '1':
            xff = request.META.get('HTTP_X_FORWARDED_FOR')
            ip  = xff.split(',')[0].strip() if xff else request.META.get('REMOTE_ADDR')
            visit, _ = EpisodeVisit.objects.get_or_create(user=request.user, episode=base)
            # avoid race by using F()
            visit.count           = F('count') + 1
            visit.last_visited    = timezone.now()
            visit.last_ip_address = ip
            visit.save()

        return super().dispatch(request, *args, **kwargs)

    def get_object(self):
        slug_ch = self.kwargs['sanitized_channel_title']
        slug_ep = self.kwargs['sanitized_episode_title']
        lang    = get_selected_language(self.request)

        if lang in ('en','en-us'):
            return get_object_or_404(
                Episode,
                channel__sanitized_channel_title=slug_ch,
                sanitized_episode_title=slug_ep
            )

        # try to find the exact translation
        tr = EpisodeTranslations.objects.filter(
            episode__channel__sanitized_channel_title=slug_ch,
            sanitized_episode_title=slug_ep,
            language=lang,
            translated=True
        ).first()
        if tr:
            return tr

        # otherwise strip any "_Lang" suffix and retry
        base_slug = slug_ep.rsplit('_',1)[0]
        return get_object_or_404(
            EpisodeTranslations,
            episode__channel__sanitized_channel_title=slug_ch,
            episode__sanitized_episode_title=base_slug,
            language=lang,
            translated=True
        )

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
            if tr_qs.exists():
                segments = tr_qs
            else:
                segments = Transcript.objects.filter(
                    episode=base
                ).order_by('segment_time')
        else:
            segments = Transcript.objects.filter(
                episode=base
            ).order_by('segment_time')

        # -- CHAPTERS: fetch both translations & originals --
        if isinstance(disp, EpisodeTranslations):
            ch_qs = ChapterTranslations.objects.filter(
                episodetranslations=disp,
                language=lang
            )
            if not ch_qs.exists():
                ch_qs = Chapter.objects.filter(episode=base)
        else:
            ch_qs = Chapter.objects.filter(episode=base)

        # Now convert to a list and sort by parsing the “HH:MM:SS” (or “MM:SS”) string:
        def _to_seconds(ts):
            parts = [int(p) for p in ts.split(':')]
            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h, m, s = 0, parts[0], parts[1]
            else:
                h, m, s = 0, 0, parts[0]
            return h*3600 + m*60 + s

        chapters = list(ch_qs)             # evaluate the QuerySet
        chapters.sort(key=lambda c: _to_seconds(c.chapter_time))

        # 3) USER INTERACTION + AGGREGATES
        interaction, _ = EpisodeInteraction.objects.get_or_create(
            user=self.request.user,
            episode=base
        )
        ctx['is_bookmarked']      = interaction.bookmarked
        ctx['bookmarks_count']    = EpisodeInteraction.objects.filter(
            episode=base, bookmarked=True
        ).count()
        ctx['comments_count'] = base.comments.count()

        stats = EpisodeInteraction.objects.filter(
            episode=base,
            rating__isnull=False
        ).aggregate(avg=Avg('rating'), cnt=Count('rating'))
        ctx['ep_avg_rating']      = stats['avg'] or 0
        ctx['ep_rating_count']    = stats['cnt'] or 0

        ctx['total_episode_views'] = EpisodeVisit.objects.filter(
            episode=base
        ).aggregate(total=Sum('count'))['total'] or 0
        
        ctx['episode_rating'] = interaction.rating or 0
        ctx['star_range']     = range(1,6)

        # 4) MERGE & CONTEXT
        ctx['merged_segments'] = self.merge_consecutive_speakers(segments)
        ctx['chapters']        = chapters
        ctx['post_episode_id'] = self.base_episode.id
        ctx['selected_language'] = lang
        # `episode` in the template is the display object (translated or original)
        ctx['episode']          = disp
        return ctx

    def merge_consecutive_speakers(self, segments):
        merged = []
        current = None
        for seg in segments:
            if current is None:
                current = {
                    'combined_time': seg.segment_time,
                    'speaker': seg.speaker,
                    'combined_text': seg.segment_text
                }
            elif seg.speaker == current['speaker']:
                try:
                    start, _   = current['combined_time'].split(' - ',1)
                    _, new_end = seg.segment_time.split(' - ',1)
                    current['combined_time'] = f"{start} - {new_end}"
                except ValueError:
                    pass
                current['combined_text'] += ' ' + seg.segment_text
            else:
                merged.append(current)
                current = {
                    'combined_time': seg.segment_time,
                    'speaker': seg.speaker,
                    'combined_text': seg.segment_text
                }
        if current:
            merged.append(current)
        return merged






class EpisodeListView(LoginRequiredMixin, ListView):
    login_url = reverse_lazy('podcasts:home')
    template_name = 'podcasts/episode_list.html'
    context_object_name = 'episodes'
    paginate_by = 10

    def get_queryset(self):
        lang = get_selected_language(self.request)
        if lang in ('en', 'en-us'):
            qs = Episode.objects.select_related('channel') \
                    .prefetch_related('transcripts', 'chapters') \
                    .order_by('-publication_date')
            # Annotate with aggregated fields:
            qs = qs.annotate(
                bookmarks_count=Count('episode_interactions', filter=Q(episode_interactions__bookmarked=True)),
                ep_avg_rating=Avg('episode_interactions__rating'),
                ep_rating_count=Count('episode_interactions__rating'),
                total_episode_views=Sum('episodevisit__count')
            )
            return qs
        else:
            # For translated episodes, annotations may not be available (unless you define them there)
            qs = EpisodeTranslations.objects.filter(language=lang, translated=True) \
                    .select_related('episode__channel') \
                    .prefetch_related('transcriptstranslations', 'chapterstranslations') \
                    .order_by('-publication_date')
            return qs

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
        context['selected_language'] = lang
        return context
    
    def render_to_response(self, context, **response_kwargs):
        if self.request.GET.get('ajax') == '1':
            return render(self.request,
                          'podcasts/episode_list_items.html',
                          context)
        return super().render_to_response(context, **response_kwargs)

class SearchResultsView(LoginRequiredMixin, ListView):
    login_url = 'podcasts:home'
   
       #––– Make these match your <input value="…"> in the form –––
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

    def paginate_queryset(self, qs, page_size):
        """
        1. If no `q`, fall back to default ListView pagination.
        2. If `search_type=='channels'`, do a pure-ORM filter on Channel.
        3. If `search_type=='episodes'` and any episode-fields are checked,
           mirror the channel logic with Q-filters on Episode (including channel title).
        4. Otherwise (full-text search), branch:
           a) Transcript-only ES query if only transcript fields checked.
           b) Multi-match ES query across EpisodeDocument.
        """
 
        q           = (self.request.GET.get('q', '') or '').strip()
        search_type = self.request.GET.get('search_type', 'episodes')

        # 1) No query → default ListView pagination
        if not q:
            return super().paginate_queryset(qs, page_size)

        # 2) CHANNELS branch (unchanged)
        if search_type == 'channels':
            wants = self.request.GET.getlist('search_in')
            filters = []
            if 'channel_title'  in wants:
                filters.append(Q(channel_title__icontains=q))
            if 'channel_author' in wants:
                filters.append(Q(channel_author__icontains=q))
            if 'channel_summary'in wants:
                filters.append(Q(channel_summary__icontains=q))

            chans = Channel.objects.all()
            if filters:
                combined = filters.pop()
                for f in filters:
                    combined |= f
                chans = chans.filter(combined)
            else:
                chans = chans.filter(channel_title__icontains=q)

            paginator = Paginator(chans, page_size)
            page_num  = int(self.request.GET.get('page', 1))
            page_obj  = paginator.get_page(page_num)
            return paginator, page_obj, list(page_obj.object_list), page_obj.has_other_pages()

        # 3) EPISODES pure-ORM branch
        if search_type == 'episodes':
            wants = self.request.GET.getlist('search_in')
            filters = []

            # mirror your channel logic on episode fields:
            if 'episode_title'     in wants:
                filters.append(Q(episode_title__icontains=q))
            if 'description'       in wants:
                filters.append(Q(description__icontains=q))
            # now include channel title too:
            if 'channel_title'     in wants:
                filters.append(Q(channel__channel_title__icontains=q))
            # transcripts checkbox:
            if 'transcripts'       in wants or 'segment_text' in wants:
                filters.append(Q(transcripts__segment_text__icontains=q))

            if filters:
                combined = filters.pop()
                for f in filters:
                    combined |= f
                episodes_qs = Episode.objects.filter(combined).distinct()
            else:
                episodes_qs = Episode.objects.filter(episode_title__icontains=q)

            paginator = Paginator(episodes_qs, page_size)
            page_num  = int(self.request.GET.get('page', 1))
            page_obj  = paginator.get_page(page_num)
            return paginator, page_obj, list(page_obj.object_list), page_obj.has_other_pages()

        # 4) FULL-TEXT / ELASTICSEARCH branch

        # 4a) Optional date window
        date_filter = self.request.GET.get('search_date', 'anytime')
        window = None
        if date_filter != 'anytime':
            days = int(date_filter)
            window = (
                timezone.timedelta(hours=24)
                if days == 24
                else timezone.timedelta(days=days)
            )

        # 4b) Transcript-only ES query
        selected = set(self.request.GET.getlist('search_in'))
        transcript_only = {
            {self.SEGMENT_FIELD},
            {self.SEGMENT_ALIAS_FIELD},
            {self.TRANSCRIPTS_FIELD},
        }
        if selected in transcript_only:
            tsearch = TranscriptDocument.search().query(
                'match',
                segment_text={'query': q}
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

            total = tsearch.count()
            page  = int(self.request.GET.get('page', 1))
            start = (page - 1) * page_size
            end   = start + page_size

            tresp  = tsearch.sort({'_score': 'desc'})[start:end].execute()
            ep_ids = [hit.episode_id for hit in tresp]

            episodes = list(
                Episode.objects
                       .filter(id__in=ep_ids)
                       .select_related('channel')
                       .prefetch_related('transcripts')
            )
            id_map    = {e.id: e for e in episodes}
            page_list = [id_map[i] for i in ep_ids if i in id_map]

            paginator = Paginator(range(total), page_size)
            try:
                page_obj = paginator.page(page)
            except:
                page_obj = paginator.page(1)

            return paginator, page_obj, page_list, page_obj.has_other_pages()

        # 4c) Default multi-match ES query
        es = EpisodeDocument.search()
        if window:
            es = es.filter('range', publication_date={'gte': timezone.now() - window})

        fields = [
            'episode_title',
            'description',
            'channel.channel_title',
            'translations.episode_title',
            'translations.description',
            'transcripts.segment_text',
        ]
        es = es.query('multi_match', query=q, fields=fields)
        es = es.sort({'_score': 'desc'}, {'publication_date': 'desc'})

        total = es.count()
        page  = int(self.request.GET.get('page', 1))
        start = (page - 1) * page_size
        end   = start + page_size

        resp = es[start:end].execute()
        ids  = [int(hit.meta.id) for hit in resp]

        episodes = list(
            Episode.objects
                   .filter(id__in=ids)
                   .select_related('channel')
                   .prefetch_related('transcripts')
        )
        id_map    = {e.id: e for e in episodes}
        page_list = [id_map[i] for i in ids if i in id_map]

        paginator = Paginator(range(total), page_size)
        try:
            page_obj = paginator.page(page)
        except:
            page_obj = paginator.page(1)

        return paginator, page_obj, page_list, page_obj.has_other_pages()


    def _compute_suggestions(self, q, limit=10):
        # your existing episode‐based trigram code
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
        # new channel‐based trigram code
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
        search_type = self.request.GET.get('search_type','episodes')

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
            logger.debug("→ Rendering AJAX with template %r", tpl)
            return render(self.request, tpl, context)

        tpl = ( search_type=='channels'
                and 'podcasts/search_results_ch.html'
                or 'podcasts/search_results.html' )
        logger.debug("→ Rendering non-AJAX with template %r", tpl)
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
        # Get IDs of channels that the user follows.
        channel_ids = ChannelInteraction.objects.filter(
            user=self.request.user,
            followed=True
        ).values_list('channel_id', flat=True)
        qs = Channel.objects.filter(id__in=channel_ids).order_by('channel_title')
        qs = qs.annotate(
            favorites_count=Count(
                'channel_interactions',
                filter=Q(channel_interactions__followed=True),
                distinct=True
            ),
            notifications_count=Count(
                'channel_interactions',
                filter=Q(channel_interactions__notifications_enabled=True),
                distinct=True
            ),
            avg_rating=Avg('channel_interactions__rating'),
            rating_count=Count(
                'channel_interactions__rating',
                distinct=True
            ),
            total_views=Sum('channelvisit__count')
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
        # Get IDs of channels that the user has enabled notifications for.
        channel_ids = ChannelInteraction.objects.filter(
            user=self.request.user,
            notifications_enabled=True
        ).values_list('channel_id', flat=True)
        qs = Episode.objects.filter(channel__id__in=channel_ids).order_by('-publication_date')
        qs = qs.annotate(
            bookmarks_count=Count(
                'episode_interactions',
                filter=Q(episode_interactions__bookmarked=True),
                distinct=True
            ),
            ep_avg_rating=Avg('episode_interactions__rating'),
            ep_rating_count=Count(
                'episode_interactions__rating',
                distinct=True
            ),
            total_episode_views=Sum('episodevisit__count')
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
    template_name = 'podcasts/bookmarks_list.html'
    context_object_name = 'episodes'
    paginate_by = 5

    def get_queryset(self):
        # Get IDs of episodes bookmarked by the user.
        episode_ids = EpisodeInteraction.objects.filter(
            user=self.request.user,
            bookmarked=True
        ).values_list('episode_id', flat=True)
        qs = Episode.objects.filter(id__in=episode_ids).order_by('-publication_date')
        qs = qs.annotate(
            bookmarks_count=Count(
                'episode_interactions',
                filter=Q(episode_interactions__bookmarked=True),
                distinct=True
            ),
            ep_avg_rating=Avg('episode_interactions__rating'),
            ep_rating_count=Count(
                'episode_interactions__rating',
                distinct=True
            ),
            total_episode_views=Sum('episodevisit__count')
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

