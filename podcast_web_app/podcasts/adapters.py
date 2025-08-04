# podcasts/adapters.py

from django.shortcuts import redirect
from django.urls import reverse
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.socialaccount.views import LoginCancelledView
from allauth.socialaccount.models import SocialApp
from django.contrib.sites.models import Site

class ConditionalSocialAdapter(DefaultSocialAccountAdapter):
    def get_app(self, request, provider, client_id=None):
        """
        Return the SocialApp instance for the given provider, scoped to the current site.
        """
        # Determine provider identifier
        provider_id = provider.id if hasattr(provider, 'id') else provider

        # Filter apps for this provider (and client_id if provided)
        apps = SocialApp.objects.filter(provider=provider_id)
        if client_id:
            apps = apps.filter(client_id=client_id)

        # Scope to the current site
        current_site = Site.objects.get_current(request)
        apps = apps.filter(sites=current_site)

        # Return the single matching SocialApp or raise
        return apps.get()

    def pre_social_login(self, request, sociallogin):
        # Preserve existing signup flow: force extra info step
        process = sociallogin.state.get("process")
        if process == "signup":
            sociallogin.user.pk = None

    def on_authentication_error(
        self, request, provider_id, error=None,
        exception=None, extra_context=None
    ):
        # Redirect to login on any authentication error
        return redirect(reverse("podcasts:login"))

class CancelledRedirectView(LoginCancelledView):
    """Redirect to login instead of showing a cancelled page."""
    def get(self, request, *args, **kwargs):
        return redirect(reverse("podcasts:login"))
