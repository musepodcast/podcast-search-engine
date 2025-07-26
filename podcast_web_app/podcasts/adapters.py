# podcasts/adapters.py

from django.shortcuts import redirect
from django.urls import reverse
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.socialaccount.views import LoginCancelledView

class ConditionalSocialAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):
        # your existing signup‑force‑extra‑info logic
        process = sociallogin.state.get("process")
        if process == "signup":
            sociallogin.user.pk = None

    def on_authentication_error(
        self, request, provider_id, error=None,
        exception=None, extra_context=None
    ):
        """
        Called on any OAuth error (including access_denied).
        Redirect straight back to your login page.
        """
        return redirect(reverse("podcasts:login"))

class CancelledRedirectView(LoginCancelledView):
    """ 
    Whenever allauth would show the 'cancelled' page, 
    we instead instantly redirect to our login.
    """
    def get(self, request, *args, **kwargs):
        return redirect(reverse("podcasts:login"))