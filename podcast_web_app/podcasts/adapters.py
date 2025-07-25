# podcasts/adapters.py

from allauth.socialaccount.adapter import DefaultSocialAccountAdapter

class ConditionalSocialAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):
        # Look at what the URL asked for
        process = sociallogin.state.get('process')
        if process == 'signup':
            # Only then clear pk and force extra info form
            sociallogin.user.pk = None
        # Otherwise, leave pk intact and let allauth do a straight login
