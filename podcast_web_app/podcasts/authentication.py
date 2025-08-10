# podcasts/authentication.py

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.db.models import Q

UserModel = get_user_model()

class EmailOrUsernameModelBackend(ModelBackend):
    """
    Login with either username or email.
    - If the identifier contains '@', treat it as an email (only).
    - Otherwise treat it as a username (only).
    - Optionally require the email used to log in to be verified.
    """
    def authenticate(self, request, username=None, password=None, **kwargs):
        if username is None:
            username = kwargs.get(UserModel.USERNAME_FIELD)

        if not username:
            return None

        # Disambiguate: email path vs username path
        if "@" in username:
            qs = UserModel.objects.filter(email__iexact=username)
        else:
            qs = UserModel.objects.filter(username__iexact=username)

        try:
            user = qs.get()
        except UserModel.DoesNotExist:
            return None

        # Password + active check (ModelBackend will also check is_active here)
        if not (user.check_password(password) and self.user_can_authenticate(user)):
            return None

        # OPTIONAL: If they typed an email to log in, require that *specific* email to be verified
        if "@" in username:
            if not EmailAddress.objects.filter(
                user=user, email__iexact=username, verified=True
            ).exists():
                return None

        return user