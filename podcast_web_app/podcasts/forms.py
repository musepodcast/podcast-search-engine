# podcasts/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from phonenumber_field.formfields import SplitPhoneNumberField
from django.contrib.auth import get_user_model
from .models import CustomUser, SupportTicket
from allauth.socialaccount.forms import SignupForm
from django.utils.translation import gettext_lazy as _

class CustomSignupForm(SignupForm):
    first_name   = forms.CharField(max_length=150, required=False)
    last_name    = forms.CharField(max_length=150, required=False)
    birthdate    = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    country      = forms.CharField(max_length=100, required=False)
    phone_number = forms.CharField(max_length=50, required=False)
    gender       = forms.ChoiceField(choices=[("M","Male"),("F","Female"),("O","Other")], required=False)

    def save(self, request):
        # Creates the user via allauth's normal flow
        user = super().save(request)

        # Persist your extra fields (assuming they exist on CustomUser)
        cd = self.cleaned_data
        user.first_name   = cd.get("first_name", "")
        user.last_name    = cd.get("last_name", "")
        if hasattr(user, "birthdate"):    user.birthdate    = cd.get("birthdate")
        if hasattr(user, "country"):      user.country      = cd.get("country")
        if hasattr(user, "phone_number"): user.phone_number = cd.get("phone_number")
        if hasattr(user, "gender"):       user.gender       = cd.get("gender")

        # Keep manual users INACTIVE until they confirm email
        user.is_active = False
        user.save()
        return user

class OTPChallengeForm(forms.Form):
    token = forms.CharField(label="OTP Token", max_length=6, widget=forms.TextInput(attrs={'placeholder': 'Enter OTP token'}))

class Disable2FAForm(forms.Form):
    token = forms.CharField(
        label="OTP Token",
        max_length=6,
        widget=forms.TextInput(attrs={'placeholder': 'Enter OTP token'})
    )

class CustomSocialSignupForm(SignupForm):
    # override the parent’s email field:
    email = forms.EmailField(
        required=True,
        label=_("Email")
    )
    username     = forms.CharField(required=True, label=_("Username"))
    birthdate    = forms.DateField(required=True, widget=forms.DateInput(attrs={"type": "date"}))
    country      = forms.CharField(required=True)
    phone_number = SplitPhoneNumberField(region="US", required=True)
    gender       = forms.ChoiceField(choices=CustomUser.GENDER_CHOICES, required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'].required = True
        # Prevent Django from tacking on “(optional)”
        self.fields['email'].label_suffix = ''

    def save(self, request):
        user = super().save(request)
        cd   = self.cleaned_data
        user.email        = cd["email"]
        user.username     = cd["username"]
        user.birthdate    = cd["birthdate"]
        user.country      = cd["country"]
        user.phone_number = cd["phone_number"]
        user.gender       = cd["gender"]
        user.save(update_fields=[
            "email","username","birthdate","country","phone_number","gender"
        ])
        return user

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    birthdate = forms.DateField(required=True, widget=forms.DateInput(attrs={'type': 'date'}))
    phone_number = SplitPhoneNumberField(region="US", required=True)
    gender = forms.ChoiceField(required=True, choices=CustomUser.GENDER_CHOICES)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make text inputs full-width Bootstrap style
        for name in [
            "username","email","first_name","last_name",
            "birthdate","country","password1","password2"
        ]:
            if name in self.fields:
                self.fields[name].widget.attrs.setdefault("class", "form-control w-100")

        # Selects (e.g., gender) as Bootstrap selects
        if "gender" in self.fields:
            self.fields["gender"].widget.attrs.setdefault("class", "form-select w-100")

        # SplitPhoneNumberField subwidgets: [0]=country code/select, [1]=national number
        pw = self.fields["phone_number"].widget
        try:
            pw.widgets[0].attrs.update({"class": "form-select w-100", "aria-label": "Country Code"})
            pw.widgets[1].attrs.update({"class": "form-control w-100", "placeholder": "Phone number", "aria-label": "Phone number"})
        except Exception:
            # fallback if widget shape changes; won't crash
            pass

    class Meta:
        model = CustomUser
        fields = (
            "username", "email", "first_name", "last_name",
            "birthdate", "country", "phone_number", "gender",
        )

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if CustomUser.objects.filter(email__iexact=email).exists():
            raise ValidationError("An account with that email already exists.")
        return email
    
User = get_user_model()

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        # Now include the new fields so users can update them.
        fields = ['first_name', 'last_name', 'birthdate', 'country', 'phone_number', 'gender', 'enforce_2fa']


class CustomAuthenticationForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Change the label to "Username or email"
        self.fields['username'].label = "Username or email"

class SupportTicketForm(forms.ModelForm):
    attachment = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={
            'accept': '.jpg,.jpeg,.png,.gif',
        }),
        help_text="One image (JPG/PNG/GIF), ≤2 MB."
    )

    class Meta:
        model  = SupportTicket
        fields = ['subject', 'message', 'attachment']
        widgets= {
            'subject': forms.TextInput(attrs={
                'class':'form-control',
                'placeholder':'Subject'
            }),
            'message': forms.Textarea(attrs={
                'class':'form-control',
                'rows':4,
                'placeholder':'Describe your request…',
                'maxlength':'3000',
            }),
        }

    def clean_attachment(self):
        f = self.cleaned_data.get('attachment')
        if f:
            errors = []
            # size check
            if f.size > 2 * 1024 * 1024:
                errors.append(_("File size is too large, must be ≤ 2 MB."))
            # type check
            if f.content_type not in ('image/jpeg', 'image/png', 'image/gif'):
                errors.append(_("Only JPG, PNG or GIF files are accepted."))
            if errors:
                # raise all errors at once
                raise forms.ValidationError(errors)
        return f