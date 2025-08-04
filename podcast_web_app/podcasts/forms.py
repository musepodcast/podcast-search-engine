# podcasts/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from phonenumber_field.formfields import SplitPhoneNumberField
from django.contrib.auth import get_user_model
from .models import CustomUser, SupportTicket
from allauth.socialaccount.forms import SignupForm
from django.utils.translation import gettext_lazy as _

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
    # Make first and last names required
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    # New fields for signup
    birthdate = forms.DateField(
        required=True, 
        widget=forms.DateInput(attrs={'type': 'date'})
    )
    
    phone_number = SplitPhoneNumberField(region="US", required=True)
        
    gender = forms.ChoiceField(
        required=True, 
        choices=CustomUser.GENDER_CHOICES  # Ensure this exists in your model
    )

    class Meta:
        model = CustomUser
        fields = (
            'username', 'email', 'first_name', 'last_name', 
            'birthdate', 'country', 'phone_number', 'gender'
        )
    
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