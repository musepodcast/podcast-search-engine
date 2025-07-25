# podcasts/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from phonenumber_field.formfields import SplitPhoneNumberField
from django.contrib.auth import get_user_model
from .models import CustomUser
from allauth.socialaccount.forms import SignupForm

class OTPChallengeForm(forms.Form):
    token = forms.CharField(label="OTP Token", max_length=6, widget=forms.TextInput(attrs={'placeholder': 'Enter OTP token'}))

class Disable2FAForm(forms.Form):
    token = forms.CharField(
        label="OTP Token",
        max_length=6,
        widget=forms.TextInput(attrs={'placeholder': 'Enter OTP token'})
    )

class CustomSocialSignupForm(SignupForm):
    # add username (since you REQUIRE it on your CustomUser)
    username = forms.CharField(required=True, label="Username")
    
    # fields Google won’t supply
    birthdate    = forms.DateField(
        required=True,
        widget=forms.DateInput(attrs={"type": "date"})
    )
    country      = forms.CharField(required=True)
    phone_number = SplitPhoneNumberField(region="US", required=True)
    gender       = forms.ChoiceField(
        choices=CustomUser.GENDER_CHOICES, required=True
    )

    def save(self, request):
        # this will fill in email/first_name/last_name from Google
        user = super().save(request)
        cd = self.cleaned_data
        user.username     = cd["username"]
        user.birthdate    = cd["birthdate"]
        user.country      = cd["country"]
        user.phone_number = cd["phone_number"]
        user.gender       = cd["gender"]
        user.save(update_fields=[
            "username", "birthdate", "country", "phone_number", "gender"
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