# podcasts/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from phonenumber_field.formfields import SplitPhoneNumberField
from django.contrib.auth import get_user_model
from .models import CustomUser

class OTPChallengeForm(forms.Form):
    token = forms.CharField(label="OTP Token", max_length=6, widget=forms.TextInput(attrs={'placeholder': 'Enter OTP token'}))

class Disable2FAForm(forms.Form):
    token = forms.CharField(
        label="OTP Token",
        max_length=6,
        widget=forms.TextInput(attrs={'placeholder': 'Enter OTP token'})
    )

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