from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, Canopy


class MultipleFileInput(forms.ClearableFileInput):
    """Custom widget for multiple file uploads"""
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    """Custom field for multiple file uploads"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        # Handle single file uploads that come through as a single file
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
            return result
        return single_file_clean(data, initial)


class SignUpForm(UserCreationForm):
    """Extended user registration form"""
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=False)
    last_name = forms.CharField(max_length=30, required=False)

    # Pilot info fields
    license_number = forms.CharField(max_length=20, required=False, help_text="Skydiving license number")
    uspa_number = forms.CharField(max_length=20, required=False, help_text="USPA member number")
    license_level = forms.ChoiceField(
        choices=[('', 'Select License Level')] + UserProfile._meta.get_field('license_level').choices,
        required=False
    )

    total_jumps = forms.IntegerField(required=False, min_value=0, help_text="Total number of jumps")
    swoop_jumps = forms.IntegerField(required=False, min_value=0, help_text="Number of swoop jumps")
    exit_weight = forms.FloatField(required=False, min_value=50, help_text="Exit weight in lbs")
    home_dz = forms.CharField(max_length=100, required=False, help_text="Home drop zone")

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2', 'first_name', 'last_name')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add CSS classes for styling
        self.fields['username'].widget.attrs.update({'class': 'form-control'})
        self.fields['email'].widget.attrs.update({'class': 'form-control'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control'})
        self.fields['first_name'].widget.attrs.update({'class': 'form-control'})
        self.fields['last_name'].widget.attrs.update({'class': 'form-control'})
        self.fields['license_number'].widget.attrs.update({'class': 'form-control'})
        self.fields['uspa_number'].widget.attrs.update({'class': 'form-control'})
        self.fields['license_level'].widget.attrs.update({'class': 'form-control'})
        self.fields['total_jumps'].widget.attrs.update({'class': 'form-control'})
        self.fields['swoop_jumps'].widget.attrs.update({'class': 'form-control'})
        self.fields['exit_weight'].widget.attrs.update({'class': 'form-control'})
        self.fields['home_dz'].widget.attrs.update({'class': 'form-control'})

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']

        if commit:
            user.save()

            # Update the UserProfile that was automatically created
            profile = user.profile
            profile.license_number = self.cleaned_data['license_number']
            profile.uspa_number = self.cleaned_data['uspa_number']
            profile.license_level = self.cleaned_data['license_level']
            profile.total_jumps = self.cleaned_data['total_jumps']
            profile.swoop_jumps = self.cleaned_data['swoop_jumps']
            profile.exit_weight = self.cleaned_data['exit_weight']
            profile.home_dz = self.cleaned_data['home_dz']
            profile.save()

        return user


class UserLoginForm(forms.Form):
    """Custom login form"""
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Username',
            'autofocus': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password'
        })
    )
    remember_me = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )


class CanopyForm(forms.ModelForm):
    """Form for adding/editing canopies"""

    class Meta:
        model = Canopy
        fields = ['manufacturer', 'model', 'size', 'line_set', 'modifications', 'is_primary']
        widgets = {
            'manufacturer': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Icarus, Performance Designs'}),
            'model': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Sabre2, Katana'}),
            'size': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Size in sq ft'}),
            'line_set': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Line set type if modified'}),
            'modifications': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Any modifications made'}),
            'is_primary': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

    def save(self, commit=True):
        canopy = super().save(commit=False)
        if self.user:
            canopy.user = self.user
        if commit:
            canopy.save()
        return canopy


class UserProfileForm(forms.ModelForm):
    """Form for editing user profile"""

    class Meta:
        model = UserProfile
        fields = [
            'license_number', 'uspa_number', 'license_level', 'coach', 'affi', 'ti',
            'exit_weight', 'home_dz', 'phone',
            'emergency_contact', 'emergency_phone', 'units', 'public_profile', 'auto_public_flights'
        ]
        widgets = {
            'license_number': forms.TextInput(attrs={'class': 'form-control'}),
            'uspa_number': forms.TextInput(attrs={'class': 'form-control'}),
            'license_level': forms.Select(attrs={'class': 'form-control'}),
            'coach': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'affi': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'ti': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'exit_weight': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'home_dz': forms.TextInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'emergency_contact': forms.TextInput(attrs={'class': 'form-control'}),
            'emergency_phone': forms.TextInput(attrs={'class': 'form-control'}),
            'units': forms.Select(attrs={'class': 'form-control'}),
            'public_profile': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'auto_public_flights': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }


class FlightUploadForm(forms.Form):
    """Form for uploading FlySight CSV files"""

    file = MultipleFileField(
        widget=MultipleFileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.CSV',
            'multiple': True
        }),
        help_text="Select one or more FlySight CSV files to upload"
    )
    canopy = forms.ModelChoiceField(
        queryset=None,
        empty_label="Select canopy (or use your primary)",
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

        if self.user:
            self.fields['canopy'].queryset = Canopy.objects.filter(user=self.user)
        else:
            self.fields['canopy'].queryset = Canopy.objects.none()

    def clean_file(self):
        files = self.cleaned_data.get('file')
        if not files:
            return files

        # Handle single file (not in a list)
        if not isinstance(files, list):
            files = [files]

        for file in files:
            if not file.name.lower().endswith('.csv'):
                raise forms.ValidationError(f"File '{file.name}' must be a CSV file.")

            if file.size > 10 * 1024 * 1024:
                raise forms.ValidationError(f"File '{file.name}' size must be less than 10MB.")

        return files


class BulkPrivacyForm(forms.Form):
    """Form for bulk setting flight privacy"""

    PRIVACY_CHOICES = [
        ('make_public', 'Make Public'),
        ('make_private', 'Make Private'),
    ]

    action = forms.ChoiceField(
        choices=PRIVACY_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    flight_ids = forms.CharField(
        widget=forms.HiddenInput()
    )


class UserSearchForm(forms.Form):
    """Form for searching public users"""

    query = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search by username, name, or drop zone...',
            'autofocus': True
        }),
        required=False
    )

    license_level = forms.ChoiceField(
        choices=[('', 'Any License')] + UserProfile._meta.get_field('license_level').choices,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    home_dz = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Home drop zone'
        })
    )