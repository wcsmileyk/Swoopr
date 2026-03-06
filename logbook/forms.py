from django import forms
from .models import Aircraft, Dropzone, Jump


class SessionHeaderForm(forms.Form):
    date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'})
    )
    dropzone = forms.ModelChoiceField(queryset=Dropzone.objects.all())
    aircraft = forms.ModelChoiceField(queryset=Aircraft.objects.all(), required=False)


class JumpEditForm(forms.ModelForm):
    class Meta:
        model = Jump
        fields = [
            'date', 'dropzone', 'aircraft', 'canopy', 'jump_type',
            'altitude', 'swoop', 'turn_rotation', 'formation_size', 'notes',
        ]
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 3}),
        }
