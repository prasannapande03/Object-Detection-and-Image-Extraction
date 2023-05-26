from django import forms
from .models import Input_Image

class ImageForm(forms.ModelForm):
    class Meta:
        model = Input_Image
        fields = ['image']

    def save(self, commit_user=None, commit=True):
        instance = super().save(commit=False)
        if commit_user:
            instance.user = commit_user
        if commit:
            instance.save()
            self.save_m2m()
        return instance
