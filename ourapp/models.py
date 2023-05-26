from django.db import models
from django.contrib.auth.models import User


class Input_Image(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, default=1)
    image = models.ImageField(null = True, upload_to= 'images')
    det_img = models.ImageField(null = True, upload_to='detetcted_images')
    detect = models.BooleanField(default = True)

    def __str__(self):
        return self.user.username


class Processed_Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,default=None, null=True)
    det_img = models.ImageField(null = True, upload_to='detetcted_images')
    ext_img = models.ImageField(null = True, upload_to='extracted_images')

    def __str__(self):
        return self.user.username

