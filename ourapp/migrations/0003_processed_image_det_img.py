# Generated by Django 4.1.1 on 2023-05-22 07:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ourapp', '0002_remove_input_image_id_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='processed_image',
            name='det_img',
            field=models.ImageField(null=True, upload_to='detetcted_images'),
        ),
    ]