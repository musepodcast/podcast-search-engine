# Generated by Django 5.1.3 on 2025-04-16 02:22

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("podcasts", "0010_comment_commentreaction"),
    ]

    operations = [
        migrations.CreateModel(
            name="Reply",
            fields=[],
            options={
                "verbose_name": "Reply",
                "verbose_name_plural": "Replies",
                "proxy": True,
                "indexes": [],
                "constraints": [],
            },
            bases=("podcasts.comment",),
        ),
        migrations.AddField(
            model_name="comment",
            name="seen_by",
            field=models.ManyToManyField(
                blank=True, related_name="seen_comments", to=settings.AUTH_USER_MODEL
            ),
        ),
    ]
