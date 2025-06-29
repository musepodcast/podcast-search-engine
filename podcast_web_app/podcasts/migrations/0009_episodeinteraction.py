# Generated by Django 5.1.3 on 2025-04-11 01:21

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("podcasts", "0008_channelinteraction"),
    ]

    operations = [
        migrations.CreateModel(
            name="EpisodeInteraction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("bookmarked", models.BooleanField(default=False)),
                (
                    "rating",
                    models.PositiveSmallIntegerField(
                        blank=True,
                        help_text="Rating from 1 (lowest) to 5 (highest)",
                        null=True,
                    ),
                ),
                (
                    "episode",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="episode_interactions",
                        to="podcasts.episode",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="episode_interactions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Episode Interaction",
                "verbose_name_plural": "Episode Interactions",
                "unique_together": {("user", "episode")},
            },
        ),
    ]
