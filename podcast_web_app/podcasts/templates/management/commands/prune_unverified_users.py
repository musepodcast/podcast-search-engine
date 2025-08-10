# podcasts/management/commands/prune_unverified_users.py
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import OuterRef, Exists, Q
from django.utils import timezone
from django.conf import settings

from allauth.account.models import EmailAddress

from podcasts.models import CustomUser


class Command(BaseCommand):
    help = "Delete unverified, inactive accounts older than N days (default: 1 day)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=1,
            help="Age threshold in days (default: 1).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do not delete anything; just print what would happen.",
        )
        parser.add_argument(
            "--verbose-list",
            action="store_true",
            help="List each user that is pruned.",
        )

    def handle(self, *args, **opts):
        days = opts["days"]
        dry_run = opts["dry_run"]
        verbose_list = opts["verbose_list"]

        cutoff = timezone.now() - timedelta(days=days)

        # Annotate whether a user has any verified EmailAddress
        unverified_qs = (
            CustomUser.objects.filter(
                is_active=False,
                is_staff=False,
                is_superuser=False,
                last_login__isnull=True,
                date_joined__lt=cutoff,
            )
            .annotate(
                has_verified_email=Exists(
                    EmailAddress.objects.filter(user=OuterRef("pk"), verified=True)
                )
            )
            .filter(Q(has_verified_email=False))
        )

        count = unverified_qs.count()
        if count == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to prune."))
            return

        self.stdout.write(
            self.style.WARNING(
                f"Found {count} unverified, inactive account(s) older than {days} day(s)."
            )
        )

        if verbose_list:
            for u in unverified_qs.only("id", "username", "email"):
                self.stdout.write(f"- id={u.id} user={u.username} email={u.email}")

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run: no deletions performed."))
            return

        # Delete in small batches to be polite to the DB
        CHUNK = 200
        deleted = 0
        ids = list(unverified_qs.values_list("id", flat=True))
        for i in range(0, len(ids), CHUNK):
            chunk_ids = ids[i : i + CHUNK]
            deleted += CustomUser.objects.filter(id__in=chunk_ids).delete()[0]

        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} account(s)."))
