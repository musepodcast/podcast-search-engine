from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('podcasts', '0015__enable_pgtrgm'),
    ]
    operations = [
        migrations.RunSQL(
            sql="CREATE INDEX IF NOT EXISTS episodes_pubdate_idx ON episodes (publication_date);",
            reverse_sql="DROP INDEX IF EXISTS episodes_pubdate_idx;"
        ),
    ]
