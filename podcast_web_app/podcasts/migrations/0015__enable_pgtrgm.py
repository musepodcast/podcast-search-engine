from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('podcasts', '0014_search_vector_triggers'),
    ]

    operations = [
        # Enable pg_trgm extension for trigram similarity
        migrations.RunSQL(
            sql="CREATE EXTENSION IF NOT EXISTS pg_trgm;",
            reverse_sql="DROP EXTENSION IF EXISTS pg_trgm;"
        ),
    ]
