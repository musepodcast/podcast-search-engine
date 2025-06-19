# podcasts/migrations/0013_add_search_vector.py
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('podcasts', '0012_alter_episode_options_and_more'),
    ]

    operations = [
        # Add search_vector columns
        migrations.RunSQL(
            sql="""
ALTER TABLE episodes
  ADD COLUMN IF NOT EXISTS search_vector tsvector;

ALTER TABLE episodestranslations
  ADD COLUMN IF NOT EXISTS search_vector tsvector;
""",
            reverse_sql="""
ALTER TABLE episodes DROP COLUMN IF EXISTS search_vector;

ALTER TABLE episodestranslations DROP COLUMN IF EXISTS search_vector;
"""
        ),
        # Create GIN indexes
        migrations.RunSQL(
            sql="""
CREATE INDEX IF NOT EXISTS episodes_search_vector_gin
  ON episodes USING gin(search_vector);

CREATE INDEX IF NOT EXISTS eptrans_search_vector_gin
  ON episodestranslations USING gin(search_vector);
""",
            reverse_sql="""
DROP INDEX IF EXISTS episodes_search_vector_gin;
DROP INDEX IF EXISTS eptrans_search_vector_gin;
"""
        ),
    ]
