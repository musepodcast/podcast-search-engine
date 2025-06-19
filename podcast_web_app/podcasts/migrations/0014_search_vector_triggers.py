# podcasts/migrations/0014_search_vector_triggers.py
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('podcasts', '0013_add_search_vector'),
    ]

    operations = [
        # Create trigger function to update search_vector
        migrations.RunSQL(
            sql="""
CREATE OR REPLACE FUNCTION podcasts_update_search_vector() RETURNS trigger AS $$
begin
  new.search_vector :=
    setweight(to_tsvector('english', coalesce(new.episode_title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(new.description,   '')), 'B');
  return new;
end
$$ LANGUAGE plpgsql;
""",
            reverse_sql="DROP FUNCTION IF EXISTS podcasts_update_search_vector();"
        ),
        # Attach triggers to episodes
        migrations.RunSQL(
            sql="""
DROP TRIGGER IF EXISTS episodes_search_vector_update ON episodes;
CREATE TRIGGER episodes_search_vector_update
  BEFORE INSERT OR UPDATE ON episodes
  FOR EACH ROW EXECUTE FUNCTION podcasts_update_search_vector();

DROP TRIGGER IF EXISTS eptrans_search_vector_update ON episodestranslations;
CREATE TRIGGER eptrans_search_vector_update
  BEFORE INSERT OR UPDATE ON episodestranslations
  FOR EACH ROW EXECUTE FUNCTION podcasts_update_search_vector();
""",
            reverse_sql="""
DROP TRIGGER IF EXISTS episodes_search_vector_update ON episodes;
DROP TRIGGER IF EXISTS eptrans_search_vector_update ON episodestranslations;
"""
        ),
    ]
