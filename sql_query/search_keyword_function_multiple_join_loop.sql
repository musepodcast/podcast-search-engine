WITH keywords(term) AS (
  VALUES
    ('covid'),
    (' rona '),
	('pandemic'),
	('corona'),
	('sv'),
	('fauci'),
	('virus'),
    ('jab'),
    ('vaccine'),
    ('vaccine injury'),
    ('insurance data'),
    ('all-cause mortality'),
    ('all cause mortality'),
    ('ivermectin'),
    ('hydroxychloroquine')
)
SELECT
  f.term,
  f.episodes_with_keyword,
  f.total_mentions,
  f.first_mention_episode_id,
  f.first_mention_episode_title,
  f.first_mention_publication_date,
  f.matched_channel_title
FROM keywords k,
LATERAL search_keyword(k.term, 'Real AF with andy frisella') AS f;
