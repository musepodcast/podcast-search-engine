DROP FUNCTION IF EXISTS search_keyword(TEXT, TEXT);

CREATE OR REPLACE FUNCTION search_keyword(search_term TEXT, channel_name TEXT)
RETURNS TABLE (
  term TEXT,
  episodes_with_keyword INT,
  total_mentions INT,
  first_mention_episode_id INT,
  first_mention_episode_title TEXT,
  first_mention_publication_date TIMESTAMP,
  matched_channel_title TEXT
) AS $$
DECLARE
  chan_id INT;
BEGIN
  -- Use ILIKE for case-insensitive channel title match
  SELECT id INTO chan_id
  FROM channels
  WHERE channel_title ILIKE channel_name
  LIMIT 1;

  IF chan_id IS NULL THEN
    RAISE EXCEPTION 'Channel "%" not found.', channel_name;
  END IF;

  RETURN QUERY
  WITH channel_episodes AS (
    SELECT e.id, e.episode_title, e.publication_date
    FROM episodes e
    WHERE e.channel_id = chan_id
  ),
  mentions AS (
    SELECT
      t.episode_id,
      SUM((LENGTH(LOWER(t.segment_text)) - LENGTH(REPLACE(LOWER(t.segment_text), LOWER(search_term), ''))) / LENGTH(search_term)) AS keyword_hits
    FROM transcripts t
    JOIN channel_episodes ce ON t.episode_id = ce.id
    WHERE LOWER(t.segment_text) LIKE '%' || LOWER(search_term) || '%'
    GROUP BY t.episode_id
  ),
  first_ep AS (
    SELECT e.id AS first_mention_episode_id, e.episode_title, e.publication_date
    FROM episodes e
    WHERE e.id IN (SELECT episode_id FROM mentions)
    ORDER BY e.publication_date ASC
    LIMIT 1
  )
  SELECT
    search_term,
    COUNT(m.episode_id)::INT,
    SUM(m.keyword_hits)::INT,
    f.first_mention_episode_id,
    f.episode_title,
    f.publication_date,
    channel_name
  FROM mentions m
  JOIN first_ep f ON TRUE
  GROUP BY
    search_term,
    f.first_mention_episode_id,
    f.episode_title,
    f.publication_date,
    channel_name;
END;
$$ LANGUAGE plpgsql;
