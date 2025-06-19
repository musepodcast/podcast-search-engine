SELECT channel_id, episode_title, COUNT(*)
FROM Episodes
GROUP BY channel_id, episode_title
HAVING COUNT(*) > 1;
