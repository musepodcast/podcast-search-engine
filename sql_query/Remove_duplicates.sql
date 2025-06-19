DELETE FROM Episodes
WHERE id NOT IN (
    SELECT MIN(id)
    FROM Episodes
    GROUP BY channel_id, episode_title
);
