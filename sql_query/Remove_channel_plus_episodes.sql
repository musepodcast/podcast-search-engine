BEGIN;

DELETE FROM episodes
WHERE channel_id = 23979;

DELETE FROM channels
WHERE id = 23979;

COMMIT;
