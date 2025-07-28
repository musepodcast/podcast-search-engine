SELECT
  pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND now() - state_change > interval '60 minutes'
  AND pid <> pg_backend_pid();
