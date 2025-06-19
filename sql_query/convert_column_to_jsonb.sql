ALTER TABLE episodes 
ALTER COLUMN categories TYPE jsonb USING to_jsonb(categories);
