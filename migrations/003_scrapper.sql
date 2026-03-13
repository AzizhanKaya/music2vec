-- users tablosu
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    user_id TEXT NOT NULL UNIQUE
);
-- playlist_links tablosu
CREATE TABLE playlist_links (
    id SERIAL PRIMARY KEY,
    playlist_id TEXT NOT NULL
);