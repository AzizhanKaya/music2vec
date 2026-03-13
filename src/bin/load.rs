use anyhow::Result;
use futures::stream::{self, StreamExt};
use itertools::Itertools;
use serde::Deserialize;
use std::{path::Path, sync::Arc};
use tokio::fs;

#[derive(Debug, Deserialize, Default)]
struct Track {
    artist_name: String,
    track_name: String,
}

#[derive(Debug, Deserialize)]
struct Playlist {
    num_tracks: i32,
    #[serde(default)]
    tracks: Vec<Track>,
}

#[derive(Debug, Deserialize)]
struct Root {
    playlists: Vec<Playlist>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let pool = Arc::new(
        sqlx::postgres::PgPoolOptions::new()
            .max_connections(10)
            .connect("postgresql://aziz:asdzxasdzx1@localhost/music2vec")
            .await?,
    );

    let data_path = Path::new("./data");
    let files: Vec<_> = std::fs::read_dir(&data_path)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()?.to_str()? == "json" {
                Some(path)
            } else {
                None
            }
        })
        .sorted()
        .collect();

    println!("Toplam {} dosya bulundu", files.len());

    stream::iter(files.into_iter().enumerate())
        .for_each_concurrent(1, |(idx, path)| {
            let pool = pool.clone();
            async move {
                if let Err(e) = process_file(idx, path, pool).await {
                    eprintln!("Hata ({}): {:?}", idx, e);
                }
            }
        })
        .await;

    Ok(())
}

async fn process_file(idx: usize, path: std::path::PathBuf, pool: Arc<sqlx::PgPool>) -> Result<()> {
    println!("[{}] {:?}", idx + 1, path.file_name());

    let data = fs::read(&path).await?;
    let root: Root = serde_json::from_slice(&data)?;

    for playlist in root.playlists {
        let tracks: Vec<(String, String)> = playlist
            .tracks
            .into_iter()
            .filter(|t| !t.artist_name.is_empty() && !t.track_name.is_empty())
            .map(|t| (t.track_name, t.artist_name))
            .collect();

        if tracks.is_empty() {
            continue;
        }

        let (names, artists): (Vec<String>, Vec<String>) = tracks
            .into_iter()
            .map(|(name, artist)| (name, artist))
            .unzip();

        let rows = sqlx::query!(
            r#"
                INSERT INTO musics (name, artist)
                SELECT DISTINCT * 
                FROM UNNEST($1::text[], $2::text[]) AS t(name, artist)
                ON CONFLICT (name, artist)
                DO UPDATE SET name = EXCLUDED.name
                RETURNING id;
            "#,
            &names,
            &artists
        )
        .fetch_all(&*pool)
        .await?;

        let music_ids: Vec<i32> = rows.into_iter().map(|r| r.id).collect();

        sqlx::query(
            r#"
                INSERT INTO playlists (num_tracks, musics)
                VALUES ($1, $2)
            "#,
        )
        .bind(playlist.num_tracks)
        .bind(&music_ids)
        .execute(&*pool)
        .await?;
    }

    Ok(())
}
