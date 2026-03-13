use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use pgvector::Vector;
use rand::rngs::ThreadRng;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::{Rng, thread_rng};
use rayon::prelude::*;
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres};
use std::collections::HashMap;
use std::collections::HashSet;
use tokio::sync::mpsc;
use tokio::task;

#[derive(Clone)]
pub struct DataSetStreamer {
    pool: Pool<Postgres>,
    pub current_offset: usize,
    pub max_id: i32,
    weights: Vec<f32>,
    music_playlists: HashMap<i32, Vec<i32>>,
    pub pb: ProgressBar,
}

impl DataSetStreamer {
    pub async fn new(pool: Pool<Postgres>) -> Self {
        println!("Initializing streamer");

        const PLAYLIST_COUNT: u64 = 1_000_000;
        let pb = ProgressBar::new(PLAYLIST_COUNT);
        pb.set_style(
            ProgressStyle::with_template("[{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
                .unwrap()
                .progress_chars("#>-"),
        );
        let weights = load_freq_weights(&pool).await;
        let music_playlists = load_music_playlists(&pool).await;

        println!("Streamer ready.");
        Self {
            max_id: vocab_size(&pool).await as i32,
            pool,
            current_offset: 0,
            weights,
            music_playlists,
            pb,
        }
    }

    pub async fn next_batch(
        &mut self,
        batch_size: usize,
        window_size: usize,
        k: usize,
    ) -> Option<Vec<(i32, i32, f32)>> {
        let mut batch = Vec::with_capacity(batch_size);

        const AVG_TRACK_SIZE: usize = 70;
        let playlist_num =
            (batch_size as f32 / ((1 + k) as f32 * AVG_TRACK_SIZE as f32)).ceil() as usize * 2;

        while batch.len() < batch_size {
            let playlists = self.next_playlists(playlist_num).await;
            if playlists.is_empty() {
                break;
            }

            let results: Vec<Vec<(i32, i32, f32)>> = playlists
                .into_par_iter()
                .map(|playlist| {
                    let mut rng = thread_rng();

                    let centers: Vec<i32> = playlist
                        .iter()
                        .copied()
                        .choose_multiple(&mut rng, window_size);

                    let positives: Vec<(i32, i32, f32)> = centers
                        .iter()
                        .flat_map(|&center| {
                            playlist
                                .iter()
                                .filter(|&&id| id != center)
                                .copied()
                                .choose_multiple(&mut rng, playlist.len() / window_size)
                                .into_iter()
                                .map(move |ctx_id| (center, ctx_id, 1.0))
                        })
                        .collect();

                    let negatives = self.sample_negatives(&playlist, k);

                    let mut local_batch = Vec::with_capacity(positives.len() + negatives.len());

                    local_batch.extend(negatives.into_iter().enumerate().map(|(i, neg_idx)| {
                        let center = centers[i % centers.len()];
                        (center, neg_idx, 0.0)
                    }));

                    local_batch.extend(positives.into_iter());

                    local_batch
                })
                .collect();

            batch.extend(results.into_iter().flatten());
        }

        if batch.is_empty() {
            None
        } else {
            let mut rng = rand::thread_rng();
            batch.shuffle(&mut rng);
            Some(batch)
        }
    }

    async fn next_playlists(&mut self, limit: usize) -> Vec<Vec<i32>> {
        let playlists: Vec<Vec<i32>> = sqlx::query_scalar!(
            r#"
                SELECT musics FROM playlists 
                WHERE musics IS NOT NULL 
                AND array_length(musics, 1) > 1
                ORDER BY id
                OFFSET $1
                LIMIT $2
            "#,
            self.current_offset as i32,
            limit as i32
        )
        .fetch_all(&self.pool)
        .await
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap().into_iter().map(|id| id - 1).collect())
        .collect();

        self.current_offset += limit;
        self.pb.set_position(self.current_offset as u64);
        playlists
    }

    fn sample_negatives(&self, music_ids: &Vec<i32>, k: usize) -> Vec<i32> {
        let mut forbidden_playlists: HashSet<i32> = HashSet::new();

        for &music_id in music_ids {
            forbidden_playlists.extend(self.music_playlists.get(&music_id).unwrap());
        }

        let mut rng = thread_rng();
        let target_size = k * music_ids.len();
        let mut negatives: Vec<i32> = Vec::with_capacity(target_size * 2);

        while negatives.len() < target_size {
            negatives.extend(
                (0..self.max_id)
                    .choose_multiple(&mut rng, target_size * 2)
                    .into_iter()
                    .filter(|candidate| {
                        self.music_playlists
                            .get(candidate)
                            .unwrap()
                            .into_iter()
                            .all(|id| !forbidden_playlists.contains(id))
                    }),
            )
        }

        negatives.truncate(target_size);

        negatives
    }

    pub async fn spawn_streamer(
        mut self,
        batch_size: usize,
        window_size: usize,
        k: usize,
        buffer: usize,
    ) -> mpsc::Receiver<Vec<(i32, i32, f32)>> {
        let (tx, rx) = mpsc::channel(buffer);

        task::spawn(async move {
            while let Some(batch) = self.next_batch(batch_size, window_size, k).await {
                if tx.send(batch).await.is_err() {
                    break;
                }
            }
        });

        rx
    }

    pub fn reset(&mut self) {
        self.current_offset = 0;
    }
}

pub async fn create_pool(database_url: &str) -> Result<Pool<Postgres>> {
    let pool = PgPoolOptions::new()
        .max_connections(8)
        .connect(database_url)
        .await?;
    Ok(pool)
}

pub async fn load_all_embeddings(pool: &Pool<Postgres>) -> HashMap<i32, Vec<f32>> {
    let rows = sqlx::query!(
        r#"SELECT id, embedding as "embedding: Vector"
               FROM musics
           "#,
    )
    .fetch_all(pool)
    .await
    .unwrap();

    let mut rng = rand::thread_rng();

    rows.into_iter()
        .map(|row| {
            let emb = row
                .embedding
                .map(|vec: Vector| vec.into())
                .unwrap_or_else(|| random_embedding(&mut rng, 64));
            (row.id, emb)
        })
        .collect()
}

pub async fn update_embeddings(
    pool: &Pool<Postgres>,
    id_emb: (Vec<i32>, Vec<Vector>),
) -> Result<()> {
    let (ids, embs): (Vec<i32>, Vec<Vector>) = id_emb;

    let query = r#"
            UPDATE musics AS m
            SET embedding = u.embedding
            FROM (
                SELECT *
                FROM UNNEST($1::vector[], $2::int4[]) AS t(embedding, id)
            ) AS u
            WHERE m.id = u.id;
        "#;

    sqlx::query(query)
        .bind(&embs)
        .bind(&ids)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn load_music_playlists(pool: &Pool<Postgres>) -> HashMap<i32, Vec<i32>> {
    let rows = sqlx::query!(
        r#"
            SELECT id, musics
            FROM playlists
            WHERE musics IS NOT NULL
        "#
    )
    .fetch_all(pool)
    .await
    .unwrap();

    let mut music_playlists: HashMap<i32, Vec<i32>> = HashMap::new();

    for row in rows {
        let playlist_id = row.id;
        let musics = row.musics.unwrap();
        for music_id in musics {
            music_playlists
                .entry(music_id - 1)
                .or_insert_with(Vec::new)
                .push(playlist_id);
        }
    }

    music_playlists
}

pub async fn load_freq_weights(pool: &Pool<Postgres>) -> Vec<f32> {
    let freqs: Vec<i64> = sqlx::query_scalar!(
        r#"
            SELECT freq
            FROM music_freq
            ORDER BY id
        "#
    )
    .fetch_all(pool)
    .await
    .unwrap()
    .into_iter()
    .map(|opt| opt.unwrap())
    .collect();

    freqs
        .into_iter()
        .map(|f| {
            if f > 0 {
                1.0 / (f as f32).powf(0.75)
            } else {
                1.0
            }
        })
        .collect()
}

pub async fn vocab_size(pool: &Pool<Postgres>) -> i64 {
    sqlx::query!(
        r#"
            SELECT COUNT(*) FROM musics
        "#
    )
    .fetch_one(pool)
    .await
    .unwrap()
    .count
    .unwrap()
}

pub fn random_embedding(rng: &mut ThreadRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-0.125..0.125)).collect()
}
