use anyhow::Result;
use csv::Writer;
use ndarray::Array2;
use pgvector::Vector;
use smartcore::decomposition::pca::PCA;
use smartcore::decomposition::pca::PCAParameters;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Arc;

use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    let pool = Arc::new(
        sqlx::postgres::PgPoolOptions::new()
            .max_connections(10)
            .connect("postgresql://aziz:asdzxasdzx1@localhost/music2vec")
            .await?,
    );

    // freq tablosu
    let freq_rows = sqlx::query("SELECT id, freq FROM music_freq")
        .fetch_all(&*pool)
        .await?;

    let mut freq_map: HashMap<i32, i64> = HashMap::new();
    for row in freq_rows {
        let music_id: i32 = row.get("id");
        let freq: i64 = row.get("freq");
        freq_map.insert(music_id, freq);
    }

    // embeddings tablosu
    let rows = sqlx::query(r#"SELECT id, name, artist, embedding FROM musics"#)
        .fetch_all(&*pool)
        .await?;

    let n = rows.len();
    let dim = 64;

    let mut ids = Vec::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    let mut artists = Vec::with_capacity(n);
    let mut freqs = Vec::with_capacity(n);

    let mut data = Array2::<f64>::zeros((n, dim));

    for (i, row) in rows.iter().enumerate() {
        let id: i32 = row.get("id");
        let name: String = row.get("name");
        let artist: String = row.get("artist");

        let embedding: Vector = row.get("embedding");
        let vec = embedding.to_vec();
        if vec.len() != dim {
            panic!(
                "Embedding length mismatch: expected {}, got {}",
                dim,
                vec.len()
            );
        }
        for (j, v) in vec.iter().enumerate() {
            if v.is_nan() || !v.is_finite() {
                panic!("Embedding is NaN or Inf id:{id}")
            } else {
                data[[i, j]] = *v as f64;
            }
        }

        ids.push(id);
        names.push(name);
        artists.push(artist);
        freqs.push(*freq_map.get(&id).unwrap_or(&0));
    }

    let flat_data: Vec<f64> = data.iter().copied().collect();
    let matrix = DenseMatrix::new(n, dim, flat_data, false);

    let pca = PCA::fit(
        &matrix,
        PCAParameters {
            n_components: 2,
            use_correlation_matrix: false,
        },
    )
    .unwrap();
    let embedding_2d = pca.transform(&matrix).unwrap();

    // CSV’ye yaz
    let mut wtr = Writer::from_path("musics_pca.csv")?;
    wtr.write_record(&["id", "name", "artist", "freq", "pc1", "pc2"])?;

    for i in 0..n {
        wtr.write_record(&[
            ids[i].to_string(),
            names[i].clone(),
            artists[i].clone(),
            freqs[i].to_string(),
            embedding_2d.get((i, 0)).to_string(),
            embedding_2d.get((i, 1)).to_string(),
        ])?;
    }

    wtr.flush()?;
    println!("✅ musics_pca.csv kaydedildi!");
    Ok(())
}
