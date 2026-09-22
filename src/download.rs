use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

pub const DEFAULT_MODEL_DIR: &str = "data";
pub const DEFAULT_MODEL_FILENAME: &str = "Hy-MT2-1.8B-1.25Bit.gguf";
pub const DEFAULT_MODEL_REPO: &str = "tencent/Hy-MT2-1.8B-1.25Bit-GGUF";
pub const DEFAULT_MODEL_REVISION: &str = "9df5c824a00a744fb0512a29c640466f4d97dfb0";
pub const DEFAULT_MODEL_SHA256: &str =
    "cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93";
pub const DEFAULT_MODEL_BYTES: u64 = 461_860_800;

/// Returns the default data directory path.
pub fn default_data_dir() -> PathBuf {
    std::env::var_os("HY_MT_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR))
}

/// Returns the default model path within the given or default data directory.
pub fn default_model_path(dir: Option<&Path>) -> PathBuf {
    match dir {
        Some(d) => d.join(DEFAULT_MODEL_FILENAME),
        None => default_data_dir().join(DEFAULT_MODEL_FILENAME),
    }
}

/// Computes the Hugging Face download URL for the default model, honoring `HF_ENDPOINT`.
pub fn default_model_url() -> String {
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let endpoint = endpoint.trim_end_matches('/');
    format!(
        "{endpoint}/{DEFAULT_MODEL_REPO}/resolve/{DEFAULT_MODEL_REVISION}/{DEFAULT_MODEL_FILENAME}?download=true"
    )
}

/// Computes the SHA-256 digest of a file.
pub fn compute_sha256(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {} for verification", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024]; // 1MB buffer
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verifies whether the file at `path` matches the expected SHA-256 hex string.
pub fn verify_file_sha256(path: &Path, expected_sha256: &str) -> Result<bool> {
    let digest = compute_sha256(path)?;
    Ok(digest.eq_ignore_ascii_case(expected_sha256))
}

/// Synchronously ensures the default 2B-1.25Bit model is downloaded and verified in the data folder.
pub fn ensure_default_model(dir: Option<&Path>) -> Result<PathBuf> {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(download_default_model(dir, false)))
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(download_default_model(dir, false))
    }
}

/// Asynchronously downloads and verifies the default 2B-1.25Bit model.
pub async fn download_default_model(dir: Option<&Path>, force: bool) -> Result<PathBuf> {
    let url = default_model_url();
    download_model_from_url(
        &url,
        dir,
        DEFAULT_MODEL_FILENAME,
        DEFAULT_MODEL_SHA256,
        Some(DEFAULT_MODEL_BYTES),
        force,
    )
    .await
}

/// Asynchronously downloads a model file from `url` into `dir`, verifying its SHA-256.
pub async fn download_model_from_url(
    url: &str,
    dir: Option<&Path>,
    filename: &str,
    expected_sha256: &str,
    expected_bytes: Option<u64>,
    force: bool,
) -> Result<PathBuf> {
    use std::io::Write;

    let target_dir = dir.map(Path::to_path_buf).unwrap_or_else(default_data_dir);
    std::fs::create_dir_all(&target_dir)
        .with_context(|| format!("Failed to create directory {}", target_dir.display()))?;

    let final_path = target_dir.join(filename);
    let part_path = target_dir.join(format!("{filename}.part"));

    if !force && final_path.exists() {
        let matches_size = match expected_bytes {
            Some(expected) => {
                let meta = std::fs::metadata(&final_path)?;
                meta.len() == expected
            }
            None => true,
        };

        if matches_size {
            tracing::info!("Verifying existing model at {}...", final_path.display());
            if verify_file_sha256(&final_path, expected_sha256)? {
                tracing::info!("Model at {} verified successfully", final_path.display());
                return Ok(final_path);
            } else {
                tracing::warn!(
                    "Checksum mismatch for {}; re-downloading...",
                    final_path.display()
                );
            }
        } else {
            tracing::warn!(
                "Existing file at {} has incorrect size; re-downloading...",
                final_path.display()
            );
        }
    }

    tracing::info!(
        "Downloading 2B-1.25Bit model from {url} to {}...",
        final_path.display()
    );

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")?;

    let mut response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to initiate download from {url}"))?;

    let status = response.status();
    ensure!(
        status.is_success(),
        "Failed to download model from {url}: HTTP {status}"
    );

    let total_size = response.content_length().or(expected_bytes).unwrap_or(0);

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .expect("valid progress template")
                .progress_chars("#>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec})")
                .expect("valid spinner template"),
        );
        pb
    };

    // Clean up any stale partial download
    if part_path.exists() {
        let _ = std::fs::remove_file(&part_path);
    }

    let mut file = std::fs::File::create(&part_path)
        .with_context(|| format!("Failed to create temporary file {}", part_path.display()))?;
    let mut hasher = Sha256::new();

    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)
            .with_context(|| format!("Failed to write chunk to {}", part_path.display()))?;
        hasher.update(&chunk);
        pb.inc(chunk.len() as u64);
    }

    file.flush()
        .with_context(|| format!("Failed to flush {}", part_path.display()))?;
    file.sync_all()
        .with_context(|| format!("Failed to sync {}", part_path.display()))?;
    drop(file);

    pb.finish_with_message("Download complete");

    let digest = format!("{:x}", hasher.finalize());
    if !digest.eq_ignore_ascii_case(expected_sha256) {
        let _ = std::fs::remove_file(&part_path);
        bail!("Checksum mismatch for downloaded model: expected {expected_sha256}, got {digest}");
    }

    std::fs::rename(&part_path, &final_path).with_context(|| {
        format!(
            "Failed to rename {} to {}",
            part_path.display(),
            final_path.display()
        )
    })?;

    tracing::info!(
        "Successfully downloaded and verified {}",
        final_path.display()
    );
    Ok(final_path)
}
