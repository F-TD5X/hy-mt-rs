use anyhow::Result;
use axum::{Router, body::Bytes, extract::State, http::StatusCode, routing::get};
use hy_mt_rs::download::{
    DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME, compute_sha256, default_data_dir,
    default_model_path, default_model_url, download_model_from_url, verify_file_sha256,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use tokio::net::TcpListener;

#[test]
fn test_default_paths_and_url() {
    let dir = default_data_dir();
    assert_eq!(dir.to_str().unwrap(), DEFAULT_MODEL_DIR);

    let path = default_model_path(None);
    assert_eq!(
        path,
        std::path::PathBuf::from(DEFAULT_MODEL_DIR).join(DEFAULT_MODEL_FILENAME)
    );

    let custom_dir = std::path::Path::new("/custom/dir");
    let custom_path = default_model_path(Some(custom_dir));
    assert_eq!(custom_path, custom_dir.join(DEFAULT_MODEL_FILENAME));

    let url = default_model_url();
    assert!(url.starts_with("https://huggingface.co/"));
    assert!(url.contains("tencent/Hy-MT2-1.8B-1.25Bit-GGUF"));
    assert!(url.contains("Hy-MT2-1.8B-1.25Bit.gguf?download=true"));
}

#[test]
fn test_sha256_verification() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("test.bin");
    let content = b"hello world";
    std::fs::write(&file_path, content)?;

    let expected_sha256 = format!("{:x}", Sha256::digest(content));
    assert_eq!(
        expected_sha256,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );

    assert_eq!(compute_sha256(&file_path)?, expected_sha256);
    assert!(verify_file_sha256(&file_path, &expected_sha256)?);
    assert!(!verify_file_sha256(
        &file_path,
        "0000000000000000000000000000000000000000000000000000000000000000"
    )?);

    Ok(())
}

#[tokio::test]
async fn test_download_from_mock_server() -> Result<()> {
    let payload = vec![42u8; 65536]; // 64KB
    let payload_bytes = Bytes::from(payload.clone());
    let expected_sha256 = format!("{:x}", Sha256::digest(&payload));
    let payload_len = payload.len() as u64;

    let app = Router::new()
        .route(
            "/model.gguf",
            get(|State(data): State<Bytes>| async move {
                (
                    StatusCode::OK,
                    [("content-length", data.len().to_string())],
                    data,
                )
            }),
        )
        .with_state(payload_bytes);

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("http://{addr}/model.gguf");
    let target_dir = tempdir()?;

    // 1. Initial download
    let downloaded_path = download_model_from_url(
        &url,
        Some(target_dir.path()),
        "test_model.gguf",
        &expected_sha256,
        Some(payload_len),
        false,
    )
    .await?;

    assert!(downloaded_path.exists());
    assert_eq!(std::fs::read(&downloaded_path)?, payload);

    // 2. Subsequent call should use existing file
    let cached_path = download_model_from_url(
        &url,
        Some(target_dir.path()),
        "test_model.gguf",
        &expected_sha256,
        Some(payload_len),
        false,
    )
    .await?;
    assert_eq!(downloaded_path, cached_path);

    // 3. Test checksum mismatch failure
    let bad_sha256 = "1111111111111111111111111111111111111111111111111111111111111111";
    let err = download_model_from_url(
        &url,
        Some(target_dir.path()),
        "bad_model.gguf",
        bad_sha256,
        Some(payload_len),
        false,
    )
    .await;
    assert!(err.is_err());
    // Partial file must be deleted on checksum mismatch
    assert!(!target_dir.path().join("bad_model.gguf.part").exists());
    assert!(!target_dir.path().join("bad_model.gguf").exists());

    // 4. Test corrupted existing file is re-downloaded
    let corrupted_path = target_dir.path().join("corrupted.gguf");
    std::fs::write(&corrupted_path, b"corrupted data")?;
    let redownloaded = download_model_from_url(
        &url,
        Some(target_dir.path()),
        "corrupted.gguf",
        &expected_sha256,
        Some(payload_len),
        false,
    )
    .await?;
    assert_eq!(std::fs::read(&redownloaded)?, payload);

    Ok(())
}
