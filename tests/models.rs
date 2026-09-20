//! Opt-in full-model tests. Download models separately; run in release mode.
use anyhow::Result;
use hy_mt_rs::{
    Model, Profile,
    generation::{self, Options, Sampling},
    tokenizer::ChatMessage,
};
use rayon::ThreadPoolBuilder;
use serde_json::Value;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

fn verify(name: &str) -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest: Value = serde_json::from_str(include_str!("fixtures/models.json"))?;
    let model_dir = std::env::var_os("HY_MT_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("models"));
    let model = Model::open(
        model_dir.join(manifest[name]["file"].as_str().unwrap()),
        Profile::Auto,
    )?;
    let reference: Value = serde_json::from_str(include_str!("fixtures/model_reference.json"))?;
    let reference = &reference[name];
    let tokenizer_fixtures: Value = serde_json::from_str(include_str!("fixtures/tokenizers.json"))?;
    let tokenizer = &tokenizer_fixtures[reference["tokenizer"].as_str().unwrap()];
    for case in tokenizer["cases"].as_array().unwrap() {
        let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone())?;
        let rendered = model.tokenizer.render(&messages)?;
        assert_eq!(rendered, case["rendered"].as_str().unwrap());
        let expected: Vec<u32> = serde_json::from_value(case["tokens"].clone())?;
        assert_eq!(model.tokenizer.encode(&rendered)?, expected);
    }
    let tokens: Vec<u32> = serde_json::from_value(reference["prompt_tokens"].clone())?;
    let bytes = std::fs::read(
        root.join("tests/fixtures")
            .join(reference["logits_file"].as_str().unwrap()),
    )?;
    let expected: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let pool = ThreadPoolBuilder::new().num_threads(4).build()?;
    let mut session = model.new_session(512)?;
    let logits =
        pool.install(|| model.prefill(&mut session, &tokens, &CancellationToken::new()))?;
    assert_eq!(logits.len(), expected.len());
    let square_error: f64 = logits
        .iter()
        .zip(&expected)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum();
    let energy: f64 = expected.iter().map(|x| (*x as f64).powi(2)).sum();
    let relative_l2 = (square_error / energy).sqrt();
    eprintln!("{name}: reference relative L2 = {relative_l2:.6}");
    // The reference rounds activations to Q8; Rust multiplies F32 activations.
    assert!(relative_l2 < reference["relative_l2_limit"].as_f64().unwrap());
    let top = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(top(&logits), top(&expected));
    let mut sampling = Sampling::for_model(model.config.architecture);
    sampling.temperature = 0.;
    sampling.repetition_penalty = 1.;
    let options = Options {
        max_tokens: 16,
        sampling,
        stop: vec![],
    };
    let completion = pool.install(|| {
        generation::generate(
            &model,
            &tokens,
            512,
            &options,
            &CancellationToken::new(),
            |_| Ok(()),
        )
    })?;
    let expected_ids: Vec<u32> = serde_json::from_value(reference["generated_tokens"].clone())?;
    assert_eq!(completion.token_ids, expected_ids);
    assert_eq!(completion.text, reference["text"].as_str().unwrap());
    Ok(())
}

#[test]
#[ignore = "requires the official 1.25Bit GGUF; run in release mode"]
fn official_stq() -> Result<()> {
    verify("stq")
}

#[test]
#[ignore = "requires the official 2Bit GGUF; run in release mode"]
fn official_q2c() -> Result<()> {
    verify("q2c")
}

#[test]
#[ignore = "requires the official 7B Q4 GGUF; run in release mode"]
fn official_7b() -> Result<()> {
    verify("7b")
}
