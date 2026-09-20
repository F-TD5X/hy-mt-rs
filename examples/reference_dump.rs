//! Test helper: write raw next-token logits for comparison with the C++ oracle.
use anyhow::{Context, Result};
use hy_mt_rs::{Model, Profile};
use rayon::ThreadPoolBuilder;
use tokio_util::sync::CancellationToken;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let model_path = args
        .get(1)
        .context("usage: reference_dump MODEL PROMPT_FILE OUTPUT_PREFIX")?;
    let prompt = std::fs::read_to_string(args.get(2).context("missing prompt file")?)?;
    let output = args.get(3).context("missing output prefix")?;
    let model = Model::open(model_path, Profile::Auto)?;
    let ids = model.tokenizer.encode(&prompt)?;
    std::fs::write(format!("{output}.prompt.json"), serde_json::to_vec(&ids)?)?;
    let mut session = model.new_session((ids.len() + 64).max(512))?;
    let pool = ThreadPoolBuilder::new().num_threads(4).build()?;
    let logits = pool.install(|| model.prefill(&mut session, &ids, &CancellationToken::new()))?;
    let bytes: Vec<_> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{output}.logits.f32"), bytes)?;
    println!("{} logits", logits.len());
    Ok(())
}
