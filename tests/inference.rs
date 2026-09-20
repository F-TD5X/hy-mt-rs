mod common;

use anyhow::Result;
use hy_mt_rs::{
    generation::{self, Options, Sampling},
    model::Architecture,
};
use tokio_util::sync::CancellationToken;

#[test]
fn cached_tokens_and_chunked_prefill_match_full_prefix() -> Result<()> {
    for moe in [false, true] {
        let (_file, model) = common::model(moe, true)?;
        let tokens: Vec<u32> = (65..85).collect();
        let cancel = CancellationToken::new();
        let mut full = model.new_session(64)?;
        let expected = model.forward(&mut full, &tokens, &cancel)?;
        for chunk_size in [1, 3, 8, 13] {
            let mut session = model.new_session(64)?;
            let mut actual = Vec::new();
            for chunk in tokens.chunks(chunk_size) {
                actual = model.forward(&mut session, chunk, &cancel)?;
            }
            assert_eq!(session.position(), tokens.len());
            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    (a - e).abs() < 2e-5,
                    "moe={moe}, chunk={chunk_size}: {a} vs {e}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn sessions_do_not_share_caches_or_rng() -> Result<()> {
    let (_file, model) = common::model(false, true)?;
    let cancel = CancellationToken::new();
    let mut first = model.new_session(64)?;
    let baseline = model.forward(&mut first, &[1, 2, 3], &cancel)?;
    let mut second = model.new_session(64)?;
    model.forward(&mut second, &[100, 101, 102, 103], &cancel)?;
    let mut third = model.new_session(64)?;
    assert_eq!(baseline, model.forward(&mut third, &[1, 2, 3], &cancel)?);
    assert_eq!(first.position(), 3);
    assert_eq!(second.position(), 4);
    let (_other_file, other_model) = common::model(false, true)?;
    assert!(other_model.forward(&mut first, &[1], &cancel).is_err());
    Ok(())
}

#[test]
fn cancellation_and_context_bounds_are_checked_before_forward() -> Result<()> {
    let (_file, model) = common::model(false, false)?;
    let mut session = model.new_session(2)?;
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(model.forward(&mut session, &[1], &cancelled).is_err());
    assert_eq!(session.position(), 0);
    let token = CancellationToken::new();
    assert!(model.forward(&mut session, &[1, 2, 3], &token).is_err());
    assert!(model.forward(&mut session, &[999], &token).is_err());
    model.forward(&mut session, &[1, 2], &token)?;
    assert!(model.forward(&mut session, &[3], &token).is_err());
    Ok(())
}

#[test]
fn stop_strings_and_usage_apply_to_real_generation() -> Result<()> {
    let (_file, model) = common::model(false, false)?;
    let mut sampling = Sampling::for_model(Architecture::Dense);
    sampling.temperature = 0.;
    let options = Options {
        max_tokens: 8,
        sampling,
        stop: vec!["aaa".into()],
    };
    let mut streamed = String::new();
    let completion =
        generation::generate(&model, &[1], 64, &options, &CancellationToken::new(), |s| {
            streamed.push_str(s);
            Ok(())
        })?;
    assert_eq!(completion.text, "");
    assert_eq!(completion.text, streamed);
    assert_eq!(completion.usage.completion_tokens, 3);
    assert_eq!(completion.usage.total_tokens, 4);
    Ok(())
}
