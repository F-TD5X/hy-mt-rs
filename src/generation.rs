//! Request-local sampling and incremental UTF-8/stop handling.

use std::time::Instant;

use anyhow::{Result, ensure};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::{
    Model,
    model::{Architecture, check_cancel},
};

/// Run CPU work in a shared pool while delivering text on the caller's thread.
/// Stream backpressure must not occupy a worker needed by other requests.
pub struct Generator<'a> {
    model: &'a Model,
    pool: Option<&'a rayon::ThreadPool>,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self { model, pool: None }
    }
    pub fn with_pool(model: &'a Model, pool: &'a rayon::ThreadPool) -> Self {
        Self {
            model,
            pool: Some(pool),
        }
    }

    fn compute<T: Send>(&self, operation: impl FnOnce() -> Result<T> + Send) -> Result<T> {
        match self.pool {
            Some(pool) => pool.install(operation),
            None => operation(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Sampling {
    pub fn for_model(architecture: Architecture) -> Self {
        let dense = architecture == Architecture::Dense;
        Self {
            temperature: 0.7,
            top_p: if dense { 0.6 } else { 1. },
            top_k: if dense { 20 } else { -1 },
            repetition_penalty: if dense { 1.05 } else { 1. },
            seed: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && (0.0..=2.).contains(&self.temperature),
            "temperature must be between 0 and 2"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0. && self.top_p <= 1.,
            "top_p must be in (0, 1]"
        );
        ensure!(
            self.top_k >= -1,
            "top_k must be -1, 0, or a positive integer"
        );
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty > 0.,
            "repetition_penalty must be positive and finite"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop: Vec<String>,
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        ensure!(self.max_tokens > 0, "output token limit must be positive");
        ensure!(
            self.stop.len() <= 4,
            "at most four stop strings are supported"
        );
        ensure!(
            self.stop.iter().all(|s| !s.is_empty() && s.len() <= 4096),
            "stop strings must contain 1..=4096 UTF-8 bytes"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Timing {
    pub prefill_ms: f64,
    pub first_token_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct Completion {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub timing: Timing,
    pub token_ids: Vec<u32>,
}

impl Generator<'_> {
    /// Run on a blocking thread. Only model computation enters the CPU pool.
    pub fn generate(
        &self,
        prompt: &[u32],
        context: usize,
        options: &Options,
        cancel: &CancellationToken,
        mut on_text: impl FnMut(&str) -> Result<()>,
    ) -> Result<Completion> {
        let model = self.model;
        options.validate()?;
        ensure!(
            !prompt.is_empty() && prompt.len() < context,
            "prompt leaves no output space"
        );
        ensure!(
            options.max_tokens <= context - prompt.len(),
            "prompt plus output limit exceeds context size"
        );
        let started = Instant::now();
        let mut session = model.new_session(context)?;
        let mut sampler = Sampler::new(options.sampling.clone(), model.config.vocab_size, prompt)?;
        let mut logits = self.compute(|| model.prefill(&mut session, prompt, cancel))?;
        let prefill_ms = started.elapsed().as_secs_f64() * 1000.;
        let mut first_token_ms = prefill_ms;
        let mut token_ids = Vec::new();
        let mut output = TextOutput::new(&options.stop);
        let mut finish_reason = FinishReason::Length;
        for step in 0..options.max_tokens {
            check_cancel(cancel)?;
            let token = sampler.sample(&mut logits)?;
            if step == 0 {
                first_token_ms = started.elapsed().as_secs_f64() * 1000.;
            }
            token_ids.push(token);
            if model.tokenizer.is_eos(token) {
                finish_reason = FinishReason::Stop;
                break;
            }
            let delta = output.push(model.tokenizer.piece(token)?);
            if !delta.is_empty() {
                on_text(&delta)?;
            }
            if output.stopped {
                finish_reason = FinishReason::Stop;
                break;
            }
            if step + 1 < options.max_tokens {
                logits = self.compute(|| model.forward(&mut session, &[token], cancel))?;
            }
        }
        let tail = output.finish();
        if !tail.is_empty() {
            on_text(&tail)?;
        }
        let total_ms = started.elapsed().as_secs_f64() * 1000.;
        Ok(Completion {
            text: output.text,
            finish_reason,
            usage: Usage {
                prompt_tokens: prompt.len(),
                completion_tokens: token_ids.len(),
                total_tokens: prompt.len() + token_ids.len(),
            },
            timing: Timing {
                prefill_ms,
                first_token_ms,
                decode_ms: total_ms - first_token_ms,
                total_ms,
            },
            token_ids,
        })
    }
}

/// Generate using the caller's Rayon pool, or the global pool if none is active.
pub fn generate(
    model: &Model,
    prompt: &[u32],
    context: usize,
    options: &Options,
    cancel: &CancellationToken,
    on_text: impl FnMut(&str) -> Result<()>,
) -> Result<Completion> {
    Generator::new(model).generate(prompt, context, options, cancel, on_text)
}

struct Sampler {
    params: Sampling,
    rng: StdRng,
    seen: Vec<bool>,
}

impl Sampler {
    fn new(params: Sampling, vocab: usize, prompt: &[u32]) -> Result<Self> {
        params.validate()?;
        let mut seen = vec![false; vocab];
        for &id in prompt {
            ensure!((id as usize) < vocab, "prompt token outside vocabulary");
            seen[id as usize] = true;
        }
        let rng = params
            .seed
            .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);
        Ok(Self { params, rng, seen })
    }

    fn sample(&mut self, logits: &mut [f32]) -> Result<u32> {
        ensure!(
            logits.len() == self.seen.len() && !logits.is_empty(),
            "invalid logit count"
        );
        ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
        if self.params.repetition_penalty != 1. {
            for (logit, seen) in logits.iter_mut().zip(&self.seen) {
                if *seen {
                    *logit = if *logit < 0. {
                        *logit * self.params.repetition_penalty
                    } else {
                        *logit / self.params.repetition_penalty
                    };
                }
            }
        }
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "repetition_penalty overflowed the logits"
        );
        let token = if self.params.temperature == 0. {
            logits
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                .expect("nonempty logits")
                .0
        } else {
            let mut ranked: Vec<_> = logits.iter().copied().enumerate().collect();
            ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            if self.params.top_k > 0 {
                ranked.truncate(self.params.top_k as usize);
            }
            let max = ranked[0].1;
            let mut sum = 0.;
            for (_, p) in &mut ranked {
                *p = ((*p - max) / self.params.temperature).exp();
                sum += *p;
            }
            let threshold = self.params.top_p * sum;
            let mut retained_sum = 0.;
            let mut count = 0;
            for (_, p) in &ranked {
                retained_sum += p;
                count += 1;
                if retained_sum >= threshold {
                    break;
                }
            }
            ranked.truncate(count);
            let target = self.rng.random::<f32>() * retained_sum;
            let mut cumulative = 0.;
            ranked
                .iter()
                .find(|(_, p)| {
                    cumulative += p;
                    cumulative > target
                })
                .unwrap_or_else(|| ranked.last().expect("nonempty distribution"))
                .0
        };
        self.seen[token] = true;
        Ok(token as u32)
    }
}

struct TextOutput {
    pending: Vec<u8>,
    stops: Vec<Vec<u8>>,
    text: String,
    stopped: bool,
}

impl TextOutput {
    fn new(stops: &[String]) -> Self {
        Self {
            pending: Vec::new(),
            stops: stops.iter().map(|s| s.as_bytes().to_vec()).collect(),
            text: String::new(),
            stopped: false,
        }
    }

    fn push(&mut self, bytes: &[u8]) -> String {
        if self.stopped {
            return String::new();
        }
        self.pending.extend_from_slice(bytes);
        if let Some(position) = self
            .stops
            .iter()
            .filter_map(|s| self.pending.windows(s.len()).position(|w| w == s))
            .min()
        {
            let result = self.emit(position, true);
            self.pending.clear();
            self.stopped = true;
            return result;
        }
        // Keep any suffix that could become a stop sequence on a later token.
        let hold = self
            .stops
            .iter()
            .map(|stop| {
                (1..stop.len().min(self.pending.len() + 1))
                    .rev()
                    .find(|&n| self.pending.ends_with(&stop[..n]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        self.emit(self.pending.len() - hold, false)
    }

    fn emit(&mut self, end: usize, flush: bool) -> String {
        let mut consumed = 0;
        let mut output = String::new();
        while consumed < end {
            match std::str::from_utf8(&self.pending[consumed..end]) {
                Ok(s) => {
                    output.push_str(s);
                    consumed = end;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    output.push_str(
                        std::str::from_utf8(&self.pending[consumed..consumed + valid])
                            .expect("valid UTF-8 prefix"),
                    );
                    consumed += valid;
                    if let Some(invalid) = e.error_len() {
                        output.push('\u{fffd}');
                        consumed += invalid;
                    } else if flush {
                        output.push('\u{fffd}');
                        consumed = end;
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        self.text.push_str(&output);
        output
    }

    fn finish(&mut self) -> String {
        self.emit(self.pending.len(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_and_utf8_can_cross_any_token_boundary() {
        let bytes = "你好世界STOPignored".as_bytes();
        for split in 0..=bytes.len() {
            let mut out = TextOutput::new(&["STOP".into()]);
            let mut combined = out.push(&bytes[..split]);
            combined.push_str(&out.push(&bytes[split..]));
            combined.push_str(&out.finish());
            assert_eq!(combined, "你好世界");
            assert_eq!(out.text, combined);
            assert!(out.stopped);
        }
    }

    #[test]
    fn incomplete_stop_is_emitted_at_length_limit() {
        let mut out = TextOutput::new(&["END".into()]);
        assert_eq!(out.push(b"hello E"), "hello ");
        assert_eq!(out.finish(), "E");
    }

    #[test]
    fn malformed_bytes_match_lossy_utf8() {
        let bytes = [0xf0, 0xff, b'a', 0xe4, 0xbd, 0xa0, 0xe5];
        let mut out = TextOutput::new(&[]);
        for b in bytes {
            out.push(&[b]);
        }
        out.finish();
        assert_eq!(out.text, String::from_utf8_lossy(&bytes));
    }

    #[test]
    fn seeds_are_request_local_and_top_k_one_is_greedy() -> Result<()> {
        let mut p = Sampling::for_model(Architecture::Dense);
        p.seed = Some(42);
        p.top_k = 1;
        let mut sampler = Sampler::new(p.clone(), 3, &[])?;
        assert_eq!(sampler.sample(&mut [1., 4., 2.])?, 1);
        p.top_k = -1;
        p.top_p = 1.;
        let mut a = Sampler::new(p.clone(), 3, &[])?;
        let mut b = Sampler::new(p, 3, &[])?;
        for _ in 0..20 {
            assert_eq!(a.sample(&mut [1., 2., 3.])?, b.sample(&mut [1., 2., 3.])?);
        }
        Ok(())
    }
}
