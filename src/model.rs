//! Dense Hunyuan and HYV3 MoE graphs. Weights are shared; KV state is not.

use std::{path::Path, sync::Arc};

use anyhow::{Result, bail, ensure};
use rayon::prelude::*;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

#[cfg(target_arch = "aarch64")]
use crate::quant::sdot_available;
use crate::{
    gguf::{Gguf, Profile},
    quant::{DType, Weight, dot},
    tokenizer::ChatTokenizer,
};

pub const PREFILL_CHUNK: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Architecture {
    #[serde(rename = "hunyuan-dense")]
    Dense,
    #[serde(rename = "hy_v3")]
    HyV3,
}

#[derive(Clone, Debug, Serialize)]
pub struct Config {
    pub architecture: Architecture,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context_length: usize,
    pub rope_base: f32,
    pub rms_epsilon: f32,
    pub vocab_size: usize,
}

impl Config {
    pub fn kv_bytes_per_token(&self) -> usize {
        self.layers * 2 * self.kv_heads * self.head_dim * 4
    }
}

pub struct Model {
    pub config: Config,
    pub tokenizer: ChatTokenizer,
    pub profile: Profile,
    pub file_bytes: usize,
    embedding: Weight,
    output: Weight,
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
    rope_frequencies: Vec<f32>,
    session_key: Arc<()>,
}

struct Layer {
    input_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q: Weight,
    k: Weight,
    v: Weight,
    attn_output: Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn: Ffn,
}

enum Ffn {
    Dense(Box<DenseFfn>),
    Moe(Box<Moe>),
}
struct DenseFfn {
    gate: Weight,
    up: Weight,
    down: Weight,
}
struct Moe {
    router: Weight,
    bias: Vec<f32>,
    gate: Weight,
    up: Weight,
    down: Weight,
    shared: DenseFfn,
    used: usize,
    normalize: bool,
    scale: f32,
}

/// A session belongs to one request. A failed forward invalidates the session.
pub struct Session {
    caches: Vec<KvCache>,
    position: usize,
    context: usize,
    poisoned: bool,
    model_key: Arc<()>,
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attended: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    ff: Vec<f32>,
    #[cfg(target_arch = "aarch64")]
    q8_hidden: crate::quant::Q8,
    #[cfg(target_arch = "aarch64")]
    q8_ffn: crate::quant::Q8,
}

#[derive(Default)]
struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl Session {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn context_size(&self) -> usize {
        self.context
    }
}

impl Model {
    pub fn open(path: impl AsRef<Path>, profile: Profile) -> Result<Self> {
        Self::from_gguf(Gguf::open(path, profile)?)
    }

    pub fn from_gguf(g: Gguf) -> Result<Self> {
        let prefix = g.string("general.architecture")?;
        let architecture = match prefix {
            "hunyuan-dense" => Architecture::Dense,
            "hy_v3" => Architecture::HyV3,
            other => bail!("unsupported model architecture {other}"),
        };
        let integer = |suffix: &str| g.usize(&format!("{prefix}.{suffix}"));
        let float = |suffix: &str| g.float(&format!("{prefix}.{suffix}"));
        let tokenizer = ChatTokenizer::from_gguf(&g)?;
        let config = Config {
            architecture,
            layers: integer("block_count")?,
            hidden: integer("embedding_length")?,
            heads: integer("attention.head_count")?,
            kv_heads: integer("attention.head_count_kv")?,
            head_dim: integer("attention.key_length")?,
            context_length: integer("context_length")?,
            rope_base: float("rope.freq_base")?,
            rms_epsilon: float("attention.layer_norm_rms_epsilon")?,
            vocab_size: tokenizer.vocab_size(),
        };
        ensure!(
            config.layers > 0 && config.layers <= 1024 && config.hidden > 0,
            "invalid model dimensions"
        );
        ensure!(
            config.heads > 0 && config.kv_heads > 0 && config.heads.is_multiple_of(config.kv_heads),
            "invalid grouped-query head counts"
        );
        ensure!(
            config.head_dim > 0 && config.head_dim.is_multiple_of(2),
            "invalid RoPE head dimension"
        );
        ensure!(
            integer("attention.value_length")? == config.head_dim,
            "different key/value dimensions are not supported"
        );
        ensure!(
            config.context_length > 0 && config.rope_base > 0. && config.rms_epsilon > 0.,
            "invalid context, RoPE, or RMS settings"
        );
        if let Some(value) = g.metadata.get(&format!("{prefix}.rope.scaling.type")) {
            ensure!(
                value.as_str() == Some("none"),
                "GGUF must contain an effective RoPE base with no extra scaling"
            );
        }
        if g.metadata
            .contains_key(&format!("{prefix}.rope.dimension_count"))
        {
            ensure!(
                integer("rope.dimension_count")? == config.head_dim,
                "partial rotary embeddings are not supported"
            );
        }
        let h = config.hidden;
        let embedding = matrix(&g, "token_embd.weight", &[h, config.vocab_size])?;
        let output = if g.tensors.contains_key("output.weight") {
            matrix(&g, "output.weight", &[h, config.vocab_size])?
        } else {
            ensure!(
                architecture == Architecture::Dense,
                "HYV3 requires separate output weights"
            );
            embedding.clone()
        };
        let output_norm = norm(&g, "output_norm.weight", h)?;
        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let p = format!("blk.{i}.");
            let weight =
                |name: &str, dims: &[usize]| matrix(&g, &format!("{p}{name}.weight"), dims);
            let layer_norm = |name: &str, dim| norm(&g, &format!("{p}{name}.weight"), dim);
            let ffn = if architecture == Architecture::HyV3 && i > 0 {
                let experts = integer("expert_count")?;
                let used = integer("expert_used_count")?;
                let mid = integer("expert_feed_forward_length")?;
                let shared = integer("expert_shared_feed_forward_length")?;
                ensure!(
                    experts > 0 && used > 0 && used <= experts && mid > 0 && shared > 0,
                    "invalid expert dimensions"
                );
                ensure!(
                    integer("expert_gating_func")? == 2,
                    "HYV3 requires a sigmoid router"
                );
                let router = weight("ffn_gate_inp", &[h, experts])?;
                ensure!(
                    router.dtype() == crate::quant::DType::F32,
                    "HYV3 router must use F32"
                );
                let bias = norm(&g, &format!("{p}exp_probs_b"), experts)?;
                let scale = float("expert_weights_scale")?;
                ensure!(scale > 0., "invalid expert weight scale");
                Ffn::Moe(Box::new(Moe {
                    router,
                    bias,
                    gate: weight("ffn_gate_exps", &[h, mid, experts])?,
                    up: weight("ffn_up_exps", &[h, mid, experts])?,
                    down: weight("ffn_down_exps", &[mid, h, experts])?,
                    shared: DenseFfn {
                        gate: weight("ffn_gate_shexp", &[h, shared])?,
                        up: weight("ffn_up_shexp", &[h, shared])?,
                        down: weight("ffn_down_shexp", &[shared, h])?,
                    },
                    used,
                    normalize: g.boolean(&format!("{prefix}.expert_weights_norm"))?,
                    scale,
                }))
            } else {
                let mid = integer("feed_forward_length")?;
                Ffn::Dense(Box::new(DenseFfn {
                    gate: weight("ffn_gate", &[h, mid])?,
                    up: weight("ffn_up", &[h, mid])?,
                    down: weight("ffn_down", &[mid, h])?,
                }))
            };
            layers.push(Layer {
                input_norm: layer_norm("attn_norm", h)?,
                ffn_norm: layer_norm("ffn_norm", h)?,
                q: weight("attn_q", &[h, config.heads * config.head_dim])?,
                k: weight("attn_k", &[h, config.kv_heads * config.head_dim])?,
                v: weight("attn_v", &[h, config.kv_heads * config.head_dim])?,
                attn_output: weight("attn_output", &[config.heads * config.head_dim, h])?,
                q_norm: layer_norm("attn_q_norm", config.head_dim)?,
                k_norm: layer_norm("attn_k_norm", config.head_dim)?,
                ffn,
            });
        }
        let rope_frequencies = (0..config.head_dim / 2)
            .map(|j| {
                config
                    .rope_base
                    .powf(-2. * j as f32 / config.head_dim as f32)
            })
            .collect();
        Ok(Self {
            config,
            tokenizer,
            profile: g.profile,
            file_bytes: g.byte_len(),
            embedding,
            output,
            output_norm,
            layers,
            rope_frequencies,
            session_key: Arc::new(()),
        })
    }

    pub fn new_session(&self, context: usize) -> Result<Session> {
        ensure!(
            context > 0 && context <= self.config.context_length,
            "context must be between 1 and {}",
            self.config.context_length
        );
        let h = self.config.hidden;
        let q_len = self.config.heads * self.config.head_dim;
        let kv_len = self.config.kv_heads * self.config.head_dim;
        let ffn_len = match &self.layers[0].ffn {
            Ffn::Dense(ff) => ff.gate.shape()[1],
            Ffn::Moe(ff) => ff.shared.gate.shape()[1],
        };
        Ok(Session {
            caches: (0..self.config.layers)
                .map(|_| KvCache::default())
                .collect(),
            position: 0,
            context,
            poisoned: false,
            model_key: self.session_key.clone(),
            hidden: vec![0.; h],
            normalized: vec![0.; h],
            q: vec![0.; q_len],
            k: vec![0.; kv_len],
            v: vec![0.; kv_len],
            attended: vec![0.; q_len],
            projected: vec![0.; h],
            gate: vec![0.; ffn_len],
            ff: vec![0.; h],
            #[cfg(target_arch = "aarch64")]
            q8_hidden: crate::quant::Q8::new(h),
            #[cfg(target_arch = "aarch64")]
            q8_ffn: crate::quant::Q8::new(ffn_len),
        })
    }

    /// Returns next-token logits after consuming these tokens into the cache.
    pub fn forward(
        &self,
        session: &mut Session,
        tokens: &[u32],
        cancel: &CancellationToken,
    ) -> Result<Vec<f32>> {
        let mut logits = vec![0.; self.config.vocab_size];
        self.forward_into(session, tokens, cancel, &mut logits)?;
        Ok(logits)
    }

    /// Computes next-token logits directly into the provided output buffer.
    pub fn forward_into(
        &self,
        session: &mut Session,
        tokens: &[u32],
        cancel: &CancellationToken,
        out: &mut [f32],
    ) -> Result<()> {
        ensure!(
            out.len() == self.config.vocab_size,
            "output buffer length mismatch"
        );
        ensure!(
            !session.poisoned,
            "session was invalidated by a failed forward"
        );
        ensure!(
            !tokens.is_empty() && tokens.len() <= PREFILL_CHUNK,
            "forward expects 1..={PREFILL_CHUNK} tokens"
        );
        ensure!(
            session.position + tokens.len() <= session.context,
            "context limit exceeded"
        );
        ensure!(
            Arc::ptr_eq(&session.model_key, &self.session_key),
            "session belongs to a different model"
        );
        ensure!(
            tokens
                .iter()
                .all(|&t| (t as usize) < self.config.vocab_size),
            "input token outside vocabulary"
        );
        check_cancel(cancel)?;
        session.poisoned = true;
        let batch = tokens.len();

        if batch == 1 {
            let id = tokens[0];
            self.embedding.row_into(id as usize, &mut session.hidden)?;

            let pos = session.position as f32;
            let half_dim = self.config.head_dim / 2;
            let mut cos_buf = [0f32; 64];
            let mut sin_buf = [0f32; 64];
            for (j, &freq) in self
                .rope_frequencies
                .iter()
                .enumerate()
                .take(half_dim.min(64))
            {
                let (s, c) = (pos * freq).sin_cos();
                sin_buf[j] = s;
                cos_buf[j] = c;
            }
            let (cos_s, sin_s) = (&cos_buf[..half_dim.min(64)], &sin_buf[..half_dim.min(64)]);

            for (layer, cache) in self.layers.iter().zip(&mut session.caches) {
                check_cancel(cancel)?;
                rms_norm_into(
                    &session.hidden,
                    &layer.input_norm,
                    self.config.rms_epsilon,
                    &mut session.normalized,
                );

                #[cfg(target_arch = "aarch64")]
                if sdot_available()
                    && matches!(layer.q.dtype(), DType::STQ1_0 | DType::Q6_K | DType::Q2_0C)
                {
                    session.q8_hidden.quantize_into(&session.normalized);
                    Weight::gemv_qkv_fast(
                        &layer.q,
                        &layer.k,
                        &layer.v,
                        &session.q8_hidden,
                        &mut session.q,
                        &mut session.k,
                        &mut session.v,
                    )?;
                } else {
                    let (q, k, v) =
                        Weight::gemv_qkv(&layer.q, &layer.k, &layer.v, &session.normalized)?;
                    session.q.copy_from_slice(&q);
                    session.k.copy_from_slice(&k);
                    session.v.copy_from_slice(&v);
                }

                if self.config.architecture == Architecture::HyV3 {
                    rms_norm(&mut session.q, &layer.q_norm, self.config.rms_epsilon);
                    rms_norm(&mut session.k, &layer.k_norm, self.config.rms_epsilon);
                }
                rope_with_cache(
                    &mut session.q,
                    self.config.heads,
                    self.config.head_dim,
                    cos_s,
                    sin_s,
                );
                rope_with_cache(
                    &mut session.k,
                    self.config.kv_heads,
                    self.config.head_dim,
                    cos_s,
                    sin_s,
                );
                if self.config.architecture == Architecture::Dense {
                    rms_norm(&mut session.q, &layer.q_norm, self.config.rms_epsilon);
                    rms_norm(&mut session.k, &layer.k_norm, self.config.rms_epsilon);
                }
                cache.append(
                    &session.k,
                    &session.v,
                    self.config.kv_heads * self.config.head_dim,
                    session.context,
                )?;
                attention_into(
                    &session.q,
                    cache,
                    &self.config,
                    session.position,
                    &mut session.attended,
                );

                #[cfg(target_arch = "aarch64")]
                if sdot_available()
                    && matches!(
                        layer.attn_output.dtype(),
                        DType::STQ1_0 | DType::Q6_K | DType::Q2_0C
                    )
                {
                    session.q8_hidden.quantize_into(&session.attended);
                    layer
                        .attn_output
                        .gemv_q8_fast(&session.q8_hidden, &mut session.projected);
                } else {
                    let projected = layer.attn_output.matmul(&session.attended, 1)?;
                    session.projected.copy_from_slice(&projected);
                }
                add(&mut session.hidden, &session.projected);

                rms_norm_into(
                    &session.hidden,
                    &layer.ffn_norm,
                    self.config.rms_epsilon,
                    &mut session.normalized,
                );

                match &layer.ffn {
                    Ffn::Dense(ff) => {
                        #[cfg(target_arch = "aarch64")]
                        if sdot_available()
                            && matches!(ff.gate.dtype(), DType::STQ1_0 | DType::Q6_K | DType::Q2_0C)
                            && ff.gate.dtype() == ff.up.dtype()
                            && matches!(ff.down.dtype(), DType::STQ1_0 | DType::Q6_K | DType::Q2_0C)
                        {
                            session.q8_hidden.quantize_into(&session.normalized);
                            ff.gate.matmul_gate_up_silu_fast(
                                &ff.up,
                                &session.q8_hidden,
                                &mut session.gate,
                            );

                            session.q8_ffn.quantize_into(&session.gate);
                            ff.down.gemv_q8_fast(&session.q8_ffn, &mut session.ff);
                        } else {
                            let ff_out = ff.forward(&session.normalized, 1)?;
                            session.ff.copy_from_slice(&ff_out);
                        }
                    }
                    Ffn::Moe(ff) => {
                        let ff_out = ff.forward(&session.normalized, 1, cancel)?;
                        session.ff.copy_from_slice(&ff_out);
                    }
                }
                add(&mut session.hidden, &session.ff);
            }

            check_cancel(cancel)?;
            rms_norm(
                &mut session.hidden,
                &self.output_norm,
                self.config.rms_epsilon,
            );
            #[cfg(target_arch = "aarch64")]
            if sdot_available()
                && matches!(
                    self.output.dtype(),
                    DType::STQ1_0 | DType::Q6_K | DType::Q2_0C
                )
            {
                session.q8_hidden.quantize_into(&session.hidden);
                self.output.gemv_q8_fast(&session.q8_hidden, out);
            } else {
                let logits = self.output.matmul(&session.hidden, 1)?;
                out.copy_from_slice(&logits);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let logits = self.output.matmul(&session.hidden, 1)?;
                out.copy_from_slice(&logits);
            }

            ensure!(
                out.iter().all(|x| x.is_finite()),
                "model produced non-finite logits; check model format and weights"
            );
            session.position += 1;
            session.poisoned = false;
            return Ok(());
        }

        let mut hidden = Vec::with_capacity(batch * self.config.hidden);
        for &id in tokens {
            hidden.extend(self.embedding.row(id as usize)?);
        }
        for (layer, cache) in self.layers.iter().zip(&mut session.caches) {
            check_cancel(cancel)?;
            let mut normalized = hidden.clone();
            rms_norm(&mut normalized, &layer.input_norm, self.config.rms_epsilon);
            let (mut q, mut k, v) = if batch == 1 {
                Weight::gemv_qkv(&layer.q, &layer.k, &layer.v, &normalized)?
            } else {
                (
                    layer.q.matmul(&normalized, batch)?,
                    layer.k.matmul(&normalized, batch)?,
                    layer.v.matmul(&normalized, batch)?,
                )
            };
            if self.config.architecture == Architecture::HyV3 {
                rms_norm(&mut q, &layer.q_norm, self.config.rms_epsilon);
                rms_norm(&mut k, &layer.k_norm, self.config.rms_epsilon);
            }
            rope(
                &mut q,
                self.config.heads,
                self.config.head_dim,
                session.position,
                &self.rope_frequencies,
            );
            rope(
                &mut k,
                self.config.kv_heads,
                self.config.head_dim,
                session.position,
                &self.rope_frequencies,
            );
            if self.config.architecture == Architecture::Dense {
                rms_norm(&mut q, &layer.q_norm, self.config.rms_epsilon);
                rms_norm(&mut k, &layer.k_norm, self.config.rms_epsilon);
            }
            cache.append(
                &k,
                &v,
                self.config.kv_heads * self.config.head_dim,
                session.context,
            )?;
            let attended = attention(&q, cache, &self.config, session.position);
            let projected = layer.attn_output.matmul(&attended, batch)?;
            add(&mut hidden, &projected);
            normalized.clone_from(&hidden);
            rms_norm(&mut normalized, &layer.ffn_norm, self.config.rms_epsilon);
            let ff = match &layer.ffn {
                Ffn::Dense(ff) => ff.forward(&normalized, batch)?,
                Ffn::Moe(ff) => ff.forward(&normalized, batch, cancel)?,
            };
            add(&mut hidden, &ff);
        }
        check_cancel(cancel)?;
        let last = &mut hidden[(batch - 1) * self.config.hidden..];
        rms_norm(last, &self.output_norm, self.config.rms_epsilon);
        let logits = self.output.matmul(last, 1)?;
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "model produced non-finite logits; check model format and weights"
        );
        out.copy_from_slice(&logits);
        session.position += batch;
        session.poisoned = false;
        Ok(())
    }

    pub fn prefill(
        &self,
        session: &mut Session,
        tokens: &[u32],
        cancel: &CancellationToken,
    ) -> Result<Vec<f32>> {
        let mut logits = vec![0.; self.config.vocab_size];
        self.prefill_into(session, tokens, &mut logits, cancel)?;
        Ok(logits)
    }

    pub fn prefill_into(
        &self,
        session: &mut Session,
        tokens: &[u32],
        out: &mut [f32],
        cancel: &CancellationToken,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "prompt must contain at least one token");
        ensure!(
            tokens.len() <= session.context - session.position,
            "prompt exceeds context limit"
        );
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            self.forward_into(session, chunk, cancel, out)?;
        }
        Ok(())
    }
}

impl KvCache {
    fn append(&mut self, keys: &[f32], values: &[f32], width: usize, context: usize) -> Result<()> {
        let next = self.keys.len() + keys.len();
        let reserve_tokens = (next / width)
            .div_ceil(PREFILL_CHUNK)
            .saturating_mul(PREFILL_CHUNK)
            .min(context);
        let capacity = reserve_tokens
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("KV cache size overflow"))?;
        if capacity > self.keys.capacity() {
            self.keys.try_reserve_exact(capacity - self.keys.len())?;
        }
        if capacity > self.values.capacity() {
            self.values
                .try_reserve_exact(capacity - self.values.len())?;
        }
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        Ok(())
    }
}

impl DenseFfn {
    fn forward(&self, input: &[f32], batch: usize) -> Result<Vec<f32>> {
        if batch == 1 {
            let gate = self.gate.matmul_gate_up_silu(&self.up, input)?;
            return self.down.matmul(&gate, 1);
        }
        let mut gate = self.gate.matmul(input, batch)?;
        let up = self.up.matmul(input, batch)?;
        for (g, u) in gate.iter_mut().zip(up) {
            *g = (*g / (1. + (-*g).exp())) * u;
        }
        self.down.matmul(&gate, batch)
    }
}

impl Moe {
    fn forward(&self, input: &[f32], batch: usize, cancel: &CancellationToken) -> Result<Vec<f32>> {
        let hidden = self.router.shape()[0];
        let experts = self.router.shape()[1];
        let logits = self.router.matmul(input, batch)?;
        let mut assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts];
        for (token, scores) in logits.chunks_exact(experts).enumerate() {
            for (expert, weight) in route(scores, &self.bias, self.used, self.normalize, self.scale)
            {
                assignments[expert].push((token, weight));
            }
        }
        let mut output = self.shared.forward(input, batch)?;
        for (expert, tokens) in assignments.into_iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            check_cancel(cancel)?;
            let mut selected = Vec::with_capacity(tokens.len() * hidden);
            for &(token, _) in &tokens {
                selected.extend_from_slice(&input[token * hidden..(token + 1) * hidden]);
            }
            let ff = DenseFfn {
                gate: self.gate.expert(expert)?,
                up: self.up.expert(expert)?,
                down: self.down.expert(expert)?,
            };
            let result = ff.forward(&selected, tokens.len())?;
            for ((token, weight), row) in tokens.into_iter().zip(result.chunks_exact(hidden)) {
                for (out, value) in output[token * hidden..(token + 1) * hidden]
                    .iter_mut()
                    .zip(row)
                {
                    *out += weight * value;
                }
            }
        }
        Ok(output)
    }
}

fn route(
    logits: &[f32],
    bias: &[f32],
    used: usize,
    normalize: bool,
    scale: f32,
) -> Vec<(usize, f32)> {
    let sigmoid: Vec<_> = logits
        .iter()
        .map(|&x| {
            if x >= 0. {
                1. / (1. + (-x).exp())
            } else {
                let e = x.exp();
                e / (1. + e)
            }
        })
        .collect();
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        (sigmoid[b] + bias[b])
            .total_cmp(&(sigmoid[a] + bias[a]))
            .then_with(|| a.cmp(&b))
    });
    indices.truncate(used);
    // Match the GGUF reference graph's minimum normal F16 denominator.
    let sum = if normalize {
        indices
            .iter()
            .map(|&i| sigmoid[i])
            .sum::<f32>()
            .max(half::f16::MIN_POSITIVE.to_f32())
    } else {
        1.
    };
    indices
        .into_iter()
        .map(|i| (i, sigmoid[i] / sum * scale))
        .collect()
}

fn matrix(g: &Gguf, name: &str, shape: &[usize]) -> Result<Weight> {
    let t = g.tensor(name)?;
    t.expect_shape(shape)?;
    Ok(t)
}

fn norm(g: &Gguf, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = matrix(g, name, &[len])?;
    let values = t.to_vec()?;
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "non-finite normalization/router tensor {name}"
    );
    Ok(values)
}

fn rms_norm_into(input: &[f32], weights: &[f32], epsilon: f32, out: &mut [f32]) {
    assert_eq!(input.len(), weights.len());
    assert_eq!(out.len(), weights.len());
    let w_len = weights.len();
    let variance = dot(input, input) / w_len as f32;
    let scale = (variance + epsilon).sqrt().recip();
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let scale_vec = unsafe { vdupq_n_f32(scale) };
        let chunks = w_len / 16;
        let mut ip = input.as_ptr();
        let mut wp = weights.as_ptr();
        let mut op = out.as_mut_ptr();
        unsafe {
            for _ in 0..chunks {
                let i0 = vld1q_f32(ip);
                let i1 = vld1q_f32(ip.add(4));
                let i2 = vld1q_f32(ip.add(8));
                let i3 = vld1q_f32(ip.add(12));
                let w0 = vld1q_f32(wp);
                let w1 = vld1q_f32(wp.add(4));
                let w2 = vld1q_f32(wp.add(8));
                let w3 = vld1q_f32(wp.add(12));
                vst1q_f32(op, vmulq_f32(i0, vmulq_f32(w0, scale_vec)));
                vst1q_f32(op.add(4), vmulq_f32(i1, vmulq_f32(w1, scale_vec)));
                vst1q_f32(op.add(8), vmulq_f32(i2, vmulq_f32(w2, scale_vec)));
                vst1q_f32(op.add(12), vmulq_f32(i3, vmulq_f32(w3, scale_vec)));
                ip = ip.add(16);
                wp = wp.add(16);
                op = op.add(16);
            }
        }
        for ((x, &weight), out_val) in input[chunks * 16..]
            .iter()
            .zip(&weights[chunks * 16..])
            .zip(&mut out[chunks * 16..])
        {
            *out_val = *x * scale * weight;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for ((x, &weight), out_val) in input.iter().zip(weights).zip(out.iter_mut()) {
        *out_val = *x * scale * weight;
    }
}

fn rms_norm(values: &mut [f32], weights: &[f32], epsilon: f32) {
    let w_len = weights.len();
    for row in values.chunks_exact_mut(w_len) {
        let variance = dot(row, row) / row.len() as f32;
        let scale = (variance + epsilon).sqrt().recip();
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let scale_vec = unsafe { vdupq_n_f32(scale) };
            let chunks = w_len / 16;
            let mut rp = row.as_mut_ptr();
            let mut wp = weights.as_ptr();
            unsafe {
                for _ in 0..chunks {
                    let r0 = vld1q_f32(rp);
                    let r1 = vld1q_f32(rp.add(4));
                    let r2 = vld1q_f32(rp.add(8));
                    let r3 = vld1q_f32(rp.add(12));
                    let w0 = vld1q_f32(wp);
                    let w1 = vld1q_f32(wp.add(4));
                    let w2 = vld1q_f32(wp.add(8));
                    let w3 = vld1q_f32(wp.add(12));
                    vst1q_f32(rp, vmulq_f32(r0, vmulq_f32(w0, scale_vec)));
                    vst1q_f32(rp.add(4), vmulq_f32(r1, vmulq_f32(w1, scale_vec)));
                    vst1q_f32(rp.add(8), vmulq_f32(r2, vmulq_f32(w2, scale_vec)));
                    vst1q_f32(rp.add(12), vmulq_f32(r3, vmulq_f32(w3, scale_vec)));
                    rp = rp.add(16);
                    wp = wp.add(16);
                }
            }
            for (x, &weight) in row[chunks * 16..].iter_mut().zip(&weights[chunks * 16..]) {
                *x = *x * scale * weight;
            }
            continue;
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (x, &weight) in row.iter_mut().zip(weights) {
            *x = *x * scale * weight;
        }
    }
}

fn rope_with_cache(values: &mut [f32], _heads: usize, dim: usize, cos_s: &[f32], sin_s: &[f32]) {
    let half_dim = dim / 2;
    for head in values.chunks_exact_mut(dim) {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let chunks = half_dim / 4;
            let mut ap = head.as_mut_ptr();
            let mut bp = unsafe { ap.add(half_dim) };
            let mut cp = cos_s.as_ptr();
            let mut sp = sin_s.as_ptr();
            unsafe {
                for _ in 0..chunks {
                    let a = vld1q_f32(ap);
                    let b = vld1q_f32(bp);
                    let c = vld1q_f32(cp);
                    let s = vld1q_f32(sp);
                    let ac = vmulq_f32(a, c);
                    let bs = vmulq_f32(b, s);
                    let as_val = vmulq_f32(a, s);
                    let bc = vmulq_f32(b, c);
                    vst1q_f32(ap, vsubq_f32(ac, bs));
                    vst1q_f32(bp, vaddq_f32(as_val, bc));
                    ap = ap.add(4);
                    bp = bp.add(4);
                    cp = cp.add(4);
                    sp = sp.add(4);
                }
            }
            for j in (chunks * 4)..half_dim {
                let (a, b) = (head[j], head[j + half_dim]);
                head[j] = a * cos_s[j] - b * sin_s[j];
                head[j + half_dim] = a * sin_s[j] + b * cos_s[j];
            }
            continue;
        }
        #[cfg(not(target_arch = "aarch64"))]
        for j in 0..half_dim {
            let (a, b) = (head[j], head[j + half_dim]);
            head[j] = a * cos_s[j] - b * sin_s[j];
            head[j + half_dim] = a * sin_s[j] + b * cos_s[j];
        }
    }
}

fn rope(values: &mut [f32], heads: usize, dim: usize, start: usize, frequencies: &[f32]) {
    let half_dim = dim / 2;
    for (token, row) in values.chunks_exact_mut(heads * dim).enumerate() {
        let pos = (start + token) as f32;
        let mut cos_buf = [0f32; 64];
        let mut sin_buf = [0f32; 64];
        let mut cos_vec;
        let mut sin_vec;
        let (cos_s, sin_s): (&[f32], &[f32]) = if half_dim <= 64 {
            for (j, &freq) in frequencies.iter().enumerate() {
                let (s, c) = (pos * freq).sin_cos();
                sin_buf[j] = s;
                cos_buf[j] = c;
            }
            (&cos_buf[..half_dim], &sin_buf[..half_dim])
        } else {
            cos_vec = Vec::with_capacity(half_dim);
            sin_vec = Vec::with_capacity(half_dim);
            for &freq in frequencies {
                let (s, c) = (pos * freq).sin_cos();
                sin_vec.push(s);
                cos_vec.push(c);
            }
            (&cos_vec[..], &sin_vec[..])
        };

        rope_with_cache(row, heads, dim, cos_s, sin_s);
    }
}

fn attention_into(q: &[f32], cache: &KvCache, config: &Config, past: usize, out: &mut [f32]) {
    let dim = config.head_dim;
    let kv_width = config.kv_heads * dim;
    let repeats = config.heads / config.kv_heads;
    let scale = (dim as f32).sqrt().recip();
    out.fill(0.);

    let compute_head = |index: usize, output: &mut [f32]| {
        let token = index / config.heads;
        let kv_head = index % config.heads / repeats;
        let query = &q[index * dim..(index + 1) * dim];
        let visible = past + token + 1;
        let mut stack_scores = [0f32; 1024];
        let mut heap_scores;
        let scores: &mut [f32] = if visible <= 1024 {
            &mut stack_scores[..visible]
        } else {
            heap_scores = vec![0f32; visible];
            &mut heap_scores[..]
        };
        for (pos, score) in scores.iter_mut().enumerate() {
            let offset = pos * kv_width + kv_head * dim;
            *score = dot(query, &cache.keys[offset..offset + dim]) * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.;
        for score in scores.iter_mut() {
            *score = (*score - max).exp();
            sum += *score;
        }
        let inv_sum = sum.recip();
        for (pos, &score) in scores.iter().enumerate() {
            let offset = pos * kv_width + kv_head * dim;
            let p = score * inv_sum;
            let val_slice = &cache.values[offset..offset + dim];
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let pv = unsafe { vdupq_n_f32(p) };
                let chunks = dim / 16;
                let mut op = output.as_mut_ptr();
                let mut vp = val_slice.as_ptr();
                unsafe {
                    for _ in 0..chunks {
                        let o0 = vld1q_f32(op);
                        let o1 = vld1q_f32(op.add(4));
                        let o2 = vld1q_f32(op.add(8));
                        let o3 = vld1q_f32(op.add(12));
                        let v0 = vld1q_f32(vp);
                        let v1 = vld1q_f32(vp.add(4));
                        let v2 = vld1q_f32(vp.add(8));
                        let v3 = vld1q_f32(vp.add(12));
                        vst1q_f32(op, vfmaq_f32(o0, v0, pv));
                        vst1q_f32(op.add(4), vfmaq_f32(o1, v1, pv));
                        vst1q_f32(op.add(8), vfmaq_f32(o2, v2, pv));
                        vst1q_f32(op.add(12), vfmaq_f32(o3, v3, pv));
                        op = op.add(16);
                        vp = vp.add(16);
                    }
                }
                for (out_val, &val) in output[chunks * 16..]
                    .iter_mut()
                    .zip(&val_slice[chunks * 16..])
                {
                    *out_val += p * val;
                }
                continue;
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (out_val, &val) in output.iter_mut().zip(val_slice) {
                *out_val += p * val;
            }
        }
    };

    if q.len() == config.heads * dim && past < 64 {
        for (index, output) in out.chunks_mut(dim).enumerate() {
            compute_head(index, output);
        }
    } else {
        out.par_chunks_mut(dim)
            .enumerate()
            .for_each(|(index, output)| {
                compute_head(index, output);
            });
    }
}

fn attention(q: &[f32], cache: &KvCache, config: &Config, past: usize) -> Vec<f32> {
    let mut out = vec![0.; q.len()];
    attention_into(q, cache, config, past, &mut out);
    out
}

fn add(dst: &mut [f32], src: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let len = dst.len();
        let chunks = len / 16;
        let mut dp = dst.as_mut_ptr();
        let mut sp = src.as_ptr();
        unsafe {
            for _ in 0..chunks {
                let d0 = vld1q_f32(dp);
                let d1 = vld1q_f32(dp.add(4));
                let d2 = vld1q_f32(dp.add(8));
                let d3 = vld1q_f32(dp.add(12));
                let s0 = vld1q_f32(sp);
                let s1 = vld1q_f32(sp.add(4));
                let s2 = vld1q_f32(sp.add(8));
                let s3 = vld1q_f32(sp.add(12));
                vst1q_f32(dp, vaddq_f32(d0, s0));
                vst1q_f32(dp.add(4), vaddq_f32(d1, s1));
                vst1q_f32(dp.add(8), vaddq_f32(d2, s2));
                vst1q_f32(dp.add(12), vaddq_f32(d3, s3));
                dp = dp.add(16);
                sp = sp.add(16);
            }
        }
        for (x, y) in dst[chunks * 16..].iter_mut().zip(&src[chunks * 16..]) {
            *x += y;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (x, y) in dst.iter_mut().zip(src) {
        *x += y;
    }
}

pub(crate) fn check_cancel(cancel: &CancellationToken) -> Result<()> {
    ensure!(!cancel.is_cancelled(), "request cancelled");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_bias_changes_selection_not_weights() {
        let r = route(&[0., 2., -2.], &[0., 0., 10.], 2, true, 2.826);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 1]);
        assert!((r.iter().map(|x| x.1).sum::<f32>() - 2.826).abs() < 1e-6);
        assert!(r[0].1 < r[1].1);
    }

    #[test]
    fn rotary_pairs_halves_not_neighbors() {
        let mut x = [1., 0., 0., 0.];
        rope(&mut x, 1, 4, 1, &[std::f32::consts::FRAC_PI_2, 1.]);
        assert!(x[0].abs() < 1e-6);
        assert_eq!(x[1], 0.);
        assert!((x[2] - 1.).abs() < 1e-6);
    }

    #[test]
    fn causal_attention_does_not_see_future_values() {
        let cfg = Config {
            architecture: Architecture::Dense,
            layers: 1,
            hidden: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            context_length: 8,
            rope_base: 10000.,
            rms_epsilon: 1e-5,
            vocab_size: 2,
        };
        let cache = KvCache {
            keys: vec![0.; 4],
            values: vec![2., 4., 100., 200.],
        };
        let out = attention(&[0.; 8], &cache, &cfg, 0);
        assert_eq!(&out[..4], &[2., 4., 2., 4.]);
        assert_eq!(&out[4..], &[51., 102., 51., 102.]);
    }
}
