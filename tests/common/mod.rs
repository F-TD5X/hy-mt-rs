//! Small, executable GGUF models for HTTP and cache tests. No model download.
#![allow(dead_code)]

use anyhow::Result;
use hy_mt_rs::{Model, Profile};
use std::{io::Write, sync::Arc};
use tempfile::NamedTempFile;

enum Value {
    U32(u32),
    Float(f32),
    Bool(bool),
    Text(String),
    Texts(Vec<String>),
    Ints(Vec<i32>),
}

pub fn model(moe: bool, numeric: bool) -> Result<(NamedTempFile, Arc<Model>)> {
    let mut file = NamedTempFile::new()?;
    file.write_all(&fixture(moe, numeric))?;
    let model = Arc::new(Model::open(file.path(), Profile::Auto)?);
    Ok((file, model))
}

pub fn fixture(moe: bool, numeric: bool) -> Vec<u8> {
    let arch = if moe { "hy_v3" } else { "hunyuan-dense" };
    let heads = if moe { 4 } else { 2 };
    let mut meta = vec![("general.architecture".into(), Value::Text(arch.into()))];
    for (key, value) in [
        ("block_count", if moe { 2 } else { 1 }),
        ("context_length", 4096),
        ("embedding_length", 4),
        ("feed_forward_length", 8),
        ("attention.head_count", heads),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 2),
        ("attention.value_length", 2),
    ] {
        meta.push((format!("{arch}.{key}"), Value::U32(value)));
    }
    meta.push((format!("{arch}.rope.freq_base"), Value::Float(10000.)));
    meta.push((
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        Value::Float(1e-5),
    ));
    if moe {
        for (key, value) in [
            ("expert_count", 3),
            ("expert_used_count", 2),
            ("expert_feed_forward_length", 4),
            ("expert_shared_feed_forward_length", 4),
            ("expert_gating_func", 2),
        ] {
            meta.push((format!("{arch}.{key}"), Value::U32(value)));
        }
        meta.push((format!("{arch}.expert_weights_norm"), Value::Bool(true)));
        meta.push((format!("{arch}.expert_weights_scale"), Value::Float(2.826)));
    }
    let mut vocab = Vec::new();
    let mut extra = 0;
    for byte in 0..256u32 {
        let ch = if (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte)
        {
            byte
        } else {
            let n = 256 + extra;
            extra += 1;
            n
        };
        vocab.push(char::from_u32(ch).unwrap().to_string());
    }
    vocab.extend(["[BOS]", "[USER]", "[ASSISTANT]", "[EOS]"].map(str::to_owned));
    let mut types = vec![1; 256];
    types.extend([3; 4]);
    meta.extend([
        ("tokenizer.ggml.model".into(), Value::Text("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Value::Text("hunyuan-dense".into())),
        ("tokenizer.ggml.tokens".into(), Value::Texts(vocab)),
        ("tokenizer.ggml.merges".into(), Value::Texts(vec![])),
        ("tokenizer.ggml.token_type".into(), Value::Ints(types)),
        ("tokenizer.ggml.bos_token_id".into(), Value::U32(256)),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(259)),
        ("tokenizer.chat_template".into(), Value::Text("{{ bos_token }}{% for m in messages %}[USER]{{ m.content }}{% endfor %}[ASSISTANT]".into())),
    ]);
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut add = |name: String, shape: &[usize], norm: bool, embedding: bool, output: bool| {
        let n = shape.iter().product();
        let values = (0..n)
            .map(|i| {
                if norm {
                    if numeric {
                        0.8 + (i % 3) as f32 * 0.1
                    } else {
                        1.
                    }
                } else if numeric {
                    ((i * 17 + name.bytes().map(usize::from).sum::<usize>()) % 31) as f32 / 50.
                        - 0.3
                } else if embedding || (output && i / 4 == b'a' as usize) {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        tensors.push((name, shape.to_vec(), values));
    };
    add("token_embd.weight".into(), &[4, 260], false, true, false);
    add("output.weight".into(), &[4, 260], false, false, true);
    add("output_norm.weight".into(), &[4], true, false, false);
    for i in 0..if moe { 2 } else { 1 } {
        for name in ["attn_norm", "ffn_norm"] {
            add(format!("blk.{i}.{name}.weight"), &[4], true, false, false);
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            add(format!("blk.{i}.{name}.weight"), &[2], true, false, false);
        }
        for (name, shape) in [
            ("attn_q", [4, heads as usize * 2]),
            ("attn_k", [4, 2]),
            ("attn_v", [4, 2]),
            ("attn_output", [heads as usize * 2, 4]),
        ] {
            add(
                format!("blk.{i}.{name}.weight"),
                &shape,
                false,
                false,
                false,
            );
        }
        if moe && i > 0 {
            add(
                format!("blk.{i}.ffn_gate_inp.weight"),
                &[4, 3],
                false,
                false,
                false,
            );
            add(format!("blk.{i}.exp_probs_b"), &[3], false, false, false);
            for name in ["gate", "up", "down"] {
                add(
                    format!("blk.{i}.ffn_{name}_exps.weight"),
                    &[4, 4, 3],
                    false,
                    false,
                    false,
                );
                add(
                    format!("blk.{i}.ffn_{name}_shexp.weight"),
                    &[4, 4],
                    false,
                    false,
                    false,
                );
            }
        } else {
            add(
                format!("blk.{i}.ffn_gate.weight"),
                &[4, 8],
                false,
                false,
                false,
            );
            add(
                format!("blk.{i}.ffn_up.weight"),
                &[4, 8],
                false,
                false,
                false,
            );
            add(
                format!("blk.{i}.ffn_down.weight"),
                &[8, 4],
                false,
                false,
                false,
            );
        }
    }
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((tensors.len() as u64).to_le_bytes());
    header.extend((meta.len() as u64).to_le_bytes());
    for (key, value) in meta {
        string(&mut header, &key);
        match value {
            Value::U32(n) => {
                header.extend(4u32.to_le_bytes());
                header.extend(n.to_le_bytes());
            }
            Value::Float(n) => {
                header.extend(6u32.to_le_bytes());
                header.extend(n.to_le_bytes());
            }
            Value::Bool(n) => {
                header.extend(7u32.to_le_bytes());
                header.push(n as u8);
            }
            Value::Text(s) => {
                header.extend(8u32.to_le_bytes());
                string(&mut header, &s);
            }
            Value::Texts(xs) => {
                header.extend(9u32.to_le_bytes());
                header.extend(8u32.to_le_bytes());
                header.extend((xs.len() as u64).to_le_bytes());
                for s in xs {
                    string(&mut header, &s);
                }
            }
            Value::Ints(xs) => {
                header.extend(9u32.to_le_bytes());
                header.extend(5u32.to_le_bytes());
                header.extend((xs.len() as u64).to_le_bytes());
                for x in xs {
                    header.extend(x.to_le_bytes());
                }
            }
        }
    }
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        string(&mut header, &name);
        header.extend((shape.len() as u32).to_le_bytes());
        for d in shape {
            header.extend((d as u64).to_le_bytes());
        }
        header.extend(0u32.to_le_bytes());
        header.extend((payload.len() as u64).to_le_bytes());
        for x in values {
            payload.extend(x.to_le_bytes());
        }
    }
    header.resize(header.len().div_ceil(32) * 32, 0);
    header.extend(payload);
    header
}

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}
