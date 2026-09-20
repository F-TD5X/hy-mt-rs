//! Tokenization and chat formatting use the information embedded in GGUF.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Context, Result, anyhow, bail, ensure};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    models::bpe::BPE,
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

use crate::gguf::Gguf;

const HUNYUAN_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const DENSE_RE: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    template: Environment<'static>,
    pieces: Vec<Vec<u8>>,
    eos: BTreeSet<u32>,
    bos_token: String,
    eos_token: String,
}

impl ChatTokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        ensure!(
            g.string("tokenizer.ggml.model")? == "gpt2",
            "only the Hy-MT2 byte BPE tokenizer is supported"
        );
        let tokens = strings(g, "tokenizer.ggml.tokens")?;
        ensure!(tokens.len() <= u32::MAX as usize, "vocabulary is too large");
        let vocab: tokenizers::models::bpe::Vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), i as u32))
            .collect();
        let merges = strings(g, "tokenizer.ggml.merges")?
            .into_iter()
            .map(|s| {
                let (a, b) = s.split_once(' ').context("invalid BPE merge")?;
                Ok((a.to_owned(), b.to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| anyhow!(e.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        let regexes: &[&str] = match g.string("tokenizer.ggml.pre")? {
            "hunyuan" => &[HUNYUAN_RE],
            "hunyuan-dense" => &[r"\p{N}{1,3}", "[一-龥぀-ゟ゠-ヿ]+", DENSE_RE],
            other => bail!("unsupported Hy-MT2 pretokenizer {other}"),
        };
        let mut stages: Vec<PreTokenizerWrapper> = Vec::new();
        // These are sequential Split operations, not one combined regex.
        for regex in regexes {
            stages.push(
                Split::new(
                    SplitPattern::Regex((*regex).to_owned()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .map_err(|e| anyhow!(e.to_string()))?
                .into(),
            );
        }
        stages.push(ByteLevel::new(false, false, false).into());
        tokenizer.with_pre_tokenizer(Some(Sequence::new(stages)));
        let types = g
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .context("missing token types")?;
        ensure!(
            types.len() == tokens.len(),
            "vocabulary and token type counts differ"
        );
        let mut is_special = vec![false; tokens.len()];
        let mut added = Vec::new();
        let mut user_defined = Vec::new();
        for (i, (token, kind)) in tokens.iter().zip(types).enumerate() {
            let kind = kind.as_u64().context("invalid token type")?;
            ensure!((1..=6).contains(&kind), "unsupported token type {kind}");
            if matches!(kind, 2 | 3 | 5) {
                is_special[i] = true;
                added.push(AddedToken::from(token.clone(), true).normalized(false));
            } else if kind == 4 {
                user_defined.push(AddedToken::from(token.clone(), false).normalized(false));
            }
        }
        tokenizer
            .add_special_tokens(added)
            .map_err(|e| anyhow!(e.to_string()))?;
        tokenizer
            .add_tokens(user_defined)
            .map_err(|e| anyhow!(e.to_string()))?;
        for (i, token) in tokens.iter().enumerate() {
            ensure!(
                tokenizer.token_to_id(token) == Some(i as u32),
                "duplicate or remapped vocabulary token at {i}"
            );
        }
        let mut eos = BTreeSet::new();
        for key in [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ] {
            if g.metadata.contains_key(key) {
                let id = g.usize(key)?;
                ensure!(id < tokens.len(), "{key} is outside the vocabulary");
                eos.insert(id as u32);
            }
        }
        ensure!(!eos.is_empty(), "GGUF does not declare an end token");
        let bos = g.usize("tokenizer.ggml.bos_token_id")?;
        let eos_id = g.usize("tokenizer.ggml.eos_token_id")?;
        ensure!(bos < tokens.len(), "BOS is outside the vocabulary");
        let template = compile_template(g.string("tokenizer.chat_template")?)?;
        let decoder = byte_decoder();
        let mut pieces = Vec::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            let piece = if is_special[i] {
                Vec::new()
            } else if types[i].as_u64() == Some(4) {
                token.as_bytes().to_vec()
            } else if types[i].as_u64() == Some(6)
                && token.starts_with("<0x")
                && token.ends_with('>')
            {
                vec![
                    u8::from_str_radix(&token[3..token.len() - 1], 16)
                        .context("invalid byte token")?,
                ]
            } else {
                token
                    .chars()
                    .map(|c| {
                        decoder
                            .get(&c)
                            .copied()
                            .with_context(|| format!("invalid byte BPE character in token {i}"))
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            pieces.push(piece);
        }
        Ok(Self {
            tokenizer,
            template,
            pieces,
            eos,
            bos_token: tokens[bos].clone(),
            eos_token: tokens[eos_id].clone(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    pub fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        ensure!(!messages.is_empty(), "messages must not be empty");
        ensure!(
            messages.iter().skip(1).all(|m| m.role != Role::System),
            "system message must be first"
        );
        Ok(self.template.get_template("chat")?.render(context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            tools => Vec::<Value>::new(),
            reasoning_effort => "no_think",
        })?)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // The chat template inserts BOS. A second BOS changes model output.
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.encode(&self.render(messages)?)
    }

    pub fn piece(&self, id: u32) -> Result<&[u8]> {
        self.pieces
            .get(id as usize)
            .map(Vec::as_slice)
            .context("token ID outside vocabulary")
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.piece(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn strings(g: &Gguf, key: &str) -> Result<Vec<String>> {
    g.metadata
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing {key}"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .with_context(|| format!("non-string in {key}"))
        })
        .collect()
}

fn compile_template(source: &str) -> Result<Environment<'static>> {
    let mut template = Environment::new();
    template.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    template.add_template_owned("chat", source.to_owned())?;
    Ok(template)
}

fn byte_decoder() -> HashMap<char, u8> {
    let mut map = HashMap::new();
    let mut extra = 0u32;
    for byte in 0..=255u32 {
        let character = if (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte)
        {
            byte
        } else {
            let c = 256 + extra;
            extra += 1;
            c
        };
        map.insert(
            char::from_u32(character).expect("valid GPT-2 code point"),
            byte as u8,
        );
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_mapping_is_bijective() {
        let d = byte_decoder();
        assert_eq!(d.len(), 256);
        assert_eq!(d[&'Ġ'], b' ');
        assert_eq!(d[&'Ċ'], b'\n');
        assert_eq!(d[&'a'], b'a');
    }

    #[test]
    fn all_official_templates_match_python_jinja_fixtures() -> Result<()> {
        let fixtures: Value =
            serde_json::from_str(include_str!("../tests/fixtures/tokenizers.json"))?;
        for (name, fixture) in fixtures.as_object().unwrap() {
            let tokenizer = ChatTokenizer {
                tokenizer: Tokenizer::new(BPE::default()),
                template: compile_template(fixture["template"].as_str().unwrap())?,
                bos_token: fixture["bos_token"].as_str().unwrap().into(),
                eos_token: fixture["eos_token"].as_str().unwrap().into(),
                pieces: vec![],
                eos: BTreeSet::new(),
            };
            for case in fixture["cases"].as_array().unwrap() {
                let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone())?;
                assert_eq!(
                    tokenizer.render(&messages)?,
                    case["rendered"].as_str().unwrap(),
                    "{name}"
                );
            }
        }
        Ok(())
    }
}
