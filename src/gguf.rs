//! Bounds-checked GGUF v3 reader. Tensor type IDs are interpreted per file profile.

use std::{collections::BTreeMap, fs::File, path::Path, sync::Arc};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use memmap2::{Mmap, MmapOptions};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::quant::{DType, Weight};

pub const STQ_SHA256: &str = "cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93";
pub const Q2C_SHA256: &str = "dcc33bbae9b28d923c8c76a64f6157840841d26f8774f3dfd770d5fabeeb1cd7";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    #[default]
    Auto,
    Standard,
    #[serde(rename = "tencent-q2-0c")]
    #[value(name = "tencent-q2-0c")]
    TencentQ2_0c,
    TencentStq1,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: contiguous input dimension first.
    pub dimensions: Vec<usize>,
    pub type_id: u32,
    pub dtype: DType,
    /// Absolute byte offset in the file.
    pub offset: usize,
    pub byte_len: usize,
}

pub struct Gguf {
    data: Arc<Mmap>,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub profile: Profile,
}

impl Gguf {
    /// The model file must remain unchanged while this mapping is in use.
    pub fn open(path: impl AsRef<Path>, requested: Profile) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("opening {}", path.as_ref().display()))?;
        // SAFETY: read-only mapping, held by Arc for the lifetime of all weights.
        // As with other mmap readers, callers must not modify/truncate the file.
        let data = Arc::new(unsafe { MmapOptions::new().map(&file)? });
        Self::parse(data, requested)
    }

    fn parse(data: Arc<Mmap>, requested: Profile) -> Result<Self> {
        let mut r = Reader {
            bytes: &data,
            pos: 0,
        };
        ensure!(r.take(4)? == b"GGUF", "not a GGUF file");
        ensure!(r.u32()? == 3, "only GGUF v3 is supported");
        let tensor_count = r.count(24)?;
        let kv_count = r.count(13)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = r.string()?;
            let kind = r.u32()?;
            let value = r.value(kind)?;
            ensure!(
                metadata.insert(key.clone(), value).is_none(),
                "duplicate metadata key {key}"
            );
        }
        let alignment = metadata.get("general.alignment").map_or(Ok(32), |v| {
            v.as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .context("invalid GGUF alignment")
        })?;
        ensure!(
            alignment.is_power_of_two() && alignment <= 65536,
            "invalid GGUF alignment {alignment}"
        );
        if let Some(n) = metadata.get("split.count") {
            ensure!(
                n.as_u64() == Some(1),
                "split GGUF files are not supported; supply a merged file"
            );
        }
        let mut raw = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let rank = r.u32()? as usize;
            ensure!((1..=4).contains(&rank), "invalid rank for tensor {name}");
            let mut dimensions = Vec::with_capacity(rank);
            for _ in 0..rank {
                let dimension = usize::try_from(r.u64()?).context("tensor dimension too large")?;
                ensure!(dimension > 0, "zero dimension in tensor {name}");
                dimensions.push(dimension);
            }
            let type_id = r.u32()?;
            let offset = usize::try_from(r.u64()?).context("tensor offset too large")?;
            ensure!(offset % alignment == 0, "unaligned tensor {name}");
            raw.push((name, dimensions, type_id, offset));
        }
        let data_start = r
            .pos
            .checked_add(alignment - 1)
            .context("offset overflow")?
            & !(alignment - 1);
        ensure!(data_start <= data.len(), "missing tensor data");
        let has_legacy_ids = raw.iter().any(|t| matches!(t.2, 40 | 42));
        let profile = match requested {
            Profile::Auto if has_legacy_ids => {
                let digest = format!("{:x}", Sha256::digest(&*data));
                match digest.as_str() {
                    Q2C_SHA256 => Profile::TencentQ2_0c,
                    STQ_SHA256 => Profile::TencentStq1,
                    _ => bail!(
                        "ambiguous GGUF type 40/42 in an unrecognized file (SHA256 {digest}); use --gguf-profile tencent-q2-0c or tencent-stq1 only for a known Tencent legacy export"
                    ),
                }
            }
            Profile::Auto => Profile::Standard,
            other => other,
        };
        if matches!(profile, Profile::TencentQ2_0c | Profile::TencentStq1) {
            ensure!(
                metadata.get("general.architecture").and_then(Value::as_str)
                    == Some("hunyuan-dense"),
                "Tencent low-bit profiles require hunyuan-dense"
            );
            let (id, file_type) = if profile == Profile::TencentQ2_0c {
                (40, 39)
            } else {
                (42, 41)
            };
            ensure!(
                metadata.get("general.file_type").and_then(Value::as_u64) == Some(file_type),
                "incompatible Tencent general.file_type"
            );
            ensure!(
                raw.iter().any(|t| t.2 == id),
                "file has no tensors for selected Tencent profile"
            );
            ensure!(
                raw.iter().all(|t| [0, 14, id].contains(&t.2)),
                "unexpected mixed types in Tencent legacy export"
            );
        }
        let mut tensors = BTreeMap::new();
        let mut extents = Vec::with_capacity(tensor_count);
        for (name, dimensions, type_id, relative_offset) in raw {
            let dtype =
                DType::from_id(type_id, profile).with_context(|| format!("tensor {name}"))?;
            ensure!(
                dimensions[0] % dtype.block_len() == 0,
                "tensor {name} has a partial quantization block in its row"
            );
            let elements = dimensions
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .context("tensor element count overflow")?;
            let byte_len = (elements / dtype.block_len())
                .checked_mul(dtype.block_bytes())
                .context("tensor byte count overflow")?;
            let offset = data_start
                .checked_add(relative_offset)
                .context("tensor offset overflow")?;
            let end = offset
                .checked_add(byte_len)
                .context("tensor end overflow")?;
            ensure!(
                end <= data.len(),
                "truncated tensor {name}: end {end}, file length {}",
                data.len()
            );
            extents.push((offset, end, name.clone()));
            let info = TensorInfo {
                name: name.clone(),
                dimensions,
                type_id,
                dtype,
                offset,
                byte_len,
            };
            ensure!(
                tensors.insert(name.clone(), info).is_none(),
                "duplicate tensor {name}"
            );
        }
        extents.sort_unstable_by_key(|x| x.0);
        for pair in extents.windows(2) {
            ensure!(
                pair[0].1 <= pair[1].0,
                "overlapping tensors {} and {}",
                pair[0].2,
                pair[1].2
            );
        }
        Ok(Self {
            data,
            metadata,
            tensors,
            profile,
        })
    }

    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    pub fn tensor(&self, name: &str) -> Result<Weight> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        Ok(Weight::new(self.data.clone(), info.clone()))
    }

    pub fn string(&self, key: &str) -> Result<&str> {
        self.metadata
            .get(key)
            .and_then(Value::as_str)
            .with_context(|| format!("missing or invalid string metadata {key}"))
    }

    pub fn usize(&self, key: &str) -> Result<usize> {
        self.metadata
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .with_context(|| format!("missing or invalid integer metadata {key}"))
    }

    pub fn float(&self, key: &str) -> Result<f32> {
        let v = self
            .metadata
            .get(key)
            .and_then(Value::as_f64)
            .with_context(|| format!("missing or invalid float metadata {key}"))?
            as f32;
        ensure!(v.is_finite(), "non-finite metadata {key}");
        Ok(v)
    }

    pub fn boolean(&self, key: &str) -> Result<bool> {
        self.metadata
            .get(key)
            .and_then(Value::as_bool)
            .with_context(|| format!("missing or invalid boolean metadata {key}"))
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("GGUF offset overflow")?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .context("truncated GGUF header")?;
        self.pos = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn count(&mut self, minimum_bytes: usize) -> Result<usize> {
        let n = usize::try_from(self.u64()?).context("GGUF count too large")?;
        ensure!(
            n <= (self.bytes.len() - self.pos) / minimum_bytes,
            "GGUF count exceeds file bounds"
        );
        Ok(n)
    }

    fn string(&mut self) -> Result<String> {
        let n = self.count(1)?;
        ensure!(n <= 64 * 1024 * 1024, "GGUF string exceeds 64 MiB limit");
        Ok(std::str::from_utf8(self.take(n)?)
            .context("invalid GGUF UTF-8")?
            .to_owned())
    }

    fn value(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::from(self.take(1)?[0]),
            1 => Value::from(self.take(1)?[0] as i8),
            2 => Value::from(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::from(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::from(self.u32()?),
            5 => Value::from(self.u32()? as i32),
            6 => {
                let n = f32::from_bits(self.u32()?);
                ensure!(n.is_finite(), "non-finite GGUF metadata");
                Value::from(n)
            }
            7 => {
                let n = self.take(1)?[0];
                ensure!(n <= 1, "invalid GGUF boolean");
                Value::from(n == 1)
            }
            8 => Value::from(self.string()?),
            9 => {
                let element_type = self.u32()?;
                let minimum_bytes = match element_type {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    8 | 10..=12 => 8,
                    _ => bail!("unsupported GGUF array element type {element_type}"),
                };
                let n = self.count(minimum_bytes)?;
                ensure!(n <= 16_000_000, "GGUF array is too large");
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.value(element_type)?);
                }
                Value::Array(items)
            }
            10 => Value::from(self.u64()?),
            11 => Value::from(self.u64()? as i64),
            12 => {
                let n = f64::from_bits(self.u64()?);
                ensure!(n.is_finite(), "non-finite GGUF metadata");
                Value::from(n)
            }
            _ => bail!("unsupported GGUF metadata type {kind}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn tiny() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        string(&mut b, "weight");
        b.extend(1u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.extend(1f32.to_le_bytes());
        b.extend((-2f32).to_le_bytes());
        b
    }

    fn open_bytes(b: &[u8]) -> Result<Gguf> {
        let mut f = tempfile::NamedTempFile::new()?;
        f.write_all(b)?;
        Gguf::open(f.path(), Profile::Auto)
    }

    #[test]
    fn reads_little_endian_tensor() -> Result<()> {
        let g = open_bytes(&tiny())?;
        assert_eq!(g.tensor("weight")?.to_vec()?, vec![1., -2.]);
        Ok(())
    }

    #[test]
    fn every_truncation_is_an_error() {
        let b = tiny();
        for n in 0..b.len() {
            assert!(open_bytes(&b[..n]).is_err(), "truncation {n}");
        }
    }

    #[test]
    fn excessive_header_counts_fail_before_allocation() {
        let mut b = tiny();
        b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(open_bytes(&b).is_err());
    }
}
