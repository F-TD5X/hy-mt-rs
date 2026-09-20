//! Packed GGUF weights and bounded-scratch CPU kernels.
//!
//! Q4_K/Q6_K layouts and the STQ codebook follow ggml's MIT-licensed
//! reference routines. See THIRD_PARTY_NOTICES.md and docs/reference.md.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail, ensure};
use candle_core::{Device, Tensor};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Serialize;

use crate::gguf::{Profile, TensorInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[allow(non_camel_case_types)]
pub enum DType {
    F32,
    F16,
    Q4_K,
    Q6_K,
    Q8_0,
    Q2_0C,
    STQ1_0,
}

impl DType {
    pub fn from_id(id: u32, profile: Profile) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            14 => Self::Q6_K,
            40 if profile == Profile::TencentQ2_0c => Self::Q2_0C,
            42 if profile == Profile::TencentStq1 => Self::STQ1_0,
            _ => bail!("unsupported GGUF tensor type {id} for profile {profile:?}"),
        })
    }

    pub fn block_len(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q8_0 => 32,
            Self::Q2_0C => 512,
            _ => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_K => 144,
            Self::Q6_K => 210,
            Self::Q8_0 => 34,
            Self::Q2_0C => 130,
            Self::STQ1_0 => 42,
        }
    }
}

#[derive(Clone)]
pub struct Weight {
    data: Arc<Mmap>,
    info: TensorInfo,
}

impl Weight {
    pub(crate) fn new(data: Arc<Mmap>, info: TensorInfo) -> Self {
        Self { data, info }
    }
    pub fn shape(&self) -> &[usize] {
        &self.info.dimensions
    }
    pub fn dtype(&self) -> DType {
        self.info.dtype
    }
    pub fn name(&self) -> &str {
        &self.info.name
    }

    pub fn expect_shape(&self, shape: &[usize]) -> Result<()> {
        ensure!(
            self.shape() == shape,
            "{}: expected GGUF shape {shape:?}, got {:?}",
            self.name(),
            self.shape()
        );
        Ok(())
    }

    fn bytes(&self) -> &[u8] {
        &self.data[self.info.offset..self.info.offset + self.info.byte_len]
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let elements = self.info.byte_len / self.dtype().block_bytes() * self.dtype().block_len();
        let mut out = vec![0.; elements];
        decode(self.dtype(), self.bytes(), &mut out)?;
        Ok(out)
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>> {
        let n = self.shape()[0];
        let row_bytes = n / self.dtype().block_len() * self.dtype().block_bytes();
        ensure!(
            row < self.info.byte_len / row_bytes,
            "{}: row {row} out of range",
            self.name()
        );
        let mut out = vec![0.; n];
        decode(
            self.dtype(),
            &self.bytes()[row * row_bytes..(row + 1) * row_bytes],
            &mut out,
        )?;
        Ok(out)
    }

    pub fn expert(&self, index: usize) -> Result<Self> {
        ensure!(
            self.shape().len() == 3 && index < self.shape()[2],
            "{}: invalid expert {index}",
            self.name()
        );
        let mut info = self.info.clone();
        info.byte_len /= info.dimensions.pop().expect("rank checked");
        info.offset += index * info.byte_len;
        Ok(Self::new(self.data.clone(), info))
    }

    /// Multiply [batch,input] activations by [output,input] packed weights.
    /// Larger prefills reuse bounded 32-row float tiles through Rust GEMM.
    pub fn matmul(&self, input: &[f32], batch: usize) -> Result<Vec<f32>> {
        ensure!(self.shape().len() == 2, "{} is not a matrix", self.name());
        let (cols, rows) = (self.shape()[0], self.shape()[1]);
        ensure!(
            batch > 0 && input.len() == batch * cols,
            "{}: invalid activation shape",
            self.name()
        );
        let row_bytes = cols / self.dtype().block_len() * self.dtype().block_bytes();
        // Output-major scratch permits independent weight rows without a lock.
        let mut transposed = vec![0.; rows * batch];
        if batch >= 8 {
            let xs = Tensor::from_slice(input, (batch, cols), &Device::Cpu)?
                .t()?
                .contiguous()?;
            transposed
                .par_chunks_mut(32 * batch)
                .enumerate()
                .try_for_each(|(tile, output)| -> Result<()> {
                    let nrows = output.len() / batch;
                    let mut w = vec![0.; nrows * cols];
                    let start = tile * 32 * row_bytes;
                    decode(
                        self.dtype(),
                        &self.bytes()[start..start + nrows * row_bytes],
                        &mut w,
                    )?;
                    let w = Tensor::from_vec(w, (nrows, cols), &Device::Cpu)?;
                    let result = w.matmul(&xs)?.flatten_all()?.to_vec1::<f32>()?;
                    output.copy_from_slice(&result);
                    Ok(())
                })?;
        } else {
            transposed
                .par_chunks_mut(batch)
                .enumerate()
                .for_each(|(row, output)| {
                    let bytes = &self.bytes()[row * row_bytes..(row + 1) * row_bytes];
                    let mut scratch = [0f32; 512];
                    let block_len = self.dtype().block_len();
                    if matches!(self.dtype(), DType::F32 | DType::F16) {
                        // Decode a row once for unquantized matrices such as the router.
                        let mut values = vec![0.; cols];
                        decode_unchecked(self.dtype(), bytes, &mut values);
                        for (b, out) in output.iter_mut().enumerate() {
                            *out = dot(&values, &input[b * cols..(b + 1) * cols]);
                        }
                    } else {
                        for (block, bytes) in
                            bytes.chunks_exact(self.dtype().block_bytes()).enumerate()
                        {
                            decode_unchecked(self.dtype(), bytes, &mut scratch[..block_len]);
                            for (b, out) in output.iter_mut().enumerate() {
                                let start = b * cols + block * block_len;
                                *out +=
                                    dot(&scratch[..block_len], &input[start..start + block_len]);
                            }
                        }
                    }
                });
        }
        let mut out = vec![0.; batch * rows];
        for b in 0..batch {
            for row in 0..rows {
                out[b * rows + row] = transposed[row * batch + b];
            }
        }
        Ok(out)
    }
}

fn half(bytes: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

pub fn decode(dtype: DType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    ensure!(
        out.len().is_multiple_of(dtype.block_len()),
        "partial {:?} output block",
        dtype
    );
    ensure!(
        bytes.len() == out.len() / dtype.block_len() * dtype.block_bytes(),
        "wrong {:?} byte count",
        dtype
    );
    decode_unchecked(dtype, bytes, out);
    Ok(())
}

const STQ_CODEBOOK: [u8; 32] = [
    0xa9, 0x89, 0x29, 0x09, 0xa6, 0x86, 0x26, 0x06, 0x9a, 0x92, 0x1a, 0x12, 0x6a, 0x62, 0x4a, 0x42,
    0x01, 0x21, 0x81, 0xa1, 0x04, 0x24, 0x84, 0xa4, 0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
];

fn decode_unchecked(dtype: DType, bytes: &[u8], out: &mut [f32]) {
    for (x, y) in bytes
        .chunks_exact(dtype.block_bytes())
        .zip(out.chunks_exact_mut(dtype.block_len()))
    {
        match dtype {
            DType::F32 => y[0] = f32::from_le_bytes(x.try_into().expect("four-byte block")),
            DType::F16 => y[0] = half(x),
            DType::Q8_0 => {
                let d = half(x);
                for (out, &q) in y.iter_mut().zip(&x[2..]) {
                    *out = d * (q as i8) as f32;
                }
            }
            DType::Q2_0C => {
                let d = half(x);
                for (i, out) in y.iter_mut().enumerate() {
                    *out = (2 * ((x[2 + i / 4] >> (2 * (i % 4))) & 3) as i32 - 3) as f32 * d;
                }
            }
            DType::STQ1_0 => {
                let d = half(&x[40..]);
                for g in 0..64 {
                    let code = (x[g / 2] >> (4 * (g % 2))) & 15;
                    let sign = (x[32 + g / 8] >> (g % 8)) & 1;
                    let packed = STQ_CODEBOOK[((sign << 4) | code) as usize];
                    for p in 0..4 {
                        y[g / 16 * 64 + g % 16 + p * 16] =
                            (((packed >> (2 * p)) & 3) as i32 - 1) as f32 * d;
                    }
                }
            }
            DType::Q4_K => {
                let d = half(x);
                let dmin = half(&x[2..]);
                let scales = &x[4..16];
                let scale_min = |j: usize| -> (f32, f32) {
                    let (s, m) = if j < 4 {
                        (scales[j] & 63, scales[j + 4] & 63)
                    } else {
                        (
                            (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
                        )
                    };
                    (d * s as f32, dmin * m as f32)
                };
                for group in 0..4 {
                    let (d1, m1) = scale_min(group * 2);
                    let (d2, m2) = scale_min(group * 2 + 1);
                    for l in 0..32 {
                        let q = x[16 + group * 32 + l];
                        y[group * 64 + l] = d1 * (q & 15) as f32 - m1;
                        y[group * 64 + 32 + l] = d2 * (q >> 4) as f32 - m2;
                    }
                }
            }
            DType::Q6_K => {
                let d = half(&x[208..]);
                for chunk in 0..2 {
                    let ql = &x[chunk * 64..];
                    let qh = &x[128 + chunk * 32..];
                    let sc = &x[192 + chunk * 8..];
                    for l in 0..32 {
                        let qs = [
                            (ql[l] & 15) | ((qh[l] & 3) << 4),
                            (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4),
                            (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4),
                            (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4),
                        ];
                        for p in 0..4 {
                            y[chunk * 128 + p * 32 + l] =
                                d * (sc[l / 16 + p * 2] as i8) as f32 * (qs[p] as i32 - 32) as f32;
                        }
                    }
                }
            }
        }
    }
}

type Dot = fn(&[f32], &[f32]) -> f32;
static DOT: OnceLock<Dot> = OnceLock::new();

pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return "neon";
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return "avx2";
    }
    "scalar"
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    DOT.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return |a, b| {
                // SAFETY: selected only when NEON is available; lengths match.
                unsafe { dot_neon(a, b) }
            };
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return |a, b| {
                // SAFETY: selected only when AVX2 is available; lengths match.
                unsafe { dot_avx2(a, b) }
            };
        }
        dot_scalar
    })(a, b)
}

pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_f32(0.);
    let end = a.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        // SAFETY: the four-element loads are within the equal-length slices.
        unsafe {
            sum = vaddq_f32(
                sum,
                vmulq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i))),
            );
        }
    }
    vaddvq_f32(sum) + dot_scalar(&a[end..], &b[end..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let end = a.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        // SAFETY: the eight-element unaligned loads stay within both slices.
        unsafe {
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                ),
            );
        }
    }
    let mut lanes = [0f32; 8];
    // SAFETY: lanes has space for the full vector.
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    }
    lanes.iter().sum::<f32>() + dot_scalar(&a[end..], &b[end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2_codes_and_scale() -> Result<()> {
        let mut block = vec![0b11100100; 130];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        let mut out = [0.; 512];
        decode(DType::Q2_0C, &block, &mut out)?;
        for group in out.as_chunks::<4>().0 {
            assert_eq!(*group, [-1.5, -0.5, 0.5, 1.5]);
        }
        Ok(())
    }

    #[test]
    fn stq_all_patterns_and_stride() -> Result<()> {
        let mut block = [0u8; 42];
        block[40..].copy_from_slice(&f16::from_f32(2.).to_le_bytes());
        for g in 0..64 {
            block[g / 2] |= ((g % 16) as u8) << (4 * (g % 2));
            block[32 + g / 8] |= ((g / 16 % 2) as u8) << (g % 8);
        }
        let mut out = [0.; 256];
        decode(DType::STQ1_0, &block, &mut out)?;
        for g in 0..64 {
            let zero = (g % 16) / 4;
            let tail = g % 4;
            let mut nz = 0;
            for p in 0..4 {
                let expected = if p == zero {
                    0.
                } else {
                    let sign = if nz == 0 || tail & (1 << (nz - 1)) == 0 {
                        1.
                    } else {
                        -1.
                    };
                    nz += 1;
                    sign * if g / 16 % 2 == 0 { 2. } else { -2. }
                };
                assert_eq!(out[g / 16 * 64 + g % 16 + 16 * p], expected);
            }
        }
        Ok(())
    }

    #[test]
    fn vector_kernel_matches_scalar_with_tails() {
        for n in [0, 1, 3, 7, 16, 32, 127, 256, 512, 2049] {
            let a: Vec<_> = (0..n).map(|i| (i as f32 * 0.7).sin()).collect();
            let b: Vec<_> = (0..n).map(|i| (i as f32 * 0.9).cos()).collect();
            let error = (dot(&a, &b) - dot_scalar(&a, &b)).abs();
            assert!(error < 1e-4, "{n}: {error}");
        }
    }

    #[test]
    fn ambiguous_ids_need_the_matching_profile() {
        assert!(DType::from_id(40, Profile::Standard).is_err());
        assert!(DType::from_id(42, Profile::TencentQ2_0c).is_err());
        assert_eq!(
            DType::from_id(42, Profile::TencentStq1).unwrap(),
            DType::STQ1_0
        );
    }
}
