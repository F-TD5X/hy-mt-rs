//! Packed GGUF weights and bounded-scratch CPU kernels.
//!
//! Q4_K/Q6_K layouts and the STQ codebook follow ggml's MIT-licensed
//! reference routines. See THIRD_PARTY_NOTICES.md and docs/reference.md.

use std::sync::Arc;

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

    pub fn bytes(&self) -> &[u8] {
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

        if batch == 1 {
            let mut out = vec![0.; rows];
            let chunk_size = chunk_rows(rows);
            #[cfg(target_arch = "aarch64")]
            if matches!(self.dtype(), DType::STQ1_0 | DType::Q6_K) && sdot_available() {
                self.gemv_q8(input, &mut out, chunk_size);
                return Ok(out);
            }
            match self.dtype() {
                DType::STQ1_0 => {
                    let all_bytes = self.bytes();
                    out.par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let base_row = chunk_idx * chunk_size;
                            let chunk_bytes = &all_bytes
                                [base_row * row_bytes..(base_row + chunk.len()) * row_bytes];
                            for (val, row_b) in
                                chunk.iter_mut().zip(chunk_bytes.chunks_exact(row_bytes))
                            {
                                *val = stq_row_dot(row_b, input);
                            }
                        });
                }
                DType::Q6_K => {
                    let all_bytes = self.bytes();
                    out.par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let base_row = chunk_idx * chunk_size;
                            let chunk_bytes = &all_bytes
                                [base_row * row_bytes..(base_row + chunk.len()) * row_bytes];
                            for (val, row_b) in
                                chunk.iter_mut().zip(chunk_bytes.chunks_exact(row_bytes))
                            {
                                *val = q6_row_dot(row_b, input);
                            }
                        });
                }
                DType::F32 | DType::F16 => {
                    out.par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let base_row = chunk_idx * chunk_size;
                            let mut values = vec![0.; cols];
                            for (i, val) in chunk.iter_mut().enumerate() {
                                let row = base_row + i;
                                let bytes = &self.bytes()[row * row_bytes..(row + 1) * row_bytes];
                                decode_unchecked(self.dtype(), bytes, &mut values);
                                *val = dot(&values, input);
                            }
                        });
                }
                _ => {
                    out.par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let base_row = chunk_idx * chunk_size;
                            let mut scratch = [0f32; 512];
                            let block_len = self.dtype().block_len();
                            let block_bytes = self.dtype().block_bytes();
                            for (i, val) in chunk.iter_mut().enumerate() {
                                let row = base_row + i;
                                let bytes = &self.bytes()[row * row_bytes..(row + 1) * row_bytes];
                                let mut sum = 0.0;
                                for (block, b_bytes) in bytes.chunks_exact(block_bytes).enumerate()
                                {
                                    let start = block * block_len;
                                    decode_unchecked(
                                        self.dtype(),
                                        b_bytes,
                                        &mut scratch[..block_len],
                                    );
                                    sum += dot(
                                        &scratch[..block_len],
                                        &input[start..start + block_len],
                                    );
                                }
                                *val = sum;
                            }
                        });
                }
            }
            return Ok(out);
        }

        // Output-major scratch permits independent weight rows without a lock.
        let mut transposed = vec![0.; rows * batch];
        if batch >= 8 {
            let xs = Tensor::from_slice(input, (batch, cols), &Device::Cpu)?
                .t()?
                .contiguous()?;
            // 64 rows fits the decoded STQ tile and its GEMM working set in cache
            // on the CPU path; wider tiles reduce long-prefill throughput.
            let tile_rows = 64;
            transposed
                .par_chunks_mut(tile_rows * batch)
                .enumerate()
                .try_for_each(|(tile, output)| -> Result<()> {
                    let nrows = output.len() / batch;
                    let mut w = vec![0.; nrows * cols];
                    let start = tile * tile_rows * row_bytes;
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
                .par_chunks_mut(32 * batch)
                .enumerate()
                .for_each(|(tile, tile_out)| {
                    let mut scratch = vec![0f32; cols];
                    for (row_idx, output) in tile_out.chunks_mut(batch).enumerate() {
                        let row = tile * 32 + row_idx;
                        let bytes = &self.bytes()[row * row_bytes..(row + 1) * row_bytes];
                        decode_unchecked(self.dtype(), bytes, &mut scratch);
                        for (b, out) in output.iter_mut().enumerate() {
                            *out = dot(&scratch, &input[b * cols..(b + 1) * cols]);
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

    /// Batch-1 GEMV through int8 activations and SDOT kernels.
    #[cfg(target_arch = "aarch64")]
    fn gemv_q8(&self, input: &[f32], out: &mut [f32], chunk_size: usize) {
        let cols = self.shape()[0];
        let row_bytes = cols / self.dtype().block_len() * self.dtype().block_bytes();
        let q8 = Q8::quantize(input);
        let dtype = self.dtype();
        let all_bytes = self.bytes();
        out.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_size;
                let chunk_bytes =
                    &all_bytes[base_row * row_bytes..(base_row + chunk.len()) * row_bytes];
                for (val, row) in chunk.iter_mut().zip(chunk_bytes.chunks_exact(row_bytes)) {
                    // SAFETY: each row is a whole number of 256-weight blocks,
                    // and `q8` holds one scale and 256 activations per block.
                    *val = unsafe { row_dot_q8(dtype, row, &q8) };
                }
            });
    }

    /// Batch-1 query, key and value projections of one attention layer.
    ///
    /// All three read the same activations, so one parallel region replaces
    /// three rayon forks and the int8 quantization is shared. Other batch
    /// sizes and architectures fall back to three independent products.
    pub fn gemv_qkv(
        q: &Weight,
        k: &Weight,
        v: &Weight,
        input: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let cols = q.shape()[0];
        ensure!(
            [q, k, v].iter().all(|weight| weight.shape().len() == 2)
                && k.shape()[0] == cols
                && v.shape()[0] == cols,
            "query, key and value must be matrices that share an input width"
        );
        ensure!(input.len() == cols, "invalid activation shape for qkv");

        #[cfg(target_arch = "aarch64")]
        if sdot_available()
            && [q, k, v]
                .iter()
                .all(|weight| matches!(weight.dtype(), DType::STQ1_0 | DType::Q6_K))
        {
            let q8 = Q8::quantize(input);
            let weights = [q, k, v];
            let mut outputs = [
                vec![0.; q.shape()[1]],
                vec![0.; k.shape()[1]],
                vec![0.; v.shape()[1]],
            ];
            // One task per row chunk of each projection, so all three run in a
            // single parallel region.
            let mut tasks: Vec<(DType, &[u8], &mut [f32])> = Vec::new();
            for (weight, output) in weights.iter().zip(outputs.iter_mut()) {
                let dtype = weight.dtype();
                let row_bytes = cols / dtype.block_len() * dtype.block_bytes();
                let bytes = weight.bytes();
                let chunk_size = chunk_rows(output.len());
                for (chunk_index, chunk) in output.chunks_mut(chunk_size).enumerate() {
                    let start = chunk_index * chunk_size;
                    tasks.push((
                        dtype,
                        &bytes[start * row_bytes..(start + chunk.len()) * row_bytes],
                        chunk,
                    ));
                }
            }
            tasks.par_iter_mut().for_each(|(dtype, bytes, output)| {
                let row_bytes = cols / dtype.block_len() * dtype.block_bytes();
                for (value, row) in output.iter_mut().zip(bytes.chunks_exact(row_bytes)) {
                    // SAFETY: whole rows against whole activation groups.
                    *value = unsafe { row_dot_q8(*dtype, row, &q8) };
                }
            });
            // Release the chunk borrows before moving the outputs out.
            drop(tasks);
            let [q_out, k_out, v_out] = outputs;
            return Ok((q_out, k_out, v_out));
        }

        Ok((
            q.matmul(input, 1)?,
            k.matmul(input, 1)?,
            v.matmul(input, 1)?,
        ))
    }

    /// Fused gate and up projection with SiLU activation: `silu(gate(x)) * up(x)`.
    pub fn matmul_gate_up_silu(&self, up: &Weight, input: &[f32]) -> Result<Vec<f32>> {
        let (cols, rows) = (self.shape()[0], self.shape()[1]);
        ensure!(up.shape() == self.shape(), "gate and up shape mismatch");
        ensure!(
            input.len() == cols,
            "invalid activation shape for gate_up_silu"
        );
        let row_bytes = cols / self.dtype().block_len() * self.dtype().block_bytes();
        let mut out = vec![0.; rows];
        let chunk_size = chunk_rows(rows);

        #[cfg(target_arch = "aarch64")]
        if sdot_available()
            && matches!(self.dtype(), DType::STQ1_0 | DType::Q6_K)
            && self.dtype() == up.dtype()
        {
            let q8 = Q8::quantize(input);
            let dtype = self.dtype();
            let self_bytes = self.bytes();
            let up_bytes = up.bytes();
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_row = chunk_idx * chunk_size;
                    let self_chunk =
                        &self_bytes[base_row * row_bytes..(base_row + chunk.len()) * row_bytes];
                    let up_chunk =
                        &up_bytes[base_row * row_bytes..(base_row + chunk.len()) * row_bytes];
                    for ((val, g_b), u_b) in chunk
                        .iter_mut()
                        .zip(self_chunk.chunks_exact(row_bytes))
                        .zip(up_chunk.chunks_exact(row_bytes))
                    {
                        // SAFETY: whole rows of 256-weight blocks against whole
                        // activation groups, one scale per block.
                        let (g, u) =
                            unsafe { (row_dot_q8(dtype, g_b, &q8), row_dot_q8(dtype, u_b, &q8)) };
                        let silu = g / (1.0 + (-g).exp());
                        *val = silu * u;
                    }
                });
            return Ok(out);
        }

        match (self.dtype(), up.dtype()) {
            (DType::STQ1_0, DType::STQ1_0) => {
                let self_bytes = self.bytes();
                let up_bytes = up.bytes();
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base_row = chunk_idx * chunk_size;
                        let self_chunk =
                            &self_bytes[base_row * row_bytes..(base_row + chunk.len()) * row_bytes];
                        let up_chunk =
                            &up_bytes[base_row * row_bytes..(base_row + chunk.len()) * row_bytes];
                        for ((val, g_b), u_b) in chunk
                            .iter_mut()
                            .zip(self_chunk.chunks_exact(row_bytes))
                            .zip(up_chunk.chunks_exact(row_bytes))
                        {
                            let g = stq_row_dot(g_b, input);
                            let u = stq_row_dot(u_b, input);
                            let silu = g / (1.0 + (-g).exp());
                            *val = silu * u;
                        }
                    });
            }
            (DType::Q6_K, DType::Q6_K) => {
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base_row = chunk_idx * chunk_size;
                        for (i, val) in chunk.iter_mut().enumerate() {
                            let row = base_row + i;
                            let g_bytes = &self.bytes()[row * row_bytes..(row + 1) * row_bytes];
                            let u_bytes = &up.bytes()[row * row_bytes..(row + 1) * row_bytes];
                            let g = q6_row_dot(g_bytes, input);
                            let u = q6_row_dot(u_bytes, input);
                            let silu = g / (1.0 + (-g).exp());
                            *val = silu * u;
                        }
                    });
            }
            _ => {
                let mut gate = self.matmul(input, 1)?;
                let up_out = up.matmul(input, 1)?;
                for (g, u) in gate.iter_mut().zip(up_out) {
                    *g = (*g / (1. + (-*g).exp())) * u;
                }
                return Ok(gate);
            }
        }
        Ok(out)
    }
}

#[inline(always)]
fn half(bytes: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

/// Elements per quantized activation group. Matches every packed block this
/// kernel set understands, so one group's integer dot takes one FP32 scale.
#[cfg(target_arch = "aarch64")]
const Q8_GROUP: usize = 256;

/// int8 activations with one scale per 256-element group.
///
/// Quantizing the activations is what lets the GEMV kernels use integer dot
/// products (SDOT on ARMv8.2): four products per instruction instead of one
/// NEON FMA per four products. The scale layout follows the packed weight
/// blocks, so a block's contribution is `weight_scale * group_scale * dot`.
#[cfg(target_arch = "aarch64")]
struct Q8 {
    values: Vec<i8>,
    scales: Vec<f32>,
}

#[cfg(target_arch = "aarch64")]
impl Q8 {
    fn quantize(input: &[f32]) -> Self {
        let mut values = vec![0i8; input.len()];
        let mut scales = Vec::with_capacity(input.len().div_ceil(Q8_GROUP));
        for (source, target) in input.chunks(Q8_GROUP).zip(values.chunks_mut(Q8_GROUP)) {
            let max = source.iter().fold(0f32, |acc, value| acc.max(value.abs()));
            let (scale, inverse) = if max > 0. {
                (max / 127., 127. / max)
            } else {
                (0., 0.)
            };
            scales.push(scale);
            // SAFETY: both slices have the same length.
            unsafe { quantize_group_neon(source, inverse, target) };
        }
        Self { values, scales }
    }

    fn values(&self) -> &[i8] {
        &self.values
    }

    fn scales(&self) -> &[f32] {
        &self.scales
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_group_neon(source: &[f32], inverse: f32, target: &mut [i8]) {
    use std::arch::aarch64::*;
    // SAFETY: `source` and `target` are equal length; all loads and stores stay
    // inside one 16-element step, and the tail is handled scalar afterwards.
    unsafe {
        let inverse_vec = vdupq_n_f32(inverse);
        let mut index = 0;
        while index + 16 <= source.len() {
            let ptr = source.as_ptr().add(index);
            let mut lanes: [int16x4_t; 4] = [vdup_n_s16(0); 4];
            for (vector, lane) in lanes.iter_mut().enumerate() {
                let scaled = vmulq_f32(vld1q_f32(ptr.add(4 * vector)), inverse_vec);
                *lane = vqmovn_s32(vcvtnq_s32_f32(scaled));
            }
            let low = vcombine_s16(lanes[0], lanes[1]);
            let high = vcombine_s16(lanes[2], lanes[3]);
            let packed = vcombine_s8(vqmovn_s16(low), vqmovn_s16(high));
            vst1q_s8(target.as_mut_ptr().add(index), packed);
            index += 16;
        }
        for (value, &activations) in target[index..].iter_mut().zip(&source[index..]) {
            *value = (activations * inverse).round().clamp(-127., 127.) as i8;
        }
    }
}

/// ARMv8.2 signed dot product over 16 int8 lanes, four products per result
/// lane. Written as one instruction because the `vdotq_s32` intrinsic needs a
/// newer compiler than this crate's minimum supported Rust version.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot(
    acc: std::arch::aarch64::int32x4_t,
    activations: std::arch::aarch64::int8x16_t,
    digits: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut out = acc;
    // SAFETY: the instruction only reads the two vector registers and
    // accumulates into `out`; no memory is touched.
    unsafe {
        std::arch::asm!(
            "sdot {accumulator:v}.4s, {activations:v}.16b, {digits:v}.16b",
            accumulator = inout(vreg) out,
            activations = in(vreg) activations,
            digits = in(vreg) digits,
            options(nostack, nomem, pure)
        );
    }
    out
}

/// True when this CPU has ARMv8.2 integer dot products (`sdot`).
#[cfg(target_arch = "aarch64")]
fn sdot_available() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let available = std::arch::is_aarch64_feature_detected!("dotprod");
            STATE.store(if available { 1 } else { 2 }, Ordering::Relaxed);
            available
        }
    }
}

/// Rows per rayon task for the batch-1 GEMV path. Rows are short once the
/// integer kernels run, so many small tasks balance better than a few large
/// ones; 32 rows is the useful floor at roughly 2 microseconds of work.
fn chunk_rows(rows: usize) -> usize {
    let target_chunks = (rayon::current_num_threads() * 32).max(1);
    (rows / target_chunks).clamp(32, 2048)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn half_raw(ptr: *const u8) -> f32 {
    let bits = unsafe { (ptr as *const u16).read_unaligned() };
    f16::from_bits(u16::from_le(bits)).to_f32()
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

/// Expand one 42-byte STQ block into 256 ternary digits in output order.
///
/// The packed layout places group `g`'s lane `p` at `(g/16)*64 + g%16 + p*16`.
/// Iterating lane-major instead of group-major keeps every write contiguous.
/// All slices are exact, so the inner loop compiles without bounds checks and
/// auto-vectorizes. `scale` folds the block's FP16 factor into each digit.
fn stq_digits(bytes: &[u8], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: out has len 256, bytes has len at least 42.
        unsafe { stq_digits_neon(bytes, scale, out) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    stq_digits_scalar(bytes, scale, out);
}

#[cfg(not(target_arch = "aarch64"))]
fn stq_digits_scalar(bytes: &[u8], scale: f32, out: &mut [f32]) {
    let mut packed = [0u8; 64];
    for (c, chunk) in packed.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        for (j, slot) in chunk.iter_mut().enumerate() {
            let g = c * 16 + j;
            let code = (bytes[g >> 1] >> (4 * (g % 2))) & 15;
            let sign = (bytes[32 + (g >> 3)] >> (g & 7)) & 1;
            *slot = STQ_CODEBOOK[((sign << 4) | code) as usize];
        }
    }
    let chunks = packed.as_chunks::<16>().0;
    for (out64, chunk) in out.as_chunks_mut::<64>().0.iter_mut().zip(chunks) {
        for (p, out16) in out64.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let shift = p * 2;
            for (slot, o) in chunk.iter().zip(out16) {
                *o = (((slot >> shift) & 3) as i32 - 1) as f32 * scale;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn stq_digits_neon(bytes: &[u8], scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let cbs = uint8x16x2_t(
            vld1q_u8(STQ_CODEBOOK.as_ptr()),
            vld1q_u8(STQ_CODEBOOK.as_ptr().add(16)),
        );
        let sign_shifts: [i8; 16] = [4, 3, 2, 1, 0, -1, -2, -3, 4, 3, 2, 1, 0, -1, -2, -3];
        let sign_shifts = vld1q_s8(sign_shifts.as_ptr());
        let one = vdupq_n_u8(1);
        let mask15 = vdup_n_u8(15);
        let mask16 = vdupq_n_u8(0x10);
        let mask3 = vdupq_n_u8(3);
        let scale_vec = vdupq_n_f32(scale);
        let out_ptr = out.as_mut_ptr();

        for c in 0..4usize {
            let cw = vld1_u8(bytes.as_ptr().add(8 * c));
            let lo = vand_u8(cw, mask15);
            let hi = vshr_n_u8(cw, 4);
            let codes = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
            let b01 = vcombine_u8(
                vdup_n_u8(*bytes.as_ptr().add(32 + 2 * c)),
                vdup_n_u8(*bytes.as_ptr().add(33 + 2 * c)),
            );
            let signs = vandq_u8(vshlq_u8(b01, sign_shifts), mask16);
            let packed = vqtbl2q_u8(cbs, vorrq_u8(codes, signs));

            let t8_0 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, mask3), one));
            let lo8_0 = vmovl_s8(vget_low_s8(t8_0));
            let hi8_0 = vmovl_high_s8(t8_0);
            let dst0 = out_ptr.add(c * 64);
            vst1q_f32(
                dst0,
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_0))), scale_vec),
            );
            vst1q_f32(
                dst0.add(4),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo8_0)), scale_vec),
            );
            vst1q_f32(
                dst0.add(8),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_0))), scale_vec),
            );
            vst1q_f32(
                dst0.add(12),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi8_0)), scale_vec),
            );

            let t8_1 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 2), mask3), one));
            let lo8_1 = vmovl_s8(vget_low_s8(t8_1));
            let hi8_1 = vmovl_high_s8(t8_1);
            let dst1 = dst0.add(16);
            vst1q_f32(
                dst1,
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_1))), scale_vec),
            );
            vst1q_f32(
                dst1.add(4),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo8_1)), scale_vec),
            );
            vst1q_f32(
                dst1.add(8),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_1))), scale_vec),
            );
            vst1q_f32(
                dst1.add(12),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi8_1)), scale_vec),
            );

            let t8_2 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 4), mask3), one));
            let lo8_2 = vmovl_s8(vget_low_s8(t8_2));
            let hi8_2 = vmovl_high_s8(t8_2);
            let dst2 = dst0.add(32);
            vst1q_f32(
                dst2,
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_2))), scale_vec),
            );
            vst1q_f32(
                dst2.add(4),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo8_2)), scale_vec),
            );
            vst1q_f32(
                dst2.add(8),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_2))), scale_vec),
            );
            vst1q_f32(
                dst2.add(12),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi8_2)), scale_vec),
            );

            let t8_3 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 6), mask3), one));
            let lo8_3 = vmovl_s8(vget_low_s8(t8_3));
            let hi8_3 = vmovl_high_s8(t8_3);
            let dst3 = dst0.add(48);
            vst1q_f32(
                dst3,
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_3))), scale_vec),
            );
            vst1q_f32(
                dst3.add(4),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo8_3)), scale_vec),
            );
            vst1q_f32(
                dst3.add(8),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_3))), scale_vec),
            );
            vst1q_f32(
                dst3.add(12),
                vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi8_3)), scale_vec),
            );
        }
    }
}

#[allow(dead_code)]
type StqDot = fn(&[u8], &[f32]) -> f32;

/// Dot one 42-byte STQ block (256 ternary weights) against 256 activations.
/// Returns the unscaled sum; the block's FP16 scale is applied by the caller.
#[allow(dead_code)]
pub(crate) fn stq_dot(bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(bytes.len(), 42);
    debug_assert_eq!(x.len(), 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers pass a 42-byte block and exactly 256 activations.
        unsafe { stq_dot_neon(bytes, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut digits = [0f32; 256];
        stq_digits(bytes, 1., &mut digits);
        dot(&digits, x)
    }
}

#[inline(always)]
pub(crate) fn stq_row_dot(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { stq_row_dot_neon(row_bytes, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0;
        for (block, bytes) in row_bytes.chunks_exact(42).enumerate() {
            let d = half(&bytes[40..]);
            let xs = &x[block * 256..(block + 1) * 256];
            sum += d * stq_dot(bytes, xs);
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn stq_row_dot_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let cbs = uint8x16x2_t(
            vld1q_u8(STQ_CODEBOOK.as_ptr()),
            vld1q_u8(STQ_CODEBOOK.as_ptr().add(16)),
        );
        let sign_shifts: [i8; 16] = [4, 3, 2, 1, 0, -1, -2, -3, 4, 3, 2, 1, 0, -1, -2, -3];
        let sign_shifts = vld1q_s8(sign_shifts.as_ptr());
        let one = vdupq_n_u8(1);
        let mask15 = vdup_n_u8(15);
        let mask16 = vdupq_n_u8(0x10);
        let mask3 = vdupq_n_u8(3);
        let mut row_acc = vdupq_n_f32(0.);

        let num_blocks = row_bytes.len() / 42;
        let mut bytes_ptr = row_bytes.as_ptr();
        let mut x_ptr = x.as_ptr();

        for _ in 0..num_blocks {
            let d = half_raw(bytes_ptr.add(40));
            let mut a0 = vdupq_n_f32(0.);
            let mut a1 = vdupq_n_f32(0.);
            let mut a2 = vdupq_n_f32(0.);
            let mut a3 = vdupq_n_f32(0.);

            for c in 0..4usize {
                let cw = vld1_u8(bytes_ptr.add(8 * c));
                let lo = vand_u8(cw, mask15);
                let hi = vshr_n_u8(cw, 4);
                let codes = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
                let b01 = vcombine_u8(
                    vdup_n_u8(*bytes_ptr.add(32 + 2 * c)),
                    vdup_n_u8(*bytes_ptr.add(33 + 2 * c)),
                );
                let signs = vandq_u8(vshlq_u8(b01, sign_shifts), mask16);
                let packed = vqtbl2q_u8(cbs, vorrq_u8(codes, signs));

                let t8_0 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, mask3), one));
                let lo8_0 = vmovl_s8(vget_low_s8(t8_0));
                let hi8_0 = vmovl_high_s8(t8_0);
                let xs0 = x_ptr.add(c * 64);
                a0 = vfmaq_f32(
                    a0,
                    vld1q_f32(xs0),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_0))),
                );
                a1 = vfmaq_f32(
                    a1,
                    vld1q_f32(xs0.add(4)),
                    vcvtq_f32_s32(vmovl_high_s16(lo8_0)),
                );
                a2 = vfmaq_f32(
                    a2,
                    vld1q_f32(xs0.add(8)),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_0))),
                );
                a3 = vfmaq_f32(
                    a3,
                    vld1q_f32(xs0.add(12)),
                    vcvtq_f32_s32(vmovl_high_s16(hi8_0)),
                );

                let t8_1 =
                    vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 2), mask3), one));
                let lo8_1 = vmovl_s8(vget_low_s8(t8_1));
                let hi8_1 = vmovl_high_s8(t8_1);
                let xs1 = xs0.add(16);
                a0 = vfmaq_f32(
                    a0,
                    vld1q_f32(xs1),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_1))),
                );
                a1 = vfmaq_f32(
                    a1,
                    vld1q_f32(xs1.add(4)),
                    vcvtq_f32_s32(vmovl_high_s16(lo8_1)),
                );
                a2 = vfmaq_f32(
                    a2,
                    vld1q_f32(xs1.add(8)),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_1))),
                );
                a3 = vfmaq_f32(
                    a3,
                    vld1q_f32(xs1.add(12)),
                    vcvtq_f32_s32(vmovl_high_s16(hi8_1)),
                );

                let t8_2 =
                    vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 4), mask3), one));
                let lo8_2 = vmovl_s8(vget_low_s8(t8_2));
                let hi8_2 = vmovl_high_s8(t8_2);
                let xs2 = xs0.add(32);
                a0 = vfmaq_f32(
                    a0,
                    vld1q_f32(xs2),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_2))),
                );
                a1 = vfmaq_f32(
                    a1,
                    vld1q_f32(xs2.add(4)),
                    vcvtq_f32_s32(vmovl_high_s16(lo8_2)),
                );
                a2 = vfmaq_f32(
                    a2,
                    vld1q_f32(xs2.add(8)),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_2))),
                );
                a3 = vfmaq_f32(
                    a3,
                    vld1q_f32(xs2.add(12)),
                    vcvtq_f32_s32(vmovl_high_s16(hi8_2)),
                );

                let t8_3 =
                    vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 6), mask3), one));
                let lo8_3 = vmovl_s8(vget_low_s8(t8_3));
                let hi8_3 = vmovl_high_s8(t8_3);
                let xs3 = xs0.add(48);
                a0 = vfmaq_f32(
                    a0,
                    vld1q_f32(xs3),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_3))),
                );
                a1 = vfmaq_f32(
                    a1,
                    vld1q_f32(xs3.add(4)),
                    vcvtq_f32_s32(vmovl_high_s16(lo8_3)),
                );
                a2 = vfmaq_f32(
                    a2,
                    vld1q_f32(xs3.add(8)),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_3))),
                );
                a3 = vfmaq_f32(
                    a3,
                    vld1q_f32(xs3.add(12)),
                    vcvtq_f32_s32(vmovl_high_s16(hi8_3)),
                );
            }

            let block_sum = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            row_acc = vfmaq_n_f32(row_acc, block_sum, d);

            bytes_ptr = bytes_ptr.add(42);
            x_ptr = x_ptr.add(256);
        }

        vaddvq_f32(row_acc)
    }
}

/// Per-lane digit tables: `STQ_DIGIT_TABLES[p][index]` is the `p`-th ternary
/// digit of codebook entry `index` as `{-1, 0, 1}`, already in byte form. They
/// let one `tbl` produce a whole 16-lane digit vector per `p`, instead of one
/// table lookup plus a shift, mask and subtract per `p`.
#[cfg(target_arch = "aarch64")]
const STQ_DIGIT_TABLES: [[u8; 32]; 4] = digit_tables();

#[cfg(target_arch = "aarch64")]
const fn digit_tables() -> [[u8; 32]; 4] {
    let mut tables = [[0u8; 32]; 4];
    let mut p = 0;
    while p < 4 {
        let mut index = 0;
        while index < 32 {
            tables[p][index] = ((STQ_CODEBOOK[index] >> (2 * p)) & 3).wrapping_sub(1);
            index += 1;
        }
        p += 1;
    }
    tables
}

/// Expand chunk `c` of one 42-byte STQ block into the four 16-lane ternary
/// digit vectors, one per output lane group `p`. Digits are `{-1, 0, 1}`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn stq_digit_lanes(
    bytes: *const u8,
    c: usize,
    tables: &[std::arch::aarch64::uint8x16x2_t; 4],
    sign_shifts: std::arch::aarch64::int8x16_t,
) -> [std::arch::aarch64::int8x16_t; 4] {
    use std::arch::aarch64::*;
    // SAFETY: the caller passes a 42-byte block and `c < 4`; every index stays
    // below 32, so each lookup lands inside its 32-byte table.
    unsafe {
        let mask15 = vdup_n_u8(15);
        let mask16 = vdupq_n_u8(0x10);
        let codes = vld1_u8(bytes.add(8 * c));
        let low = vand_u8(codes, mask15);
        let high = vshr_n_u8(codes, 4);
        let nibbles = vcombine_u8(vzip1_u8(low, high), vzip2_u8(low, high));
        let sign_bits = vcombine_u8(
            vdup_n_u8(*bytes.add(32 + 2 * c)),
            vdup_n_u8(*bytes.add(33 + 2 * c)),
        );
        let signs = vandq_u8(vshlq_u8(sign_bits, sign_shifts), mask16);
        let index = vorrq_u8(nibbles, signs);
        [
            vreinterpretq_s8_u8(vqtbl2q_u8(tables[0], index)),
            vreinterpretq_s8_u8(vqtbl2q_u8(tables[1], index)),
            vreinterpretq_s8_u8(vqtbl2q_u8(tables[2], index)),
            vreinterpretq_s8_u8(vqtbl2q_u8(tables[3], index)),
        ]
    }
}

/// ARMv8.2 SDOT row dot: each instruction accumulates sixteen int8 products,
/// so one 256-weight block costs sixteen SDOTs plus its expansion.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn stq_row_dot_sdot(row_bytes: &[u8], activations: &[i8], scales: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: callers pass whole 42-byte blocks, one activation group and one
    // scale per block, and at least 256 activations per block.
    unsafe {
        let tables: [uint8x16x2_t; 4] = std::array::from_fn(|p| {
            uint8x16x2_t(
                vld1q_u8(STQ_DIGIT_TABLES[p].as_ptr()),
                vld1q_u8(STQ_DIGIT_TABLES[p].as_ptr().add(16)),
            )
        });
        let sign_shifts: [i8; 16] = [4, 3, 2, 1, 0, -1, -2, -3, 4, 3, 2, 1, 0, -1, -2, -3];
        let sign_shifts = vld1q_s8(sign_shifts.as_ptr());
        let mut row_acc = vdupq_n_f32(0.);
        let mut x_ptr = activations.as_ptr();

        for (&scale, block) in scales.iter().zip(row_bytes.as_chunks::<42>().0) {
            let block_scale = half_raw(block.as_ptr().add(40)) * scale;
            let mut acc = vdupq_n_s32(0);
            for c in 0..4usize {
                for (p, digits) in stq_digit_lanes(block.as_ptr(), c, &tables, sign_shifts)
                    .iter()
                    .enumerate()
                {
                    acc = sdot(acc, vld1q_s8(x_ptr.add(c * 64 + p * 16)), *digits);
                }
            }
            row_acc = vfmaq_n_f32(row_acc, vcvtq_f32_s32(acc), block_scale);
            x_ptr = x_ptr.add(256);
        }

        vaddvq_f32(row_acc)
    }
}

/// Fused NEON kernel: gathers codebook bytes with a 32-entry table lookup,
/// extracts each 2-bit lane, and FMA-accumulates against the activations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn stq_dot_neon(bytes: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: callers pass a 42-byte block and exactly 256 activations, so
    // every load and pointer offset stays within both slices.
    unsafe {
        let cbs = uint8x16x2_t(
            vld1q_u8(STQ_CODEBOOK.as_ptr()),
            vld1q_u8(STQ_CODEBOOK.as_ptr().add(16)),
        );
        let sign_shifts: [i8; 16] = [4, 3, 2, 1, 0, -1, -2, -3, 4, 3, 2, 1, 0, -1, -2, -3];
        let sign_shifts = vld1q_s8(sign_shifts.as_ptr());
        let one = vdupq_n_u8(1);
        let mask15 = vdup_n_u8(15);
        let mask16 = vdupq_n_u8(0x10);
        let mask3 = vdupq_n_u8(3);
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
        );
        for c in 0..4usize {
            let cw = vld1_u8(bytes.as_ptr().add(8 * c));
            let lo = vand_u8(cw, mask15);
            let hi = vshr_n_u8(cw, 4);
            let codes = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
            let b01 = vcombine_u8(
                vdup_n_u8(*bytes.as_ptr().add(32 + 2 * c)),
                vdup_n_u8(*bytes.as_ptr().add(33 + 2 * c)),
            );
            let signs = vandq_u8(vshlq_u8(b01, sign_shifts), mask16);
            let packed = vqtbl2q_u8(cbs, vorrq_u8(codes, signs));

            let t8_0 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, mask3), one));
            let lo8_0 = vmovl_s8(vget_low_s8(t8_0));
            let hi8_0 = vmovl_high_s8(t8_0);
            let xs0 = x.as_ptr().add(c * 64);
            a0 = vfmaq_f32(
                a0,
                vld1q_f32(xs0),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_0))),
            );
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(xs0.add(4)),
                vcvtq_f32_s32(vmovl_high_s16(lo8_0)),
            );
            a2 = vfmaq_f32(
                a2,
                vld1q_f32(xs0.add(8)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_0))),
            );
            a3 = vfmaq_f32(
                a3,
                vld1q_f32(xs0.add(12)),
                vcvtq_f32_s32(vmovl_high_s16(hi8_0)),
            );

            let t8_1 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 2), mask3), one));
            let lo8_1 = vmovl_s8(vget_low_s8(t8_1));
            let hi8_1 = vmovl_high_s8(t8_1);
            let xs1 = xs0.add(16);
            a0 = vfmaq_f32(
                a0,
                vld1q_f32(xs1),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_1))),
            );
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(xs1.add(4)),
                vcvtq_f32_s32(vmovl_high_s16(lo8_1)),
            );
            a2 = vfmaq_f32(
                a2,
                vld1q_f32(xs1.add(8)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_1))),
            );
            a3 = vfmaq_f32(
                a3,
                vld1q_f32(xs1.add(12)),
                vcvtq_f32_s32(vmovl_high_s16(hi8_1)),
            );

            let t8_2 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 4), mask3), one));
            let lo8_2 = vmovl_s8(vget_low_s8(t8_2));
            let hi8_2 = vmovl_high_s8(t8_2);
            let xs2 = xs0.add(32);
            a0 = vfmaq_f32(
                a0,
                vld1q_f32(xs2),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_2))),
            );
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(xs2.add(4)),
                vcvtq_f32_s32(vmovl_high_s16(lo8_2)),
            );
            a2 = vfmaq_f32(
                a2,
                vld1q_f32(xs2.add(8)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_2))),
            );
            a3 = vfmaq_f32(
                a3,
                vld1q_f32(xs2.add(12)),
                vcvtq_f32_s32(vmovl_high_s16(hi8_2)),
            );

            let t8_3 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(vshrq_n_u8(packed, 6), mask3), one));
            let lo8_3 = vmovl_s8(vget_low_s8(t8_3));
            let hi8_3 = vmovl_high_s8(t8_3);
            let xs3 = xs0.add(48);
            a0 = vfmaq_f32(
                a0,
                vld1q_f32(xs3),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8_3))),
            );
            a1 = vfmaq_f32(
                a1,
                vld1q_f32(xs3.add(4)),
                vcvtq_f32_s32(vmovl_high_s16(lo8_3)),
            );
            a2 = vfmaq_f32(
                a2,
                vld1q_f32(xs3.add(8)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8_3))),
            );
            a3 = vfmaq_f32(
                a3,
                vld1q_f32(xs3.add(12)),
                vcvtq_f32_s32(vmovl_high_s16(hi8_3)),
            );
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)))
    }
}

#[allow(dead_code)]
type Q6Dot = fn(&[u8], &[f32]) -> f32;

/// Dot one 210-byte Q6_K block (256 weights) against 256 activations,
/// including the block's FP16 scale.
#[allow(dead_code)]
pub(crate) fn q6_dot(bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(bytes.len(), 210);
    debug_assert_eq!(x.len(), 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers pass a 210-byte block and exactly 256 activations.
        unsafe { q6_dot_neon(bytes, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut decoded = [0f32; 256];
        decode_unchecked(DType::Q6_K, bytes, &mut decoded);
        dot(&decoded, x)
    }
}

#[inline(always)]
pub(crate) fn q6_row_dot(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { q6_row_dot_neon(row_bytes, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0;
        for (block, bytes) in row_bytes.chunks_exact(210).enumerate() {
            let xs = &x[block * 256..(block + 1) * 256];
            sum += q6_dot(bytes, xs);
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q6_row_dot_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
        );
        let mask15 = vdupq_n_u8(15);
        let mask3 = vdupq_n_u8(3);
        let c32 = vdupq_n_u8(32);
        let num_blocks = row_bytes.len() / 210;
        let mut bytes_ptr = row_bytes.as_ptr();
        let mut x_ptr = x.as_ptr();

        for _ in 0..num_blocks {
            let d = half_raw(bytes_ptr.add(208));
            for c in 0..2usize {
                let ql = bytes_ptr.add(c * 64);
                let qh = bytes_ptr.add(128 + c * 32);
                let sc = bytes_ptr.add(192 + c * 8);
                for p in 0..4usize {
                    for h in 0..2usize {
                        let qlv = vld1q_u8(ql.add((p & 1) * 32 + h * 16));
                        let qhv = vld1q_u8(qh.add(h * 16));
                        let lo6 = match p {
                            2 | 3 => vandq_u8(vshrq_n_u8(qlv, 4), mask15),
                            _ => vandq_u8(qlv, mask15),
                        };
                        let bits = match p {
                            0 => qhv,
                            1 => vshrq_n_u8(qhv, 2),
                            2 => vshrq_n_u8(qhv, 4),
                            _ => vshrq_n_u8(qhv, 6),
                        };
                        let q6 = vorrq_u8(lo6, vshlq_n_u8(vandq_u8(bits, mask3), 4));
                        let t8 = vreinterpretq_s8_u8(vsubq_u8(q6, c32));
                        let s = vdupq_n_f32(d * (*sc.add(2 * p + h) as i8) as f32);
                        let lo8 = vmovl_s8(vget_low_s8(t8));
                        let hi8 = vmovl_high_s8(t8);
                        let base = x_ptr.add(c * 128 + p * 32 + h * 16);
                        a0 = vfmaq_f32(
                            a0,
                            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8))), s),
                            vld1q_f32(base),
                        );
                        a1 = vfmaq_f32(
                            a1,
                            vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo8)), s),
                            vld1q_f32(base.add(4)),
                        );
                        a2 = vfmaq_f32(
                            a2,
                            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8))), s),
                            vld1q_f32(base.add(8)),
                        );
                        a3 = vfmaq_f32(
                            a3,
                            vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi8)), s),
                            vld1q_f32(base.add(12)),
                        );
                    }
                }
            }
            bytes_ptr = bytes_ptr.add(210);
            x_ptr = x_ptr.add(256);
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)))
    }
}

/// One packed row of `dtype` against quantized activations.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn row_dot_q8(dtype: DType, row: &[u8], q8: &Q8) -> f32 {
    // SAFETY: callers pass whole rows, so every row is a whole number of
    // 256-weight blocks, and `q8` holds one scale and 256 values per block.
    unsafe {
        if dtype == DType::STQ1_0 {
            stq_row_dot_sdot(row, q8.values(), q8.scales())
        } else {
            q6_row_dot_sdot(row, q8.values(), q8.scales())
        }
    }
}

/// ARMv8.2 SDOT row dot for Q6_K. Integer dot products replace the per-lane
/// widening, and each 16-weight plane keeps its own FP16 and int8 scales.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q6_row_dot_sdot(row_bytes: &[u8], activations: &[i8], scales: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: callers pass whole 210-byte blocks, one activation group and one
    // scale per block, and at least 256 activations per block.
    unsafe {
        let mask15 = vdupq_n_u8(15);
        let mask3 = vdupq_n_u8(3);
        let c32 = vdupq_n_s8(32);
        let mut row_acc = vdupq_n_f32(0.);
        let mut x_ptr = activations.as_ptr();

        for (&scale, block) in scales.iter().zip(row_bytes.as_chunks::<210>().0) {
            let d = half_raw(block.as_ptr().add(208)) * scale;
            for c in 0..2usize {
                let ql = block.as_ptr().add(c * 64);
                let qh = block.as_ptr().add(128 + c * 32);
                let sc = block.as_ptr().add(192 + c * 8);
                for p in 0..4usize {
                    for h in 0..2usize {
                        let qlv = vld1q_u8(ql.add((p & 1) * 32 + h * 16));
                        let qhv = vld1q_u8(qh.add(h * 16));
                        let lo6 = match p {
                            2 | 3 => vandq_u8(vshrq_n_u8(qlv, 4), mask15),
                            _ => vandq_u8(qlv, mask15),
                        };
                        let bits = match p {
                            0 => qhv,
                            1 => vshrq_n_u8(qhv, 2),
                            2 => vshrq_n_u8(qhv, 4),
                            _ => vshrq_n_u8(qhv, 6),
                        };
                        let q6 = vorrq_u8(lo6, vshlq_n_u8(vandq_u8(bits, mask3), 4));
                        let digits = vsubq_s8(vreinterpretq_s8_u8(q6), c32);
                        let base = x_ptr.add(c * 128 + p * 32 + h * 16);
                        let acc = sdot(vdupq_n_s32(0), vld1q_s8(base), digits);
                        let plane = d * (*sc.add(2 * p + h) as i8) as f32;
                        row_acc = vfmaq_n_f32(row_acc, vcvtq_f32_s32(acc), plane);
                    }
                }
            }
            x_ptr = x_ptr.add(256);
        }
        vaddvq_f32(row_acc)
    }
}

/// Fused NEON kernel for Q6_K: bit-extract each 6-bit code, apply the plane
/// scale, and FMA-accumulate. Products and accumulation order match
/// `decode_unchecked` followed by `dot`, so results are bit-identical.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q6_dot_neon(bytes: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: callers pass a 210-byte block and exactly 256 activations, so
    // every load and pointer offset stays within both slices.
    unsafe {
        let d = half(&bytes[208..]);
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
            vdupq_n_f32(0.),
        );
        for c in 0..2usize {
            let ql = bytes.as_ptr().add(c * 64);
            let qh = bytes.as_ptr().add(128 + c * 32);
            let sc = bytes.as_ptr().add(192 + c * 8);
            for p in 0..4usize {
                for h in 0..2usize {
                    let qlv = vld1q_u8(ql.add((p & 1) * 32 + h * 16));
                    let qhv = vld1q_u8(qh.add(h * 16));
                    let lo6 = match p {
                        2 | 3 => vandq_u8(vshrq_n_u8(qlv, 4), vdupq_n_u8(15)),
                        _ => vandq_u8(qlv, vdupq_n_u8(15)),
                    };
                    let bits = match p {
                        0 => qhv,
                        1 => vshrq_n_u8(qhv, 2),
                        2 => vshrq_n_u8(qhv, 4),
                        _ => vshrq_n_u8(qhv, 6),
                    };
                    let q6 = vorrq_u8(lo6, vshlq_n_u8(vandq_u8(bits, vdupq_n_u8(3)), 4));
                    let t8 = vreinterpretq_s8_u8(vsubq_u8(q6, vdupq_n_u8(32)));
                    let s = vdupq_n_f32(d * (*sc.add(2 * p + h) as i8) as f32);
                    let lo8 = vmovl_s8(vget_low_s8(t8));
                    let hi8 = vmovl_s8(vget_high_s8(t8));
                    let base = x.as_ptr().add(c * 128 + p * 32 + h * 16);
                    a0 = vfmaq_f32(
                        a0,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8))), s),
                        vld1q_f32(base),
                    );
                    a1 = vfmaq_f32(
                        a1,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo8))), s),
                        vld1q_f32(base.add(4)),
                    );
                    a2 = vfmaq_f32(
                        a2,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8))), s),
                        vld1q_f32(base.add(8)),
                    );
                    a3 = vfmaq_f32(
                        a3,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi8))), s),
                        vld1q_f32(base.add(12)),
                    );
                }
            }
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)))
    }
}

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
                stq_digits(x, d, y);
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
                    let ql = &x[chunk * 64..chunk * 64 + 64];
                    let qh = &x[128 + chunk * 32..128 + chunk * 32 + 32];
                    let sc = &x[192 + chunk * 8..192 + chunk * 8 + 8];
                    for p in 0..4 {
                        let nib = (p >> 1) * 4;
                        let bits = 2 * p;
                        for half in 0..2 {
                            let s = d * (sc[2 * p + half] as i8) as f32;
                            let ql16 = &ql[half * 16 + (p & 1) * 32..][..16];
                            let qh16 = &qh[half * 16..][..16];
                            let out16 = &mut y[chunk * 128 + p * 32 + half * 16..][..16];
                            for ((&qv, &hv), o) in ql16.iter().zip(qh16).zip(out16) {
                                let q = ((qv >> nib) & 15) | (((hv >> bits) & 3) << 4);
                                *o = s * (q as i32 - 32) as f32;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return "neon";
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return "avx2";
    }
    "scalar"
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: on aarch64, NEON is always available; lengths match.
        unsafe { dot_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_avx2(a, b) };
        }
        dot_scalar(a, b)
    }
}

pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let (mut s0, mut s1, mut s2, mut s3) = (
        vdupq_n_f32(0.),
        vdupq_n_f32(0.),
        vdupq_n_f32(0.),
        vdupq_n_f32(0.),
    );
    let chunks = a.len() / 16;
    let mut ap = a.as_ptr();
    let mut bp = b.as_ptr();
    for _ in 0..chunks {
        // SAFETY: all sixteen-element loads stay within the equal-length slices.
        unsafe {
            s0 = vfmaq_f32(s0, vld1q_f32(ap), vld1q_f32(bp));
            s1 = vfmaq_f32(s1, vld1q_f32(ap.add(4)), vld1q_f32(bp.add(4)));
            s2 = vfmaq_f32(s2, vld1q_f32(ap.add(8)), vld1q_f32(bp.add(8)));
            s3 = vfmaq_f32(s3, vld1q_f32(ap.add(12)), vld1q_f32(bp.add(12)));
            ap = ap.add(16);
            bp = bp.add(16);
        }
    }
    let mut sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    let simd_end = chunks * 16;
    let vec_end = a.len() / 4 * 4;
    for i in (simd_end..vec_end).step_by(4) {
        // SAFETY: the four-element loads are within the equal-length slices.
        unsafe {
            sum = vfmaq_f32(
                sum,
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
        }
    }
    vaddvq_f32(sum) + dot_scalar(&a[vec_end..], &b[vec_end..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let chunks = a.len() / 16;
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    for i in 0..chunks {
        let i = i * 16;
        // SAFETY: all sixteen-element unaligned loads stay within both slices.
        unsafe {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), s0);
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(i + 8)),
                _mm256_loadu_ps(bp.add(i + 8)),
                s1,
            );
        }
    }
    let mut sum = _mm256_add_ps(s0, s1);
    let simd_end = chunks * 16;
    let vec_end = a.len() / 8 * 8;
    for i in (simd_end..vec_end).step_by(8) {
        // SAFETY: the eight-element unaligned loads stay within both slices.
        unsafe {
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), sum);
        }
    }
    let mut lanes = [0f32; 8];
    // SAFETY: lanes has space for the full vector.
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    }
    lanes.iter().sum::<f32>() + dot_scalar(&a[vec_end..], &b[vec_end..])
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
    fn fused_stq_dot_matches_decode() {
        for seed in 0..64u32 {
            let mut block = [0u8; 42];
            for (i, slot) in block.iter_mut().enumerate().take(40) {
                *slot = (seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 7)) as u8;
            }
            block[40..].copy_from_slice(&f16::from_f32(1. + seed as f32 % 0.5).to_le_bytes());
            let x: Vec<_> = (0..256)
                .map(|i| ((i as f32 + seed as f32) * 0.31).cos())
                .collect();
            let mut digits = [0f32; 256];
            stq_digits(&block, half(&block[40..]), &mut digits);
            let expected = dot(&digits, &x);
            assert!((stq_dot(&block, &x) - expected).abs() < 1e-3, "seed {seed}");
        }
    }

    #[test]
    fn fused_q6_dot_matches_decode() {
        for seed in 0..64u32 {
            let mut block = [0u8; 210];
            for (i, slot) in block.iter_mut().enumerate().take(208) {
                *slot = (seed.wrapping_mul(40503).wrapping_add(i as u32 * 13)) as u8;
            }
            block[208..].copy_from_slice(&f16::from_f32(0.25 + seed as f32 % 0.75).to_le_bytes());
            let x: Vec<_> = (0..256)
                .map(|i| ((i as f32 + seed as f32) * 0.17).sin())
                .collect();
            let mut decoded = [0f32; 256];
            decode_unchecked(DType::Q6_K, &block, &mut decoded);
            let expected = dot(&decoded, &x);
            assert!((q6_dot(&block, &x) - expected).abs() < 1e-3, "seed {seed}");
        }
    }

    /// Kernel-level throughput probe for the batch-1 GEMV path. Needs a local
    /// 1.25Bit GGUF:
    /// `cargo test --release --lib -- --ignored --nocapture kernel_throughput`.
    /// Set `HY_MT_BENCH_MODEL` to use another model file.
    #[test]
    #[ignore = "requires a local model file"]
    fn kernel_throughput() -> Result<()> {
        use crate::gguf::Gguf;
        use std::time::Instant;

        let path = std::env::var("HY_MT_BENCH_MODEL").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/models/Hy-MT2-1.8B-1.25Bit.gguf"
            )
            .into()
        });
        let gguf = Gguf::open(&path, Profile::Auto)?;

        let report = |label: &str, rows: usize, elements: usize, bytes: usize, secs: f64| {
            println!(
                "{label:<26} {rows:>7} rows {:>7.1} ns/row {:>7.2} Gweight/s {:>7.2} GB/s",
                secs * 1e9 / rows as f64,
                elements as f64 / secs / 1e9,
                bytes as f64 / secs / 1e9,
            );
        };

        let time = |mut body: Box<dyn FnMut() + '_>| -> f64 {
            body();
            body();
            let start = Instant::now();
            body();
            start.elapsed().as_secs_f64()
        };

        // Single-thread row kernels: the shipping integer path against the F32
        // fallback, over the three shapes one decode step touches.
        let shapes = [
            ("blk.0.ffn_gate.weight", "stq ffn_gate"),
            ("blk.0.attn_q.weight", "stq attn_q"),
            ("blk.0.ffn_down.weight", "stq ffn_down"),
            ("token_embd.weight", "q6 lm_head"),
        ];
        for (name, label) in shapes {
            let weight = gguf.tensor(name)?;
            let (rows, cols) = (weight.shape()[1], weight.shape()[0]);
            let row_bytes = cols / 256 * weight.dtype().block_bytes();
            let bytes = weight.bytes();
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
            let fallback: fn(&[u8], &[f32]) -> f32 = if weight.dtype() == DType::STQ1_0 {
                stq_row_dot
            } else {
                q6_row_dot
            };
            let secs = time(Box::new(|| {
                let mut acc = 0f32;
                for row in bytes.chunks_exact(row_bytes) {
                    acc += fallback(row, &x);
                }
                std::hint::black_box(acc);
            }));
            report(
                &format!("{label} f32"),
                rows,
                rows * cols,
                rows * row_bytes,
                secs,
            );

            #[cfg(target_arch = "aarch64")]
            if sdot_available() {
                let q8 = Q8::quantize(&x);
                let dtype = weight.dtype();
                let secs = time(Box::new(|| {
                    let mut acc = 0f32;
                    for row in bytes.chunks_exact(row_bytes) {
                        // SAFETY: whole rows against whole activation groups.
                        acc += unsafe { row_dot_q8(dtype, row, &q8) };
                    }
                    std::hint::black_box(acc);
                }));
                report(
                    &format!("{label} sdot"),
                    rows,
                    rows * cols,
                    rows * row_bytes,
                    secs,
                );
            }
        }

        // The fused projection group must match three separate products.
        let (q, k, v) = (
            gguf.tensor("blk.0.attn_q.weight")?,
            gguf.tensor("blk.0.attn_k.weight")?,
            gguf.tensor("blk.0.attn_v.weight")?,
        );
        let input: Vec<f32> = (0..q.shape()[0])
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();
        let (fused_q, fused_k, fused_v) = Weight::gemv_qkv(&q, &k, &v, &input)?;
        for (label, fused, separate) in [
            ("q", fused_q, q.matmul(&input, 1)?),
            ("k", fused_k, k.matmul(&input, 1)?),
            ("v", fused_v, v.matmul(&input, 1)?),
        ] {
            assert_eq!(fused, separate, "fused {label} projection differs");
        }

        // F32 attention-style dot, 128-wide keys.
        for len in [128usize, 2048] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.3).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.7).cos()).collect();
            let reps = 20_000_000 / len;
            let secs = time(Box::new(|| {
                let mut acc = 0f32;
                for _ in 0..reps {
                    acc += dot(&a, &b);
                }
                std::hint::black_box(acc);
            }));
            report(
                &format!("dot f32 len={len}"),
                reps,
                reps * len,
                reps * len * 8,
                secs,
            );
        }

        // Every decoder weight one decode step touches, through the shipping
        // `matmul` entry point at several pool sizes.
        let mut decoder: Vec<Weight> = gguf
            .tensors
            .keys()
            .filter(|name| name.starts_with("blk.") && name.ends_with(".weight"))
            .map(|name| gguf.tensor(name))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|weight| weight.dtype() == DType::STQ1_0)
            .collect();
        decoder.sort_by_key(|weight| weight.name().to_string());
        let decoder_bytes: usize = decoder.iter().map(|weight| weight.bytes().len()).sum();
        let decoder_elements: usize = decoder
            .iter()
            .map(|weight| weight.shape()[0] * weight.shape()[1])
            .sum();
        let activations: std::collections::HashMap<usize, Vec<f32>> = decoder
            .iter()
            .map(|weight| weight.shape()[0])
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .map(|cols| {
                (
                    cols,
                    (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect(),
                )
            })
            .collect();
        for threads in [1usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .start_handler(|_| pin_to_performance_cores())
                .build()?;
            let secs = time(Box::new(|| {
                let mut acc = 0f32;
                pool.install(|| {
                    for weight in &decoder {
                        let out = weight
                            .matmul(&activations[&weight.shape()[0]], 1)
                            .expect("gemv");
                        acc += out[0];
                    }
                });
                std::hint::black_box(acc);
            }));
            report(
                &format!("decoder gemv {threads}t"),
                decoder.len(),
                decoder_elements,
                decoder_bytes,
                secs,
            );

            // The same rows in a single parallel region: the difference
            // against the loop above is one rayon region per `matmul` call.
            #[cfg(target_arch = "aarch64")]
            if sdot_available() {
                let flat: Vec<(&[u8], DType, usize)> = decoder
                    .iter()
                    .flat_map(|weight| {
                        let cols = weight.shape()[0];
                        let dtype = weight.dtype();
                        let row_bytes = cols / 256 * dtype.block_bytes();
                        weight
                            .bytes()
                            .chunks_exact(row_bytes)
                            .map(move |row| (row, dtype, cols))
                    })
                    .collect();
                let quantized: std::collections::HashMap<usize, Q8> = activations
                    .iter()
                    .map(|(cols, values)| (*cols, Q8::quantize(values)))
                    .collect();
                let secs = time(Box::new(|| {
                    let acc: f32 = pool.install(|| {
                        flat.par_iter()
                            .map(|(row, dtype, cols)| {
                                // SAFETY: whole rows against whole activation groups.
                                unsafe { row_dot_q8(*dtype, row, &quantized[cols]) }
                            })
                            .sum()
                    });
                    std::hint::black_box(acc);
                }));
                report(
                    &format!("decoder flat {threads}t"),
                    decoder.len(),
                    decoder_elements,
                    decoder_bytes,
                    secs,
                );
            }
        }
        Ok(())
    }

    fn pin_to_performance_cores() {
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
            }
            // SAFETY: only this worker thread's QoS class changes.
            unsafe {
                pthread_set_qos_class_self_np(0x21, 0);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn int8_activation_groups_round_trip() {
        let values: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.37).sin() * 3.).collect();
        let q8 = Q8::quantize(&values);
        assert_eq!(q8.values().len(), values.len());
        assert_eq!(q8.scales().len(), 4);
        for (group, scale) in q8.scales().iter().enumerate() {
            for (offset, &value) in values[group * Q8_GROUP..(group + 1) * Q8_GROUP]
                .iter()
                .enumerate()
            {
                let restored = q8.values()[group * Q8_GROUP + offset] as f32 * scale;
                assert!(
                    (restored - value).abs() <= scale * 0.5 + 1e-6,
                    "group {group} offset {offset}: {restored} vs {value}"
                );
            }
        }
        // An all-zero group must not divide by zero.
        let q8 = Q8::quantize(&vec![0.; 256]);
        assert_eq!(q8.scales(), &[0.]);
        assert!(q8.values().iter().all(|&v| v == 0));
    }

    /// SDOT kernels must agree with the exact decode of the same activations.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sdot_kernels_match_int8_reference() {
        if !sdot_available() {
            return;
        }
        for seed in 0..8u32 {
            let mut x: Vec<f32> = (0..512)
                .map(|i| ((i as f32 + seed as f32) * 0.11).sin() * 2.)
                .collect();
            // Give each 256-element group a different dynamic range.
            for (group, values) in x.chunks_mut(Q8_GROUP).enumerate() {
                for value in values {
                    *value *= 1. + group as f32;
                }
            }
            let q8 = Q8::quantize(&x);

            let mut stq = vec![0u8; 84];
            for (i, slot) in stq.iter_mut().enumerate() {
                *slot = (seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 7)) as u8;
            }
            stq[40..42].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
            stq[82..84].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
            let mut stq_weights = vec![0.; 512];
            decode(DType::STQ1_0, &stq, &mut stq_weights).unwrap();
            let expected: f64 = stq_weights
                .iter()
                .enumerate()
                .map(|(i, w)| *w as f64 * q8.values()[i] as f64 * q8.scales()[i / Q8_GROUP] as f64)
                .sum();
            // SAFETY: the row is two whole 42-byte blocks and `q8` covers them.
            let got = unsafe { stq_row_dot_sdot(&stq, q8.values(), q8.scales()) };
            let magnitude: f64 = stq_weights
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    (*w as f64 * q8.values()[i] as f64 * q8.scales()[i / Q8_GROUP] as f64).abs()
                })
                .sum();
            assert!(
                (got as f64 - expected).abs() <= magnitude * 1e-5,
                "seed {seed}: sdot {got} vs reference {expected}"
            );

            let mut q6 = vec![0u8; 420];
            for (i, slot) in q6.iter_mut().enumerate() {
                *slot = (seed.wrapping_mul(40503).wrapping_add(i as u32 * 13)) as u8;
            }
            q6[208..210].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
            q6[418..420].copy_from_slice(&f16::from_f32(0.75).to_le_bytes());
            let mut q6_weights = vec![0.; 512];
            decode(DType::Q6_K, &q6, &mut q6_weights).unwrap();
            let expected: f64 = q6_weights
                .iter()
                .enumerate()
                .map(|(i, w)| *w as f64 * q8.values()[i] as f64 * q8.scales()[i / Q8_GROUP] as f64)
                .sum();
            let magnitude: f64 = q6_weights
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    (*w as f64 * q8.values()[i] as f64 * q8.scales()[i / Q8_GROUP] as f64).abs()
                })
                .sum();
            // SAFETY: the row is two whole 210-byte blocks and `q8` covers them.
            let got = unsafe { q6_row_dot_sdot(&q6, q8.values(), q8.scales()) };
            assert!(
                (got as f64 - expected).abs() <= magnitude * 1e-5,
                "seed {seed}: sdot {got} vs reference {expected}"
            );
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
